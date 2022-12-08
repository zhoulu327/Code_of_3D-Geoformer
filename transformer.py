import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import h5py


class Geoformer(nn.Module):
    def __init__(self, mypara):
        super().__init__()
        self.mypara = mypara
        d_size = mypara.d_size
        self.device = mypara.device
        if self.mypara.needtauxy:
            self.cube_dim = (
                (mypara.input_channal + 2)
                * mypara.patch_size[0]
                * mypara.patch_size[1]
            )  # C*h0*w0
        else:
            self.cube_dim = (
                mypara.input_channal * mypara.patch_size[0] * mypara.patch_size[1]
            )
        self.predictor_emb = make_embedding(
            cube_dim=self.cube_dim,
            d_size=d_size,
            emb_spatial_size=mypara.emb_spatial_size,
            max_len=mypara.input_length,
            device=self.device,
        )
        self.tgt_emb = make_embedding(
            cube_dim=self.cube_dim,
            d_size=d_size,
            emb_spatial_size=mypara.emb_spatial_size,
            max_len=mypara.output_length,
            device=self.device,
        )
        encoder_layer = EncoderLayer(
            d_size, mypara.nheads, mypara.dim_feedforward, mypara.dropout
        )
        decoder_layer = DecoderLayer(
            d_size, mypara.nheads, mypara.dim_feedforward, mypara.dropout
        )
        self.encoder = Encoder(
            encoder_layer=encoder_layer, num_layers=mypara.num_encoder_layers
        )
        self.decoder = Decoder(
            decoder_layer=decoder_layer, num_layers=mypara.num_decoder_layers
        )
        # 将解码端输出的d_model恢复到原始大小input_dim
        self.linear_output = nn.Linear(d_size, self.cube_dim)

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        memory_mask=None,
        train=True,
        ssr_ratio=0,
    ):
        """
        Args:
            src: (batch, lb, C, H, W)
            tgt: (batch, T_tgt, C, H, W),T_tgt is 1 during test and 20 during train
            src_mask: (lb, lb)
            tgt_mask: (T_tgt, T_tgt)
            memory_mask: (T_tgt, lb)
        Returns:
            outvar_pred: (batch, T_tgt, C, H, W)
        """
        memory = self.encode(src=src, src_mask=src_mask)
        # memory:[batch,S,lb,d_size]
        if train:
            with torch.no_grad():
                tgt2 = torch.cat(
                    [src[:, -1:], tgt[:, :-1]], dim=1
                )  # (batch,T_tgt,C,H,W)
                tgt_mask = self.generate_square_subsequent_mask(tgt2.size(1))
                outvar_pred = self.decode(
                    tgt2,
                    memory,
                    tgt_mask,
                    memory_mask,
                )  # out: (batch,T_tgt,C,H,W)
            if ssr_ratio > 1e-6:
                teacher_forcing_mask = torch.bernoulli(
                    ssr_ratio * torch.ones(tgt.size(0), tgt.size(1) - 1, 1, 1, 1)
                ).to(
                    self.device
                )  # [batch,T_tgt-1,1,1,1]
            # torch.bernoulli:根据输入矩阵内的数值（0-1之间），由贝努力概率随机选择输出0或者1
            else:
                teacher_forcing_mask = 0
            # 概率融合
            tgt = (
                teacher_forcing_mask * tgt[:, :-1]
                + (1 - teacher_forcing_mask) * outvar_pred[:, :-1]
            )  # [batch,T_tgt-1,C,H,W]
            tgt = torch.cat([src[:, -1:], tgt], dim=1)
            # src的最后一个月src[:,-1:]+前T_tgt-1个月的tgt值，而此tgt值可能是真值也可能是decode输出值
            # 最后的tgt:[batch,T_tgt,C,H,W]
            # decode预测
            outvar_pred = self.decode(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
            )
        else:
            assert tgt is None
            tgt = src[:, -1:]  # use last src as the input during test
            for t in range(self.mypara.output_length):
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
                outvar_pred = self.decode(
                    tgt,
                    memory,
                    tgt_mask,
                    memory_mask,
                )
                tgt = torch.cat([tgt, outvar_pred[:, -1:]], dim=1)
        return outvar_pred

    # ---------------------------------------------------
    # encode:切分为patches->编码->召唤encoder
    # encoder: Encoder类的实例，定义编码器层数
    # Encoder: encoder_layer的串联器
    # encoder_layer: EncoderLayer的实例
    # EncoderLayer：实现T-S attention,残差链接、归一化、前馈层、残差连接、归一化

    # decode: 切分为patch->编码->召唤decoder->实现解码器最后的linear层并将output恢复到原始大小
    # decoder: Decoder的实例
    # Decoder：串联decoder_layer
    # decoder_layer：DecoderLayer的实例
    # DecoderLayer：实现解码端的除最后linear的其他功能
    # ------------------------------------------------------
    def encode(self, src, src_mask):
        """
        切分成patch->编码->召唤encoder
        Args:
            src: (Batch, lb, C, H, W)
            src_mask: (lb, lb)
        Returns:
            memory: (Batch, S, lb, d_size)
        """
        lb = src.size(1)
        src = unfold_StackOverChannel(
            src, self.mypara.patch_size
        )  # output:[batch, lb, C*k0*k1, H', W']
        # src = self.unfold_StackOverChannelCNN(src)
        src = src.reshape(src.size(0), lb, self.cube_dim, -1).permute(0, 3, 1, 2)
        # [B,lb,C*k0*k1,H',W']->[B,lb,C*k0*k1,H'*W']->(B,S=H'*W',lb,C*k0*k1)

        src = self.predictor_emb(src)  # patch线性层编码并嵌入时间和位置信息 out: [batch,S,lb,d_size]

        memory = self.encoder(src, src_mask)
        # in:[batch, H'*W', lb, d_size]
        # out:[batch, H'*W', lb, d_size]
        return memory

    def decode(self, tgt, memory, tgt_mask, memory_mask):
        """
        Args:
            tgt: (batch, T_tgt, C, H, W)
            memory: (batch, S, lb, d_size)
            tgt_mask: (T_tgt, T_tgt)
            memory_mask: (T_tgt, lb)
        Returns:
            (batch, T_tgt, C, H, W)
        """
        H, W = tgt.size()[-2:]
        T = tgt.size(1)
        tgt = unfold_StackOverChannel(tgt, self.mypara.patch_size)
        # output:[B,T_tgt,C*h0*w0,H',W']
        tgt = tgt.reshape(tgt.size(0), T, self.cube_dim, -1).permute(
            0, 3, 1, 2
        )  # (batch,S,T_tgt,C*h0*w0)
        tgt = self.tgt_emb(tgt)  # [B,S,T,d_size]
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        output = self.linear_output(output).permute(0, 2, 3, 1)
        # (B, T_tgt, cube_dim, S)
        output = output.reshape(
            tgt.size(0),
            T,
            self.cube_dim,
            H // self.mypara.patch_size[0],
            W // self.mypara.patch_size[1],
        )  # (B, T_tgt, cube_dim, H', W')
        output = fold_tensor(
            output, output_size=(H, W), kernel_size=self.mypara.patch_size
        )  # (B, T_tgt, C, H, W)
        return output

    def generate_square_subsequent_mask(self, sz: int):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf')
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 0).T
        # 返回矩阵上三角部分为false其余为true的方阵的转置矩阵！
        return mask.to(self.mypara.device)


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])

    def forward(self, x, memory, tgt_mask, memory_mask):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_size, nheads, dim_feedforward, dropout):
        super().__init__()
        self.sublayer = clones(SublayerConnection(size=d_size, dropout=dropout), 2)
        self.time_attn = MultiHeadedAttention(d_size, nheads, TimeAttention, dropout)
        self.space_attn = MultiHeadedAttention(d_size, nheads, SpaceAttention, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_size),
        )

    def divided_space_time_attn(self, query, key, value, mask):
        """
        Apply space and time attention sequentially
        Args:
            query (batch, S, T, d_size)
            key (batch, S, T, d_size)
            value (batch, S, T, d_size)
        Returns:
            (batch, S, T, d_size)
        """
        m = self.time_attn(query, key, value, mask)
        # m: [B,S,T,d_size]
        return self.space_attn(m, m, m, mask)

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.divided_space_time_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self, d_size, nheads, dim_feedforward, dropout):
        super().__init__()
        self.sublayer = clones(SublayerConnection(d_size, dropout), 3)
        self.encoder_attn = MultiHeadedAttention(
            d_size, nheads, TimeAttention, dropout
        )
        self.time_attn = MultiHeadedAttention(d_size, nheads, TimeAttention, dropout)
        self.space_attn = MultiHeadedAttention(d_size, nheads, SpaceAttention, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_size),
        )

    def divided_space_time_attn(self, query, key, value, mask=None):
        """
        Apply time and space attention sequentially
        Args:
            query (N, S, T_q, D)
            key (N, S, T, D)
            value (N, S, T, D)
        Returns:
            (N, S, T_q, D)
        """
        m = self.time_attn(query, key, value, mask)
        return self.space_attn(m, m, m, mask)

    def forward(self, x, memory, tgt_mask, memory_mask):
        x = self.sublayer[0](
            x, lambda x: self.divided_space_time_attn(x, x, x, tgt_mask)
        )
        x = self.sublayer[1](
            x, lambda x: self.encoder_attn(x, memory, memory, memory_mask)
        )
        return self.sublayer[2](x, self.feed_forward)


def unfold_StackOverChannel(img, kernel_size):
    """
    divide the original image to patches, then stack the grids in each patch along the channels
    Args:
        img (Batch, lb, C, H, W): the last two dimensions must be the spatial dimension
        kernel_size: tuple of length 2
    Returns:
        output (Batch, lb, C*H_k*N_k, H_output, W_output)
    """
    n_dim = len(img.size())
    assert n_dim == 4 or n_dim == 5
    # 切割原图
    # torch.unfold的作用就是手动实现的滑动窗口操作，也就是只有卷，没有积
    pt = img.unfold(-2, size=kernel_size[0], step=kernel_size[0])
    pt = pt.unfold(-2, size=kernel_size[1], step=kernel_size[1]).flatten(-2)
    # pt: (Batch, lb, C, n0, n1, k0*k1) n0,n1是原图利用k0,k1和相应步长分割后的尺寸,如21/3=7,40/2=20
    # 调换维度，并融合同一block不同通道信息
    if n_dim == 4:  # (Batch, C, H, W)
        pt = pt.permute(0, 1, 4, 2, 3).flatten(1, 2)
    elif n_dim == 5:  # (batch, lb, C, H, W)
        pt = pt.permute(0, 1, 2, 5, 3, 4).flatten(2, 3)
        # pt:[batch,lb,C,n0,n1,k0*k1]-->[batch,lb,C,k0*k1,n0,n1]-->[batch,lb,C*k0*k1,n0,n1]
    assert pt.size(-3) == img.size(-3) * kernel_size[0] * kernel_size[1]
    return pt


def fold_tensor(tensor, output_size, kernel_size):
    """
    reconstruct the image from its non-overlapping patches
    Args:
        input tensor of size (N, *, C*k_h*k_w, n_h, n_w)
        output_size of size(H, W), the size of the original image to be reconstructed
        kernel_size: (k_h, k_w)
        stride is usually equal to kernel_size for non-overlapping sliding window
    Returns:
        (N, *, C, H=n_h*k_h, W=n_w*k_w)
    """
    tensor = tensor.float()
    n_dim = len(tensor.size())
    assert n_dim == 4 or n_dim == 5
    f = tensor.flatten(0, 1) if n_dim == 5 else tensor
    folded = F.fold(
        f.flatten(-2),
        output_size=output_size,
        kernel_size=kernel_size,
        stride=kernel_size,
    )
    if n_dim == 5:
        folded = folded.reshape(tensor.size(0), tensor.size(1), *folded.size()[1:])
    return folded


def clones(in_module, N):
    return nn.ModuleList([copy.deepcopy(in_module) for _ in range(N)])


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class make_embedding(nn.Module):
    def __init__(
        self,
        cube_dim,
        d_size,
        emb_spatial_size,
        max_len,
        device,
    ):
        """
        :param cube_dim: The number of grids in one patch cube
        :param d_size: the embedding length
        :param emb_spatial_size:The number of patches decomposed in a field, S
        :param max_len: look back or prediction length, T
        :param device: CPU or GPU
        """
        super().__init__()
        # 1. position embedding
        pe = torch.zeros(max_len, d_size)  # [T,d_size]
        position = torch.arange(0, max_len).unsqueeze(1)  # [T,1]
        div_term = torch.exp(
            torch.arange(0, d_size, 2) * -(np.log(10000.0) / d_size)
        )  # [d_size/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe_time = pe[None, None].to(device)  # (1, 1, T, d_size)
        # 2. spatial embedding：加一个可学习的位置特征在原特征上
        self.spatial_pos = torch.arange(emb_spatial_size)[None, :, None].to(
            device
        )  # (1,S,1)
        self.emb_space = nn.Embedding(emb_spatial_size, d_size)
        # torch.nn.Embedding(num_emb,emb_dim):创建一个词嵌入模型，
        # num_emb:一共有多少个词, emb_dim:想要为每个词创建一个多少维的向量来表示它
        # 在这里，将每一个切割的小块看作一个词，对其进行编码
        # 3. 每个patch展平以及线性层编码成不同特征
        self.linear = nn.Linear(cube_dim, d_size)
        # [batch,S=H'*W',T,cube_dim]-->[batch,S,T,d_size]
        # 4. Normalization
        self.norm = nn.LayerNorm(d_size)

    def forward(self, x):
        """
        Add temporal encoding and learnable spatial embedding
        to the input (after patch and reshape)
        Args:
             x  [batch, S, T, cube_dim]
                S: (H/patch_size[0])*(W/patch_size[1])
                cube_dim:channal*patch_size[0]*patch_size[1]
        Returns:
            embedded array [batch, S, T, d_size]
        """
        assert len(x.size()) == 4
        # 加一个可学习的位置特征在原特征上
        # spatial_pos:[1,S,1]-->embedded_space:[1, S, 1, d_mdoel]
        embedded_space = self.emb_space(self.spatial_pos)
        # 每个patch线性层编码并加上时间和位置编码信息
        x = self.linear(x) + self.pe_time[:, :, : x.size(2)] + embedded_space
        # 最后归一化
        return self.norm(x)


def TimeAttention(query, key, value, mask=None, dropout=None):
    """
    attention over the time axis
    Args:
        query, key, value: linearly-transformed query, key, value (N, h, S, T, d_k)
        mask: of size (T (query), T (key)) specifying locations (which key) the query can and cannot attend to
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(
        d_k
    )  # (batch, head, S, T, T)
    if mask is not None:
        assert mask.dtype == torch.bool
        assert len(mask.size()) == 2
        # masked_fill: 将mask中为1的元素所在的索引，在原矩阵中相同的的索引处替换为指定值
        scores = scores.masked_fill(mask[None, None, None], float("-inf"))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # return：(batch, head, S, T, d_k)
    return torch.matmul(p_attn, value)


def SpaceAttention(query, key, value, mask=None, dropout=None):
    """
    attention over the two space axes
    Args:
        query, key, value: linearly-transformed query, key, value [batch, head, S, T, d_k]  head*d_k=d_size
        mask: None (space attention does not need mask), this argument is intentionally set for consistency
    """
    d_k = query.size(-1)
    query = query.transpose(2, 3)  # (batch, head, T, S, d_k)
    key = key.transpose(2, 3)  # (batch, h, T, S, d_k)
    value = value.transpose(2, 3)  # (batch, h, T, S, d_k)
    # torch.matmul:当输入有多维时，把多出的维作为batch提出来，其他部分做矩阵乘法
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(
        d_k
    )  # (batch, head, T, S, S)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # return  (batch, h, S, T, d_k)
    return torch.matmul(p_attn, value).transpose(2, 3)


class MultiHeadedAttention(nn.Module):
    # 实现Q,K,V通过linear层分头--进行注意力计算--结果重组形状
    def __init__(self, d_size, nheads, attn, dropout):
        super().__init__()
        assert d_size % nheads == 0
        self.d_k = d_size // nheads
        self.nheads = nheads
        self.linears = nn.ModuleList([nn.Linear(d_size, d_size) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)
        self.attn = attn

    def forward(self, query, key, value, mask=None):
        """
        Transform the query, key, value into different heads, then apply the attention in parallel
        Args:
            query, key, value: size (batch, S, T, d_size)
        Returns:
            (batch, S, T, d_size)
        """
        nbatches = query.size(0)
        nspace = query.size(1)
        ntime = query.size(2)

        # view相当于reshape函数，zip是一个迭代器
        # 此步骤是将X,X,X经过线性映射层处理并重组形状得到用于计算多头atten的Q,K,V矩阵
        # output:(batch, nheads, S, T, d_k)  d_k*nheads=d_size
        query, key, value = [
            l(x)
            .view(x.size(0), x.size(1), x.size(2), self.nheads, self.d_k)
            .permute(0, 3, 1, 2, 4)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # out: x:[batch, head, S:192, T:12, d_k:64)
        x = self.attn(query, key, value, mask=mask, dropout=self.dropout)

        # configuous:把tensor变成在内存中连续分布的形式,因为view只能用在contiguous的variable上
        # 如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy
        # [B,head,S,T,d_k]-->[B,S,T,head,d_k]-->[B,S,T,head*d_k=d_size]
        x = (
            x.permute(0, 2, 3, 1, 4)
            .contiguous()
            .view(nbatches, nspace, ntime, self.nheads * self.d_k)
        )  # out: [B,S,T,d_size]
        return self.linears[-1](x)
