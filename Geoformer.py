import torch
import torch.nn as nn
from my_tools import make_embedding, unfold_func, miniEncoder, miniDecoder, fold_func


class Geoformer(nn.Module):
    def __init__(self, mypara):
        super().__init__()
        self.mypara = mypara
        d_size = mypara.d_size
        self.device = mypara.device
        if self.mypara.needtauxy:
            self.cube_dim = (
                (mypara.input_channal + 2) * mypara.patch_size[0] * mypara.patch_size[1]
            )
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
        self.predictand_emb = make_embedding(
            cube_dim=self.cube_dim,
            d_size=d_size,
            emb_spatial_size=mypara.emb_spatial_size,
            max_len=mypara.output_length,
            device=self.device,
        )
        enc_layer = miniEncoder(
            d_size, mypara.nheads, mypara.dim_feedforward, mypara.dropout
        )
        dec_layer = miniDecoder(
            d_size, mypara.nheads, mypara.dim_feedforward, mypara.dropout
        )
        self.encoder = multi_enc_layer(
            enc_layer=enc_layer, num_layers=mypara.num_encoder_layers
        )
        self.decoder = multi_dec_layer(
            dec_layer=dec_layer, num_layers=mypara.num_decoder_layers
        )
        self.linear_output = nn.Linear(d_size, self.cube_dim)

    def forward(
        self,
        predictor,
        predictand,
        in_mask=None,
        enout_mask=None,
        train=True,
        sv_ratio=0,
    ):
        """
        Args:
            predictor: (batch, lb, C, H, W)
            predictand: (batch, pre_len, C, H, W)
        Returns:
            outvar_pred: (batch, pre_len, C, H, W)
        """
        en_out = self.encode(predictor=predictor, in_mask=in_mask)
        if train:
            with torch.no_grad():
                connect_inout = torch.cat(
                    [predictor[:, -1:], predictand[:, :-1]], dim=1
                )
                out_mask = self.make_mask_matrix(connect_inout.size(1))
                outvar_pred = self.decode(
                    connect_inout,
                    en_out,
                    out_mask,
                    enout_mask,
                )
            if sv_ratio > 1e-7:
                supervise_mask = torch.bernoulli(
                    sv_ratio
                    * torch.ones(predictand.size(0), predictand.size(1) - 1, 1, 1, 1)
                ).to(self.device)
            else:
                supervise_mask = 0
            predictand = (
                supervise_mask * predictand[:, :-1]
                + (1 - supervise_mask) * outvar_pred[:, :-1]
            )
            predictand = torch.cat([predictor[:, -1:], predictand], dim=1)
            # predicting
            outvar_pred = self.decode(
                predictand,
                en_out,
                out_mask,
                enout_mask,
            )
        else:
            assert predictand is None
            predictand = predictor[:, -1:]
            for t in range(self.mypara.output_length):
                out_mask = self.make_mask_matrix(predictand.size(1))
                outvar_pred = self.decode(
                    predictand,
                    en_out,
                    out_mask,
                    enout_mask,
                )
                predictand = torch.cat([predictand, outvar_pred[:, -1:]], dim=1)
        return outvar_pred

    def encode(self, predictor, in_mask):
        """
        predictor: (B, lb, C, H, W)
        en_out: (Batch, S, lb, d_size)
        """
        lb = predictor.size(1)
        predictor = unfold_func(predictor, self.mypara.patch_size)
        predictor = predictor.reshape(predictor.size(0), lb, self.cube_dim, -1).permute(
            0, 3, 1, 2
        )
        predictor = self.predictor_emb(predictor)
        en_out = self.encoder(predictor, in_mask)
        return en_out

    def decode(self, predictand, en_out, out_mask, enout_mask):
        """
        Args:
            predictand: (B, pre_len, C, H, W)
        output:
            (B, pre_len, C, H, W)
        """
        H, W = predictand.size()[-2:]
        T = predictand.size(1)
        predictand = unfold_func(predictand, self.mypara.patch_size)
        predictand = predictand.reshape(
            predictand.size(0), T, self.cube_dim, -1
        ).permute(0, 3, 1, 2)
        predictand = self.predictand_emb(predictand)
        output = self.decoder(predictand, en_out, out_mask, enout_mask)
        output = self.linear_output(output).permute(0, 2, 3, 1)
        output = output.reshape(
            predictand.size(0),
            T,
            self.cube_dim,
            H // self.mypara.patch_size[0],
            W // self.mypara.patch_size[1],
        )
        output = fold_func(
            output, output_size=(H, W), kernel_size=self.mypara.patch_size
        )
        return output

    def make_mask_matrix(self, sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 0).T
        return mask.to(self.mypara.device)


class multi_enc_layer(nn.Module):
    def __init__(self, enc_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([enc_layer for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class multi_dec_layer(nn.Module):
    def __init__(self, dec_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([dec_layer for _ in range(num_layers)])

    def forward(self, x, en_out, out_mask, enout_mask):
        for layer in self.layers:
            x = layer(x, en_out, out_mask, enout_mask)
        return x
