from transformer import Geoformer
from myconfig import mypara
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import numpy as np
import math
from LoadData import make_dataset
import os
import h5py


class NoamOpt:
    """
    learning rate warmup and decay
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )


class Trainer:
    def __init__(self, mypara):
        assert mypara.input_channal == mypara.output_channal
        self.mypara = mypara
        self.device = mypara.device
        self.network = Geoformer(mypara).to(mypara.device)
        # self.adam = torch.optim.Adam(self.network.parameters(), lr=5e-5)
        # 定义一个warming up的优化器，内核采用Adam
        adam = torch.optim.Adam(self.network.parameters(), lr=0)
        factor = math.sqrt(mypara.d_model * mypara.warmup) * 0.0015
        self.opt = NoamOpt(
            model_size=mypara.d_model,
            factor=factor,
            warmup=mypara.warmup,
            optimizer=adam,
        )
        self.sstlevel = 0
        if self.mypara.needtauxy:
            self.sstlevel = 2

        # 计算accskill的权重
        weight = torch.from_numpy(
            np.array([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6)
            * np.log(np.arange(24) + 1)
        ).to(mypara.device)
        # i<4, a=1.5; 5<=i<=11, a=2; 12<=i<=18, a=3; 19<=i, a=4
        self.weight = weight[: self.mypara.output_length]

    def score(self, y_pred, y_true):
        # compute Nino-prediction score
        with torch.no_grad():
            pred = y_pred - y_pred.mean(dim=0, keepdim=True)  # (batch, 24)
            true = y_true - y_true.mean(dim=0, keepdim=True)  # (batch, 24)
            cor = (pred * true).sum(dim=0) / (
                torch.sqrt(torch.sum(pred ** 2, dim=0) * torch.sum(true ** 2, dim=0))
                + 1e-6
            )
            acc = (self.weight * cor).sum()
            rmse = torch.mean((y_pred - y_true) ** 2, dim=0).sqrt().sum()
            sc = 2 / 3.0 * acc - rmse
        return sc.item()

    def loss_var(self, y_pred, y_true):
        # y_pred/y_true (batch, T_len, C, H, W)
        rmse = torch.mean((y_pred - y_true) ** 2, dim=[3, 4])
        rmse = rmse.sqrt().mean(dim=0)
        rmse = torch.sum(rmse, dim=[0, 1])  # [T_len]-->[1]
        return rmse

    def loss_nino(self, y_pred, y_true):
        # with torch.no_grad():
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=0))
        return rmse.sum()

    def combien_loss(self, loss1, loss2):
        combine_loss = loss1 + loss2
        return combine_loss

    def save_configs(self, config_path):
        with open(config_path, "wb") as path:
            pickle.dump(self.mypara, path)

    def infer(self, dataloader):
        self.network.eval()
        nino_pred = []
        var_pred = []
        nino_true = []
        var_true = []
        with torch.no_grad():
            for input_var, var_true1, _ in dataloader:
                # ----------------cal nino_true
                # 因为原数据集中nino_true是使用未标准化的SST计算的，而预测得到的nino是根据标准化的SST计算的
                # 二者之间有割裂，所以这里单独根据var_true计算nino_true
                SST = var_true1[:, :, self.sstlevel]
                nino_true1 = SST[
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[
                        1
                    ],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[
                        1
                    ],
                ].mean(dim=[2, 3])
                # NN模型推测
                out_var = self.network(
                    src=input_var.float().to(self.device),
                    tgt=None,
                    train=False,
                )
                SST_out = out_var[:, :, self.sstlevel]
                out_nino = SST_out[
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[
                        1
                    ],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[
                        1
                    ],
                ].mean(dim=[2, 3])
                var_true.append(var_true1)
                nino_true.append(nino_true1)
                var_pred.append(out_var)
                nino_pred.append(out_nino)
            var_pred = torch.cat(var_pred, dim=0)
            nino_pred = torch.cat(nino_pred, dim=0)
            nino_true = torch.cat(nino_true, dim=0)
            var_true = torch.cat(var_true, dim=0)
            # --------------------
            sc = self.score(nino_pred, nino_true.float().to(self.device))
            loss_var = self.loss_var(var_pred, var_true.float().to(self.device)).item()
            loss_nino = self.loss_nino(
                nino_pred, nino_true.float().to(self.device)
            ).item()
            combine_loss = self.combien_loss(loss_var, loss_nino)
        return (
            var_pred,
            nino_pred,
            loss_var,
            loss_nino,
            combine_loss,
            sc,
        )

    def train(self, dataset_train, dataset_eval, chk_path=None):
        torch.manual_seed(self.mypara.seeds)
        dataloader_train = DataLoader(
            dataset_train, batch_size=self.mypara.batch_size_train, shuffle=True
        )
        dataloader_eval = DataLoader(
            dataset_eval, batch_size=self.mypara.batch_size_eval, shuffle=False
        )
        count = 0
        best = math.inf
        ssr_ratio = 1
        for i_epoch in range(self.mypara.num_epochs):
            print("==========" * 8)
            print("\n-->epoch: {0}".format(i_epoch))
            # ---------train
            self.network.train()
            for j, (input_var, var_true, _) in enumerate(dataloader_train):
                # input_var: [B, lb,C,h,w] var_true: [B,pre_len,C,h,w] nino_true: [B,pre_len]
                # ----------------cal nino_true
                # 因为原数据集中nino_true是使用未标准化的SST计算的，而预测得到的nino是根据标准化的SST计算的
                # 二者之间有割裂，所以这里根据标准化的var_true计算nino_true
                SST = var_true[:, :, self.sstlevel]
                nino_true = SST[
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[
                        1
                    ],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[
                        1
                    ],
                ].mean(dim=[2, 3])
                # ----------------------
                chk_path = self.mypara.model_savepath + "TSFormer{}_{}.pkl".format(
                    i_epoch, j
                )
                if ssr_ratio > 0:
                    ssr_ratio = max(ssr_ratio - self.mypara.ssr_decay_rate, 0)
                # -------training for one batch
                var_pred = self.network(
                    src=input_var.float().to(self.device),
                    tgt=var_true.float().to(self.device),
                    train=True,
                    ssr_ratio=ssr_ratio,
                )
                # var_pred: [batch,T_tgt,C,H,W]  nino_pred: [batch,T_tgt]
                SST_pred = var_pred[:, :, self.sstlevel]
                nino_pred = SST_pred[
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[
                        1
                    ],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[
                        1
                    ],
                ].mean(dim=[2, 3])
                self.opt.optimizer.zero_grad()
                # self.adam.zero_grad()
                loss_var = self.loss_var(var_pred, var_true.float().to(self.device))
                loss_nino = self.loss_nino(nino_pred, nino_true.float().to(self.device))
                # loss_var.backward()
                combine_loss = self.combien_loss(loss_var, loss_nino)
                combine_loss.backward()
                # 防止梯度爆炸引入的梯度裁剪方法，梯度大于阈值会将其拉回阈值
                if mypara.gradient_clipping:
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), mypara.clipping_threshold
                    )
                self.opt.step()
                # self.adam.step()
                # --------每训练display_interval个batch后输出一下信息
                if j % self.mypara.display_interval == 0:
                    sc = self.score(nino_pred, nino_true.float().to(self.device))
                    print(
                        "\n-->Training batch:{} loss_var:{:.3f}, loss_nino:{:.3f}, score:{:.3f}, ssr: {:.3f}".format(
                            j, loss_var, loss_nino, sc, ssr_ratio
                        )
                    )

                # ---------epoch>M时在测试集加密检测，并保存score最大的模型
                if i_epoch >= 4 and (j + 1) % 300 == 0:
                    (
                        _,
                        _,
                        lossvar_eval,
                        lossnino_eval,
                        comloss_eval,
                        sceval,
                    ) = self.infer(dataloader=dataloader_eval)
                    print(
                        "-->加密验证中... \nloss_var:{:.3f} \nloss_nino:{:.3f} \nloss_com:{:.3f} \nscore:{:.3f}".format(
                            lossvar_eval, lossnino_eval, comloss_eval, sceval
                        )
                    )
                    if comloss_eval < best:
                        torch.save(
                            self.network.state_dict(),
                            chk_path,
                        )
                        best = comloss_eval
                        count = 0
                        print("\nsaving model...")
                        print(chk_path)

            # ----------每个epoch之后在验证集上验证一下
            (
                _,
                _,
                lossvar_eval,
                lossnino_eval,
                comloss_eval,
                sceval,
            ) = self.infer(dataloader=dataloader_eval)
            print(
                "\n-->epoch{}结束, 验证中... \nloss_var:{:.3f} \nloss_nino:{:.3f} \nloss_com:{:.3f} \nscore: {:.3f}".format(
                    i_epoch, lossvar_eval, lossnino_eval, comloss_eval, sceval
                )
            )
            if comloss_eval >= best:
                count += 1
                print("\nloss is not decrease for {} epoch".format(count))
            else:
                count = 0
                print(
                    "\nloss is decrease from {:.4f} to {:.4f}   \nsaving model...\n".format(
                        best, comloss_eval
                    )
                )
                torch.save(
                    self.network.state_dict(),
                    chk_path,
                )
                print(chk_path)
                best = comloss_eval

            # ---------early stop
            if count == self.mypara.patience:
                print(
                    "\n-----!!!early stopping reached, min(loss_var)= {:3f}!!!-----".format(
                        best
                    )
                )
                break
        del self.network


if __name__ == "__main__":
    print(mypara.__dict__)
    # -----------------------pre-training----------------------------------
    print("\nloading pre-training data...")
    predata = make_dataset(mypara=mypara)
    print(predata.getdatashape())
    print(predata.selectregion())

    train_size = int(mypara.TraindataProportion * len(predata))
    eval_size = len(predata) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        predata, [train_size, eval_size], generator=torch.Generator().manual_seed(0)
    )
    # -------------------------------------------------------------
    trainer = Trainer(mypara)
    trainer.save_configs(mypara.model_savepath + "config_train.pk")
    trainer.train(
        dataset_train=train_dataset,
        dataset_eval=eval_dataset,
    )
