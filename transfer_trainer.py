from transformer import SpaceTimeTransformer
from config import configs
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from LoadData_2 import make_TFdataset


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".pkl":
                L.append(os.path.join(root, file))
    return L


class TFtrainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        self.network = SpaceTimeTransformer(configs).to(configs.device)
        self.adam = torch.optim.Adam(self.network.parameters(), lr=configs.TFlr)
        weight = torch.from_numpy(
            np.array([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6)
            * np.log(np.arange(24) + 1)
        ).to(configs.device)
        # i<4, a=1.5; 5<=i<=11, a=2; 12<=i<=18, a=3; 19<=i, a=4
        self.weight = weight[: self.configs.output_length]
        if self.configs.needtauxy:
            self.sstlev = 2
        else:
            self.sstlev = 0

    def score(self, y_pred, y_true):
        # compute Nino-prediction score
        with torch.no_grad():
            pred = y_pred - y_pred.mean(dim=0, keepdim=True)  # (batch, T_len)
            true = y_true - y_true.mean(dim=0, keepdim=True)  # (batch, T_len)
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

    def infer(self, dataloader):
        self.network.eval()
        nino_pred = []
        var_pred = []
        nino_true = []
        var_true = []
        with torch.no_grad():
            for input_var, var_true1 in dataloader:
                out_var = self.network(
                    src=input_var.float().to(self.device),
                    tgt=None,
                    train=False,
                )
                nino_true1 = var_true1[:, :, self.sstlev][
                    :,
                    :,
                    self.configs.lat_nino_relative[0] : self.configs.lat_nino_relative[
                        1
                    ],
                    self.configs.lon_nino_relative[0] : self.configs.lon_nino_relative[
                        1
                    ],
                ].mean(dim=[2, 3])
                out_nino = out_var[:, :, self.sstlev][
                    :,
                    :,
                    self.configs.lat_nino_relative[0] : self.configs.lat_nino_relative[
                        1
                    ],
                    self.configs.lon_nino_relative[0] : self.configs.lon_nino_relative[
                        1
                    ],
                ].mean(dim=[2, 3])
                var_true.append(var_true1)
                var_pred.append(out_var)
                nino_pred.append(out_nino)
                nino_true.append(nino_true1)
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

    def transfer_learning(self, dataset_train, dataset_eval, adr_model):
        dataloader_train = DataLoader(
            dataset_train, batch_size=self.configs.batch_size_train, shuffle=True
        )
        dataloader_eval = DataLoader(
            dataset_eval, batch_size=self.configs.batch_size_eval, shuffle=False
        )
        chk_path = self.configs.model_savepath + "TSFormer{}_TF.pkl".format(
            adr_model[-14:-4]
        )
        self.network.load_state_dict(torch.load(adr_model))
        count = 0
        (
            _,
            _,
            loss_var,
            _,
            _,
            sceval,
        ) = self.infer(dataloader=dataloader_eval)
        best = loss_var-sceval
        for i_epoch in range(self.configs.TFnum_epochs):
            print("\n-->TFepoch: {0}".format(i_epoch))
            self.network.train()
            for j, (input_var, var_true) in enumerate(dataloader_train):
                out_var = self.network(
                    src=input_var.float().to(configs.device),
                    tgt=var_true.float().to(self.device),
                    train=True,
                    ssr_ratio=0,
                )
                # var_pred: [batch,T_tgt,C,H,W]
                nino_true = var_true[:, :, self.sstlev][
                    :,
                    :,
                    self.configs.lat_nino_relative[0] : self.configs.lat_nino_relative[
                        1
                    ],
                    self.configs.lon_nino_relative[0] : self.configs.lon_nino_relative[
                        1
                    ],
                ].mean(dim=[2, 3])
                nino_pred = out_var[:, :, self.sstlev][
                    :,
                    :,
                    self.configs.lat_nino_relative[0] : self.configs.lat_nino_relative[
                        1
                    ],
                    self.configs.lon_nino_relative[0] : self.configs.lon_nino_relative[
                        1
                    ],
                ].mean(dim=[2, 3])
                self.adam.zero_grad()
                sc = self.score(nino_pred, nino_true.float().to(self.device))
                loss_var = self.loss_var(out_var, var_true.float().to(self.device))
                loss_nino = self.loss_nino(nino_pred, nino_true.float().to(self.device))
                combine_loss = self.combien_loss(loss_var, loss_nino)
                combine_loss.backward()
                self.adam.step()
                # --------每训练display_interval个batch后输出一下信息
                if j % 10 == 0:
                    print(
                        "\n-->batch:{} loss_nino:{:.3f}, score:{:.3f}".format(
                            j, loss_nino, sc
                        )
                    )
                # ---------epoch>M时在测试集加密检测，并保存score最大的模型
                if (i_epoch + 1 >= 2) and (j + 1) % 30 == 0:
                    (
                        _,
                        _,
                        loss_var,
                        lossnino_eval,
                        _,
                        sceval,
                    ) = self.infer(dataloader=dataloader_eval)
                    print(
                        "-->加密验证中... \nloss_var:{:.3f} \nscore:{:.3f}".format(
                            loss_var, sceval
                        )
                    )
                    if loss_var-sceval < best:
                        torch.save(
                            self.network.state_dict(),
                            chk_path,
                        )
                        best = loss_var-sceval
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
            if lossvar_eval-sceval >= best:
                count += 1
                print("\nloss is not decrease for {} epoch".format(count))
            else:
                count = 0
                print(
                    "\nloss is decrease from {:.5f} to {:.5f}   \nsaving model...\n".format(
                        best, lossvar_eval-sceval
                    )
                )
                torch.save(
                    self.network.state_dict(),
                    chk_path,
                )
                print(chk_path)
                best = lossvar_eval-sceval
        del self.network


if __name__ == "__main__":
    adr_TF = "/home/zhoulu/mycode/data/up150_tauxy/SODA_ORAS_group_temp_tauxy_before1979_kb.nc"
    print("\nloading TF_data...")
    dataTF = make_TFdataset(
        address=adr_TF,
        config=configs,
        ngroup=1400,
    )
    print(dataTF.getdatashape())
    print(dataTF.selectregion())

    train_size = int(0.9 * len(dataTF))
    eval_size = len(dataTF) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataTF, [train_size, eval_size], generator=torch.Generator().manual_seed(0)
    )
    # -------------------------------------------------------------
    file_adr = "./model"
    files = file_name(file_adr)
    file_num = len(files)
    for i_file in files[: file_num + 1]:
        print(i_file)
        trainer = TFtrainer(configs)
        trainer.transfer_learning(
            dataset_train=train_dataset, dataset_eval=eval_dataset, adr_model=i_file
        )
