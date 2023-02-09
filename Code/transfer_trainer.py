from Geoformer import Geoformer
from myconfig import mypara
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from LoadData import make_TFdataset


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".pkl":
                L.append(os.path.join(root, file))
    return L


class TFtrainer:
    def __init__(self, mypara):
        self.mypara = mypara
        self.device = mypara.device
        self.mymodel = Geoformer(mypara).to(mypara.device)
        self.adam = torch.optim.Adam(self.mymodel.parameters(), lr=mypara.TFlr)
        weight = torch.from_numpy(
            np.array([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6)
            * np.log(np.arange(24) + 1)
        ).to(mypara.device)
        self.weight = weight[: self.mypara.output_length]
        if self.mypara.needtauxy:
            self.sstlev = 2
        else:
            self.sstlev = 0

    def calscore(self, y_pred, y_true):
        with torch.no_grad():
            pred = y_pred - y_pred.mean(dim=0, keepdim=True)
            true = y_true - y_true.mean(dim=0, keepdim=True)
            cor = (pred * true).sum(dim=0) / (
                torch.sqrt(torch.sum(pred ** 2, dim=0) * torch.sum(true ** 2, dim=0))
                + 1e-6
            )
            acc = (self.weight * cor).sum()
            rmse = torch.mean((y_pred - y_true) ** 2, dim=0).sqrt().sum()
            sc = 2 / 3.0 * acc - rmse
        return sc.item()

    def loss_var(self, y_pred, y_true):
        rmse = torch.mean((y_pred - y_true) ** 2, dim=[3, 4])
        rmse = rmse.sqrt().mean(dim=0)
        rmse = torch.sum(rmse, dim=[0, 1])
        return rmse

    def loss_nino(self, y_pred, y_true):
        # with torch.no_grad():
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=0))
        return rmse.sum()

    def combien_loss(self, loss1, loss2):
        combine_loss = loss1 + loss2
        return combine_loss

    def model_pred(self, dataloader):
        self.mymodel.eval()
        nino_pred = []
        var_pred = []
        nino_true = []
        var_true = []
        with torch.no_grad():
            for input_var, var_true1 in dataloader:
                out_var = self.mymodel(
                    input_var.float().to(self.device),
                    None,
                    train=False,
                )
                nino_true1 = var_true1[:, :, self.sstlev][
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                ].mean(dim=[2, 3])
                out_nino = out_var[:, :, self.sstlev][
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
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
            sc = self.calscore(nino_pred, nino_true.float().to(self.device))
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
            dataset_train, batch_size=self.mypara.batch_size_train, shuffle=True
        )
        dataloader_eval = DataLoader(
            dataset_eval, batch_size=self.mypara.batch_size_eval, shuffle=False
        )
        chk_path = self.mypara.model_savepath + "Geoformer_TF.pkl"
        self.mymodel.load_state_dict(torch.load(adr_model))
        count = 0
        (
            _,
            _,
            loss_var,
            _,
            _,
            sceval,
        ) = self.model_pred(dataloader=dataloader_eval)
        best = loss_var - sceval
        for i_epoch in range(self.mypara.TFnum_epochs):
            print("\n-->TFepoch: {0}".format(i_epoch))
            self.mymodel.train()
            for j, (input_var, var_true) in enumerate(dataloader_train):
                out_var = self.mymodel(
                    input_var.float().to(mypara.device),
                    var_true.float().to(self.device),
                    train=True,
                    sv_ratio=0,
                )
                nino_true = var_true[:, :, self.sstlev][
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                ].mean(dim=[2, 3])
                nino_pred = out_var[:, :, self.sstlev][
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                ].mean(dim=[2, 3])
                self.adam.zero_grad()
                sc = self.calscore(nino_pred, nino_true.float().to(self.device))
                loss_var = self.loss_var(out_var, var_true.float().to(self.device))
                loss_nino = self.loss_nino(nino_pred, nino_true.float().to(self.device))
                combine_loss = self.combien_loss(loss_var, loss_nino)
                combine_loss.backward()
                self.adam.step()
                if j % 10 == 0:
                    print(
                        "\n-->batch:{} loss_nino:{:.3f}, score:{:.3f}".format(
                            j, loss_nino, sc
                        )
                    )
                if (i_epoch + 1 >= 2) and (j + 1) % 30 == 0:
                    (
                        _,
                        _,
                        loss_var,
                        lossnino_eval,
                        _,
                        sceval,
                    ) = self.model_pred(dataloader=dataloader_eval)
                    print(
                        "-->variation... \nloss_var:{:.3f} \nscore:{:.3f}".format(
                            loss_var, sceval
                        )
                    )
                    if loss_var - sceval < best:
                        torch.save(
                            self.mymodel.state_dict(),
                            chk_path,
                        )
                        best = loss_var - sceval
                        count = 0
                        print("\nsaving model...")
                        print(chk_path)
            (
                _,
                _,
                lossvar_eval,
                lossnino_eval,
                comloss_eval,
                sceval,
            ) = self.model_pred(dataloader=dataloader_eval)
            print(
                "\n-->epoch{} end... \nloss_var:{:.3f} \nloss_nino:{:.3f} \nloss_com:{:.3f} \nscore: {:.3f}".format(
                    i_epoch, lossvar_eval, lossnino_eval, comloss_eval, sceval
                )
            )
            if lossvar_eval - sceval >= best:
                count += 1
                print("\nloss is not decrease for {} epoch".format(count))
            else:
                count = 0
                print(
                    "\nloss is decrease from {:.5f} to {:.5f}   \nsaving model...\n".format(
                        best, lossvar_eval - sceval
                    )
                )
                torch.save(
                    self.mymodel.state_dict(),
                    chk_path,
                )
                best = lossvar_eval - sceval
        del self.mymodel


if __name__ == "__main__":
    adr_TF = "./data/SODA_ORAS_group_temp_tauxy_before1979_kb.nc"
    print("\nloading TF_data...")
    dataTF = make_TFdataset(
        address=adr_TF,
        mypara=mypara,
        ngroup=1500,
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
        trainer = TFtrainer(mypara)
        trainer.transfer_learning(
            dataset_train=train_dataset, dataset_eval=eval_dataset, adr_model=i_file
        )
