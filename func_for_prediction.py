from Geoformer import Geoformer
import torch
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
from torch.utils.data import Dataset


class make_dataset_test(Dataset):
    def __init__(
        self,
        address,
        needtauxy,
        lev_range=(0, 1),
        lon_range=(0, 1),
        lat_range=(0, 1),
    ):
        data_in = xr.open_dataset(address)
        self.lev = data_in["lev"].values
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.lev_range = lev_range
        self.lon_range = lon_range
        self.lat_range = lat_range

        temp = data_in["temperatureNor"][
            :,
            :,
            lev_range[0] : lev_range[1],
            lat_range[0] : lat_range[1],
            lon_range[0] : lon_range[1],
        ].values
        temp = np.nan_to_num(temp)
        temp[abs(temp) > 999] = 0
        if needtauxy:
            taux = data_in["tauxNor"][
                :,
                :,
                lat_range[0] : lat_range[1],
                lon_range[0] : lon_range[1],
            ].values
            taux = np.nan_to_num(taux)
            taux[abs(taux) > 999] = 0
            tauy = data_in["tauyNor"][
                :,
                :,
                lat_range[0] : lat_range[1],
                lon_range[0] : lon_range[1],
            ].values
            tauy = np.nan_to_num(tauy)
            tauy[abs(tauy) > 999] = 0
            # --------------
            self.dataX = np.concatenate(
                (taux[:, :, None], tauy[:, :, None], temp), axis=2
            )
            del temp, taux, tauy
        else:
            self.dataX = temp
            del temp

    def getdatashape(self):
        return {
            "dataX.shape": self.dataX.shape,
        }

    def selectregion(self):
        return {
            "lon: {}E to {}E".format(
                self.lon[self.lon_range[0]],
                self.lon[self.lon_range[1] - 1],
            ),
            "lat: {}S to {}N".format(
                self.lat[self.lat_range[0]],
                self.lat[self.lat_range[1] - 1],
            ),
            "lev: {}m to {}m".format(
                self.lev[self.lev_range[0]], self.lev[self.lev_range[1] - 1]
            ),
        }

    def __len__(self):
        return self.dataX.shape[0]

    def __getitem__(self, idx):
        return self.dataX[idx]


def func_pre(mypara, adr_model, adr_datain, adr_oridata, needtauxy):
    lead_max = mypara.output_length
    # -------------
    data_ori = xr.open_dataset(adr_oridata)
    temp_ori_region = data_ori["temperatureNor"][
        :,
        mypara.lev_range[0] : mypara.lev_range[1],
        mypara.lat_range[0] : mypara.lat_range[1],
        mypara.lon_range[0] : mypara.lon_range[1],
    ].values
    nino34 = data_ori["nino34"].values
    stdtemp = data_ori["stdtemp"][mypara.lev_range[0] : mypara.lev_range[1]].values
    stdtemp = np.nanmean(stdtemp, axis=(1, 2))
    if needtauxy:
        taux_ori_region = data_ori["tauxNor"][
            :,
            mypara.lat_range[0] : mypara.lat_range[1],
            mypara.lon_range[0] : mypara.lon_range[1],
        ].values
        tauy_ori_region = data_ori["tauyNor"][
            :,
            mypara.lat_range[0] : mypara.lat_range[1],
            mypara.lon_range[0] : mypara.lon_range[1],
        ].values
        stdtaux = data_ori["stdtaux"].values
        stdtaux = np.nanmean(stdtaux, axis=(0, 1))
        stdtauy = data_ori["stdtauy"].values
        stdtauy = np.nanmean(stdtauy, axis=(0, 1))
        # ---------
        var_ori_region = np.concatenate(
            (taux_ori_region[:, None], tauy_ori_region[:, None], temp_ori_region),
            axis=1,
        )
        del taux_ori_region, tauy_ori_region, temp_ori_region
        stds = np.concatenate((stdtaux[None], stdtauy[None], stdtemp), axis=0)
        del stdtemp, stdtauy, stdtaux
    else:
        var_ori_region = temp_ori_region
        del temp_ori_region
        stds = stdtemp
        del stdtemp
    # -----------------------------------------------
    dataCS = make_dataset_test(
        address=adr_datain,
        needtauxy=needtauxy,
        lev_range=mypara.lev_range,
        lon_range=mypara.lon_range,
        lat_range=mypara.lat_range,
    )
    test_group = len(dataCS)
    print(dataCS.getdatashape())
    print(dataCS.selectregion())
    dataloader_test = DataLoader(
        dataCS, batch_size=mypara.batch_size_eval, shuffle=False
    )
    mymodel = Geoformer(mypara).to(mypara.device)
    mymodel.load_state_dict(torch.load(adr_model))
    mymodel.eval()
    if needtauxy:
        n_lev = mypara.lev_range[1] - mypara.lev_range[0] + 2
        sst_lev = 2
    else:
        n_lev = mypara.lev_range[1] - mypara.lev_range[0]
        sst_lev = 0
    var_pred = np.zeros(
        [
            test_group,
            lead_max,
            n_lev,
            mypara.lat_range[1] - mypara.lat_range[0],
            mypara.lon_range[1] - mypara.lon_range[0],
        ]
    )
    ii = 0
    iii = 0
    with torch.no_grad():
        for input_var in dataloader_test:
            out_var = mymodel(
                input_var.float().to(mypara.device),
                predictand=None,
                train=False,
            )
            ii += out_var.shape[0]
            if torch.cuda.is_available():
                var_pred[iii:ii] = out_var.cpu().detach().numpy()
            else:
                var_pred[iii:ii] = out_var.detach().numpy()
            iii = ii
    del (
        out_var,
        input_var,
    )
    del mymodel, dataCS, dataloader_test

    # ---------------------------------------------------
    len_data = test_group - lead_max
    print("len_data:", len_data)
    # Obs fields
    cut_var_true = var_ori_region[(12 + lead_max) - 1 :]
    cut_var_true = cut_var_true * stds[None, :, None, None]
    cut_nino_true = nino34[(12 + lead_max) - 1 :]
    assert cut_nino_true.shape[0] == cut_var_true.shape[0] == len_data
    # Pred fields
    cut_var_pred = np.zeros(
        [lead_max, len_data, var_pred.shape[2], var_pred.shape[3], var_pred.shape[4]]
    )
    cut_nino_pred = np.zeros([lead_max, len_data])
    for i in range(lead_max):
        l = i + 1
        cut_var_pred[i] = (
            var_pred[lead_max - l : test_group - l, i] * stds[None, :, None, None]
        )
        cut_nino_pred[i] = np.nanmean(
            cut_var_pred[
                i,
                :,
                sst_lev,
                mypara.lat_nino_relative[0] : mypara.lat_nino_relative[1],
                mypara.lon_nino_relative[0] : mypara.lon_nino_relative[1],
            ],
            axis=(1, 2),
        )
    assert cut_var_pred.shape[1] == cut_var_true.shape[0]
    return (
        cut_var_pred,
        cut_var_true,
        cut_nino_pred,
        cut_nino_true,
    )
