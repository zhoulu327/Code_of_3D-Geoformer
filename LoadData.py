import numpy as np
from torch.utils.data import Dataset
import xarray as xr
import random


class make_dataset(Dataset):
    def __init__(self, mypara):
        self.mypara = mypara
        data_in = xr.open_dataset(mypara.adr_pretr)
        self.lev = data_in["lev"].values
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.lev_range = mypara.lev_range
        self.lon_range = mypara.lon_range
        self.lat_range = mypara.lat_range

        nino34 = data_in["nino34"].values
        temp = data_in["temperatureNor"][
            :,
            :,
            mypara.lev_range[0] : mypara.lev_range[1],
            mypara.lat_range[0] : mypara.lat_range[1],
            mypara.lon_range[0] : mypara.lon_range[1],
        ].values
        temp = np.nan_to_num(temp)
        temp[abs(temp) > 999] = 0
        if mypara.needtauxy:
            print("loading tauxy fields")
            taux = data_in["tauxNor"][
                :,
                :,
                mypara.lat_range[0] : mypara.lat_range[1],
                mypara.lon_range[0] : mypara.lon_range[1],
            ].values
            taux = np.nan_to_num(taux)
            taux[abs(taux) > 999] = 0
            tauy = data_in["tauyNor"][
                :,
                :,
                mypara.lat_range[0] : mypara.lat_range[1],
                mypara.lon_range[0] : mypara.lon_range[1],
            ].values
            tauy = np.nan_to_num(tauy)
            tauy[abs(tauy) > 999] = 0

            field_data = np.concatenate(
                (taux[:, :, None], tauy[:, :, None], temp), axis=2
            )  # [n_model,mon,lev,lat,lon]
            del temp, taux, tauy
        else:
            field_data = temp
            del temp
        self.dataX, self.dataY, self.dataY_nino = self.deal_oridata(
            field_data=field_data,
            nino_data=nino34,
            look_back=mypara.look_back,
            pre_len=mypara.pre_len,
            interval=mypara.interval,
            need_nino=False,
        )
        del field_data

    def deal_oridata(
        self, field_data, nino_data, look_back, pre_len, interval, need_nino
    ):
        """
        :param field_data: [n_model,mon,lev,lat,lon] if mypara.needtauxy==True,lev=temp_lev+2
        :param nino_data: [num_model,mon]
        :param look_back: the length of predictors
        :param pre_len: the length of predictand
        :param interval: Random sampling interval
        :param need_nino: output nino34 in this function?
        :return:
        """
        print("Random sampling...")
        st_min = look_back - 1
        ed_max = field_data.shape[1] - pre_len
        one_model_group = int(np.floor((ed_max - st_min) / interval))
        all_group = int(field_data.shape[0] * one_model_group)
        out_field_x = np.zeros(
            [
                all_group,
                look_back,
                field_data.shape[2],
                field_data.shape[3],
                field_data.shape[4],
            ]
        )
        out_field_y = np.zeros(
            [
                all_group,
                pre_len,
                field_data.shape[2],
                field_data.shape[3],
                field_data.shape[4],
            ]
        )
        out_nino = np.zeros([all_group, pre_len])
        iii = 0
        for i in range(field_data.shape[0]):
            for j in range(one_model_group):
                rd = random.randint(st_min, ed_max - 1)
                out_field_x[iii] = field_data[i, rd - look_back + 1 : rd + 1]
                out_field_y[iii] = field_data[i, rd + 1 : rd + pre_len + 1]
                if need_nino == True:
                    out_nino[iii] = nino_data[i, rd + 1 : rd + pre_len + 1]
                iii += 1
        print("Sampling completed...")
        return out_field_x, out_field_y, out_nino

    def getdatashape(self):
        return {
            "dataX": self.dataX.shape,
            "dataY": self.dataY.shape,
            "dataY_nino": self.dataY_nino.shape,
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
            "lev of temperature: {}m to {}m".format(
                self.lev[self.lev_range[0]], self.lev[self.lev_range[1] - 1]
            ),
        }

    def __len__(self):
        return self.dataX.shape[0]

    def __getitem__(self, idx):
        return self.dataX[idx], self.dataY[idx], self.dataY_nino[idx]


class make_testdataset(Dataset):
    def __init__(
        self,
        address,
        lookback,
        Tout,
        ngroup=None,
        needtauxy=False,
        lev_range=(0, 1),
        lon_range=(0, 1),
        lat_range=(0, 1),
    ):
        """
        :param address: 训练集地址
        :param lev_range: 海温场取得lev范围
        :param needtauxy: 是否读取风应力
        :param lon_range: 空间场范围
        :param lat_range:
        """
        data_in = xr.open_dataset(address)
        self.lev = data_in["lev"].values
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.lev_range = lev_range
        self.lon_range = lon_range
        self.lat_range = lat_range

        temp_in = data_in["temperatureNor_in"][
            :,
            :,
            lev_range[0] : lev_range[1],
            lat_range[0] : lat_range[1],
            lon_range[0] : lon_range[1],
        ].values
        temp_in = np.nan_to_num(temp_in)
        temp_in[abs(temp_in) > 999] = 0
        assert lookback == temp_in.shape[1]
        if needtauxy:
            print("已使用tauxy...")
            taux_in = data_in["tauxNor_in"][
                :, :, lat_range[0] : lat_range[1], lon_range[0] : lon_range[1]
            ].values
            taux_in = np.nan_to_num(taux_in)
            taux_in[abs(taux_in) > 999] = 0
            tauy_in = data_in["tauyNor_in"][
                :, :, lat_range[0] : lat_range[1], lon_range[0] : lon_range[1]
            ].values
            tauy_in = np.nan_to_num(tauy_in)
            tauy_in[abs(tauy_in) > 999] = 0
            # ------------拼接
            field_data_in = np.concatenate(
                (taux_in[:, :, None], tauy_in[:, :, None], temp_in), axis=2
            )  # [group,lb,all_lev,lat,lon]
            del temp_in, taux_in, tauy_in
        else:
            field_data_in = temp_in
            del temp_in
        # ====================out
        temp_out = data_in["temperatureNor_out"][
            :,
            :,
            lev_range[0] : lev_range[1],
            lat_range[0] : lat_range[1],
            lon_range[0] : lon_range[1],
        ].values
        temp_out = np.nan_to_num(temp_out)
        temp_out[abs(temp_out) > 999] = 0
        assert Tout == temp_out.shape[1]
        if needtauxy:
            print("已使用tauxy...")
            taux_out = data_in["tauxNor_out"][
                :, :, lat_range[0] : lat_range[1], lon_range[0] : lon_range[1]
            ].values
            taux_out = np.nan_to_num(taux_out)
            taux_out[abs(taux_out) > 999] = 0
            tauy_out = data_in["tauyNor_out"][
                :, :, lat_range[0] : lat_range[1], lon_range[0] : lon_range[1]
            ].values
            tauy_out = np.nan_to_num(tauy_out)
            tauy_out[abs(tauy_out) > 999] = 0
            # ------------拼接
            field_data_out = np.concatenate(
                (taux_out[:, :, None], tauy_out[:, :, None], temp_out), axis=2
            )  # [group,lb,all_lev,lat,lon]
            del temp_out, taux_out, tauy_out
        else:
            field_data_out = temp_out
            del temp_out
        # -----------------------------
        self.dataX, self.dataY = self.deal_testdata(
            field_data_in=field_data_in, field_data_out=field_data_out, ngroup=ngroup
        )
        del field_data_in, field_data_out

    def deal_testdata(self, field_data_in, field_data_out, ngroup):
        print("正在采样...")
        lb = field_data_in.shape[1]
        Tout = field_data_out.shape[1]
        if ngroup is None:
            ngroup = field_data_in.shape[0]
        out_field_x = np.zeros(
            [
                ngroup,
                lb,
                field_data_in.shape[2],
                field_data_in.shape[3],
                field_data_in.shape[4],
            ]
        )
        out_field_y = np.zeros(
            [
                ngroup,
                Tout,
                field_data_out.shape[2],
                field_data_out.shape[3],
                field_data_out.shape[4],
            ]
        )
        iii = 0
        for j in range(ngroup):
            rd = random.randint(0, field_data_in.shape[0] - 1)
            out_field_x[iii] = field_data_in[rd]
            out_field_y[iii] = field_data_out[rd]
            iii += 1
        print("采样完成...")
        return out_field_x, out_field_y

    def getdatashape(self):
        return {
            "dataX": self.dataX.shape,
            "dataY": self.dataY.shape,
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
        return self.dataX[idx], self.dataY[idx]
