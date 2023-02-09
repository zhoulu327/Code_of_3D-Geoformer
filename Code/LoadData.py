import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset
import xarray as xr
import random


class make_dataset1(Dataset):
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


class make_dataset2(IterableDataset):
    """
    online reading dataset
    """
    def __init__(self, mypara):
        self.mypara = mypara
        data_in = xr.open_dataset(mypara.adr_pretr)
        self.lev = data_in["lev"].values
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.lev_range = mypara.lev_range
        self.lon_range = mypara.lon_range
        self.lat_range = mypara.lat_range
        self.input_length = mypara.input_length
        self.output_length = mypara.output_length
        self.all_group = mypara.all_group
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
            print("loading tauxy...")
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
            self.field_data = np.concatenate(
                (taux[:, :, None], tauy[:, :, None], temp), axis=2
            )
            del temp, taux, tauy
        else:
            self.field_data = temp
            del temp

    def __iter__(self):
        st_min = self.input_length - 1
        ed_max = self.field_data.shape[1] - self.output_length
        for i in range(self.all_group):
            rd_m = random.randint(0, self.field_data.shape[0] - 1)
            rd = random.randint(st_min, ed_max - 1)
            dataX = self.field_data[rd_m, rd - self.input_length + 1 : rd + 1]
            dataY = self.field_data[rd_m, rd + 1 : rd + self.output_length + 1]
            yield dataX, dataY

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
            "temp lev: {}m to {}m".format(
                self.lev[self.lev_range[0]], self.lev[self.lev_range[1] - 1]
            ),
        }


class make_testdataset(Dataset):
    def __init__(self, mypara, ngroup):
        self.mypara = mypara
        data_in = xr.open_dataset(mypara.adr_eval)
        self.lev = data_in["lev"].values
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.lev_range = mypara.lev_range
        self.lon_range = mypara.lon_range
        self.lat_range = mypara.lat_range

        temp_in = data_in["temperatureNor_in"][
            :,
            :,
            mypara.lev_range[0] : mypara.lev_range[1],
            mypara.lat_range[0] : mypara.lat_range[1],
            mypara.lon_range[0] : mypara.lon_range[1],
        ].values
        temp_in = np.nan_to_num(temp_in)
        temp_in[abs(temp_in) > 999] = 0
        assert mypara.input_length == temp_in.shape[1]
        if mypara.needtauxy:
            print("loading tauxy...")
            taux_in = data_in["tauxNor_in"][
                :,
                :,
                mypara.lat_range[0] : mypara.lat_range[1],
                mypara.lon_range[0] : mypara.lon_range[1],
            ].values
            taux_in = np.nan_to_num(taux_in)
            taux_in[abs(taux_in) > 999] = 0
            tauy_in = data_in["tauyNor_in"][
                :,
                :,
                mypara.lat_range[0] : mypara.lat_range[1],
                mypara.lon_range[0] : mypara.lon_range[1],
            ].values
            tauy_in = np.nan_to_num(tauy_in)
            tauy_in[abs(tauy_in) > 999] = 0
            field_data_in = np.concatenate(
                (taux_in[:, :, None], tauy_in[:, :, None], temp_in), axis=2
            )
            del temp_in, taux_in, tauy_in
        else:
            field_data_in = temp_in
            del temp_in
        # ====================out
        temp_out = data_in["temperatureNor_out"][
            :,
            :,
            mypara.lev_range[0] : mypara.lev_range[1],
            mypara.lat_range[0] : mypara.lat_range[1],
            mypara.lon_range[0] : mypara.lon_range[1],
        ].values
        temp_out = np.nan_to_num(temp_out)
        temp_out[abs(temp_out) > 999] = 0
        assert mypara.output_length == temp_out.shape[1]
        if mypara.needtauxy:
            print("loading tauxy...")
            taux_out = data_in["tauxNor_out"][
                :,
                :,
                mypara.lat_range[0] : mypara.lat_range[1],
                mypara.lon_range[0] : mypara.lon_range[1],
            ].values
            taux_out = np.nan_to_num(taux_out)
            taux_out[abs(taux_out) > 999] = 0
            tauy_out = data_in["tauyNor_out"][
                :,
                :,
                mypara.lat_range[0] : mypara.lat_range[1],
                mypara.lon_range[0] : mypara.lon_range[1],
            ].values
            tauy_out = np.nan_to_num(tauy_out)
            tauy_out[abs(tauy_out) > 999] = 0
            # ------------
            field_data_out = np.concatenate(
                (taux_out[:, :, None], tauy_out[:, :, None], temp_out), axis=2
            )
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
        print("Random sampling...")
        lb = field_data_in.shape[1]
        output_length = field_data_out.shape[1]
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
                output_length,
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
        print("End of sampling...")
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


class make_TFdataset(Dataset):
    def __init__(
        self,
        address,
        mypara,
        ngroup=None,
    ):
        self.mypara = mypara
        data_in = xr.open_dataset(address)
        self.lev = data_in["lev"].values
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.lev_range = mypara.lev_range
        self.lon_range = mypara.lon_range
        self.lat_range = mypara.lat_range

        temp_in = data_in["temperatureNor_in"][
            :,
            :,
            self.lev_range[0] : self.lev_range[1],
            self.lat_range[0] : self.lat_range[1],
            self.lon_range[0] : self.lon_range[1],
        ].values
        temp_in = np.nan_to_num(temp_in)
        temp_in[abs(temp_in) > 999] = 0
        assert mypara.input_length == temp_in.shape[1]
        if mypara.needtauxy:
            print("loading tauxy...")
            taux_in = data_in["tauxNor_in"][
                :,
                :,
                self.lat_range[0] : self.lat_range[1],
                self.lon_range[0] : self.lon_range[1],
            ].values
            taux_in = np.nan_to_num(taux_in)
            taux_in[abs(taux_in) > 999] = 0
            tauy_in = data_in["tauyNor_in"][
                :,
                :,
                self.lat_range[0] : self.lat_range[1],
                self.lon_range[0] : self.lon_range[1],
            ].values
            tauy_in = np.nan_to_num(tauy_in)
            tauy_in[abs(tauy_in) > 999] = 0
            field_data_in = np.concatenate(
                (taux_in[:, :, None], tauy_in[:, :, None], temp_in), axis=2
            )
            del temp_in, taux_in, tauy_in
        else:
            field_data_in = temp_in
            del temp_in
        # ================
        temp_out = data_in["temperatureNor_out"][
            :,
            :,
            self.lev_range[0] : self.lev_range[1],
            self.lat_range[0] : self.lat_range[1],
            self.lon_range[0] : self.lon_range[1],
        ].values
        temp_out = np.nan_to_num(temp_out)
        temp_out[abs(temp_out) > 999] = 0
        assert mypara.output_length == temp_out.shape[1]
        if mypara.needtauxy:
            print("loading tauxy...")
            taux_out = data_in["tauxNor_out"][
                :,
                :,
                self.lat_range[0] : self.lat_range[1],
                self.lon_range[0] : self.lon_range[1],
            ].values
            taux_out = np.nan_to_num(taux_out)
            taux_out[abs(taux_out) > 999] = 0
            tauy_out = data_in["tauyNor_out"][
                :,
                :,
                self.lat_range[0] : self.lat_range[1],
                self.lon_range[0] : self.lon_range[1],
            ].values
            tauy_out = np.nan_to_num(tauy_out)
            tauy_out[abs(tauy_out) > 999] = 0
            # ------------
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
        lb = field_data_in.shape[1]
        output_length = field_data_out.shape[1]
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
                output_length,
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
