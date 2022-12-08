import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset
import xarray as xr
import random

"""
online读取训练集
"""


class make_dataset(IterableDataset):
    def __init__(
        self,
        address,
        look_back=1,
        pre_len=1,
        all_group=1,
        needtauxy=False,
        lev_range=(0, 1),
        lon_range=(0, 1),
        lat_range=(0, 1),
        needextrenino=False,
        adr_extrenino=None,
    ):
        """
        :param address: 训练集地址
        :param look_back: 输入数据长度
        :param pre_len: 输出数据长度
        :param all_group: 总共取多少组
        :param lev_range: 海温场需要的lev范围
        :param needtauxy: 是否读取风应力
        :param lon_range: 空间场范围
        :param lat_range:
        :param needextrenino:是否在训练集中加入更多的极端El Nino的个例
        :param adr_extrenino：极端el nino数据集地址
        """
        data_in = xr.open_dataset(address)
        self.lev = data_in["lev"].values
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.lev_range = lev_range
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.look_back = look_back
        self.pre_len = pre_len
        self.all_group = all_group
        self.needextrenino = needextrenino
        # nino34 = data_in["nino34"].values
        temp = data_in["temperatureNor"][
            :,
            :,
            lev_range[0] : lev_range[1],
            lat_range[0] : lat_range[1],
            lon_range[0] : lon_range[1],
        ].values
        temp = np.nan_to_num(temp)
        temp[abs(temp) > 999] = 0
        # stdtemp = data_in["stdtemp"][:, lev_range[0] : lev_range[1]].values
        # stdtemp = np.nanmean(stdtemp, axis=(2, 3))  # [n_model,n_lev]
        if needtauxy:
            print("已使用tauxy...")
            taux = data_in["tauxNor"][
                :, :, lat_range[0] : lat_range[1], lon_range[0] : lon_range[1]
            ].values
            taux = np.nan_to_num(taux)
            taux[abs(taux) > 999] = 0
            tauy = data_in["tauyNor"][
                :, :, lat_range[0] : lat_range[1], lon_range[0] : lon_range[1]
            ].values
            tauy = np.nan_to_num(tauy)
            tauy[abs(tauy) > 999] = 0
            # stdtaux = data_in["stdtaux"].values
            # stdtaux = np.nanmean(stdtaux, axis=(1, 2))  # [n_model]
            # stdtauy = data_in["stdtauy"].values
            # stdtauy = np.nanmean(stdtauy, axis=(1, 2))
            self.field_data = np.concatenate(
                (taux[:, :, None], tauy[:, :, None], temp), axis=2
            )  # [n_model,mon,all_lev,lat,lon]
            # stds = np.concatenate((stdtaux[:, None], stdtauy[:, None], stdtemp), axis=1)
            del temp, taux, tauy
        else:
            self.field_data = temp
            # stds = stdtemp
            del temp
        if needextrenino:
            print("极端El Nino样本已加入…… \n！！注意：目前极端事件样本的海温数据只支持取全水深！！")
            data_ex = xr.open_dataset(adr_extrenino)
            if needtauxy:
                st = 0
            else:
                st = 2
            self.exdataX = data_ex["InputDataNor"][
                :,
                :,
                st:,
                self.lat_range[0] : self.lat_range[1],
                self.lon_range[0] : self.lon_range[1],
            ].values
            self.exdataY = data_ex["OutputDataNor"][
                :,
                :,
                st:,
                self.lat_range[0] : self.lat_range[1],
                self.lon_range[0] : self.lon_range[1],
            ].values
            self.exdataX = np.nan_to_num(self.exdataX)
            self.exdataX[abs(self.exdataX) > 999] = 0
            self.exdataY = np.nan_to_num(self.exdataY)
            self.exdataY[abs(self.exdataY) > 999] = 0
            assert self.exdataX.shape[1] == look_back
            assert self.exdataY.shape[1] == pre_len

    def __iter__(self):
        st_min = self.look_back - 1
        ed_max = self.field_data.shape[1] - self.pre_len
        if self.needextrenino:
            for i in range(self.exdataX.shape[0]):
                dataX = self.exdataX[i]
                dataY = self.exdataY[i]
                yield dataX, dataY
        for i in range(self.all_group):
            rd_m = random.randint(0, self.field_data.shape[0] - 1)
            rd = random.randint(st_min, ed_max - 1)
            dataX = self.field_data[rd_m, rd - self.look_back + 1 : rd + 1]
            dataY = self.field_data[rd_m, rd + 1 : rd + self.pre_len + 1]
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
    def __init__(
        self,
        address,
        look_back,
        pre_len,
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
        assert look_back == temp_in.shape[1]
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
        assert pre_len == temp_out.shape[1]
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
        pre_len = field_data_out.shape[1]
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
                pre_len,
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


class make_TFdataset(Dataset):
    def __init__(
        self,
        address,
        config,
        ngroup=None,
    ):
        self.config = config
        data_in = xr.open_dataset(address)
        self.lev = data_in["lev"].values
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.lev_range = config.lev_range
        self.lon_range = config.lon_range
        self.lat_range = config.lat_range

        temp_in = data_in["temperatureNor_in"][
            :,
            :,
            self.lev_range[0] : self.lev_range[1],
            self.lat_range[0] : self.lat_range[1],
            self.lon_range[0] : self.lon_range[1],
        ].values
        temp_in = np.nan_to_num(temp_in)
        temp_in[abs(temp_in) > 999] = 0
        assert config.input_length == temp_in.shape[1]
        if config.needtauxy:
            print("已使用tauxy...")
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
            self.lev_range[0] : self.lev_range[1],
            self.lat_range[0] : self.lat_range[1],
            self.lon_range[0] : self.lon_range[1],
        ].values
        temp_out = np.nan_to_num(temp_out)
        temp_out[abs(temp_out) > 999] = 0
        assert config.output_length == temp_out.shape[1]
        if config.needtauxy:
            print("已使用tauxy...")
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
        pre_len = field_data_out.shape[1]
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
                pre_len,
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
