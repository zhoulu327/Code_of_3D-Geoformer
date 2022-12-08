from transformer import SpaceTimeTransformer
import torch
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
from copy import deepcopy


class make_dataset_test(Dataset):
    def __init__(
        self,
        address,
        needtauxy,
        lev_range=(0, 1),
        lon_range=(0, 1),
        lat_range=(0, 1),
    ):
        """
        读取已经分组的input数据进行采样
        :param address: 输入数据路径
        :param needtauxy: 是否整合tauxy场
        :param lev_range: 海温垂向层数取值范围，不包含tauxy层
        :param lon_range: 数据经度取值范围
        :param lat_range: 数据纬度取值范围
        """
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
        ].values  # [ngroup,lb,lev,lat,lon]
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
            # --------------合并多变量
            self.dataX = np.concatenate(
                (taux[:, :, None], tauy[:, :, None], temp), axis=2
            )  # [ngroup,mon,all_lev,lat,lon]
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


def func_pre(configs, adr_model, adr_datain, adr_oridata, needtauxy):
    """
    Predict and output the prediction fields and real fields
    :param adr_model:模型路径
    :param adr_datain:模型输入数据的路径
    :param adr_oridata:未分组的原始数据路径，用于获取变量观测值和std
    :return:
    """
    lead_max = configs.output_length
    # -------------读取var_true和nino_true-------------------
    data_ori = xr.open_dataset(adr_oridata)
    temp_ori_region = data_ori["temperatureNor"][
        :,
        configs.lev_range[0] : configs.lev_range[1],
        configs.lat_range[0] : configs.lat_range[1],
        configs.lon_range[0] : configs.lon_range[1],
    ].values  # [n_mon,lev,lat,lon]
    nino34 = data_ori["nino34"].values
    # nino34可以选用原数据集中计算好的Nino3.4真值，也可以根据temp_region计算的
    # aa = np.nanmean(
    #     temp_ori_region[
    #         :,0,
    #         configs.lat_nino_relative[0] : configs.lat_nino_relative[1],
    #         configs.lon_nino_relative[0] : configs.lon_nino_relative[1],
    #     ],
    #     axis=-1,
    # )
    # nino34 = np.nanmean(aa, axis=-1)
    # del aa
    # del temp_ori
    stdtemp = data_ori["stdtemp"][configs.lev_range[0] : configs.lev_range[1]].values
    stdtemp = np.nanmean(stdtemp, axis=(1, 2))  # [n_lev]
    if needtauxy:
        taux_ori_region = data_ori["tauxNor"][
            :,
            configs.lat_range[0] : configs.lat_range[1],
            configs.lon_range[0] : configs.lon_range[1],
        ].values
        tauy_ori_region = data_ori["tauyNor"][
            :,
            configs.lat_range[0] : configs.lat_range[1],
            configs.lon_range[0] : configs.lon_range[1],
        ].values
        stdtaux = data_ori["stdtaux"].values
        stdtaux = np.nanmean(stdtaux, axis=(0, 1))
        stdtauy = data_ori["stdtauy"].values
        stdtauy = np.nanmean(stdtauy, axis=(0, 1))

        # ---------变量拼接
        var_ori_region = np.concatenate(
            (taux_ori_region[:, None], tauy_ori_region[:, None], temp_ori_region),
            axis=1,
        )  # [mon,all_lev,lat,lon]
        del taux_ori_region, tauy_ori_region, temp_ori_region
        stds = np.concatenate((stdtaux[None], stdtauy[None], stdtemp), axis=0)  # [levs]
        del stdtemp, stdtauy, stdtaux
    else:
        var_ori_region = temp_ori_region
        del temp_ori_region
        stds = stdtemp
        del stdtemp
    # -----------------------预测-------------------------
    dataCS = make_dataset_test(
        address=adr_datain,
        needtauxy=needtauxy,
        lev_range=configs.lev_range,
        lon_range=configs.lon_range,
        lat_range=configs.lat_range,
    )
    test_group = len(dataCS)
    print("预测数据集大小：")
    print(dataCS.getdatashape())
    print("所选区域：")
    print(dataCS.selectregion())
    dataloader_test = DataLoader(
        dataCS, batch_size=configs.batch_size_eval, shuffle=False
    )
    network = SpaceTimeTransformer(configs).to(configs.device)
    network.load_state_dict(torch.load(adr_model))
    print(adr_model)
    network.eval()
    if needtauxy:
        n_lev = configs.lev_range[1] - configs.lev_range[0] + 2
        sst_lev = 2
    else:
        n_lev = configs.lev_range[1] - configs.lev_range[0]
        sst_lev = 0
    var_pred = np.zeros(
        [
            test_group,
            lead_max,
            n_lev,
            configs.lat_range[1] - configs.lat_range[0],
            configs.lon_range[1] - configs.lon_range[0],
        ]
    )
    ii = 0
    iii = 0
    with torch.no_grad():
        for input_var in dataloader_test:
            out_var = network(
                src=input_var.float().to(configs.device),
                tgt=None,
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
    del network, dataCS, dataloader_test
    # np.save(
    #     "./model/var_pred_nocut_{}.npy".format(adr_model[-14:-4]),
    #     var_pred,
    # )

    # --------------------------整理结果-------------------------
    len_data = test_group - lead_max
    print("len_data是根据lead_max确定的测试数据中拥有lead=1:lead_max所有预测值的组数,具体长度根据表格确定")
    print("len_data:", len_data)
    # Obs fields
    cut_var_true = var_ori_region[(12 + lead_max) - 1 :]
    cut_var_true = cut_var_true * stds[None, :, None, None]
    cut_nino_true = nino34[(12 + lead_max) - 1 :]
    assert cut_nino_true.shape[0] == cut_var_true.shape[0] == len_data
    # Pred fields
    cut_var_pred = np.zeros(
        [lead_max, len_data, var_pred.shape[2], var_pred.shape[3], var_pred.shape[4]]
    )  # [leadmax,len,C,H,W]
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
                configs.lat_nino_relative[0] : configs.lat_nino_relative[1],
                configs.lon_nino_relative[0] : configs.lon_nino_relative[1],
            ],
            axis=(1, 2),
        )
    assert cut_var_pred.shape[1] == cut_var_true.shape[0]
    print("返回值已×std")
    return (
        cut_var_pred,
        cut_var_true,
        cut_nino_pred,
        cut_nino_true,
    )
