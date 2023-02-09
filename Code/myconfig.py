import torch


class Mypara:
    def __init__(self):
        pass


mypara = Mypara()
mypara.device = torch.device("cuda:0")
mypara.batch_size_train = 8
mypara.batch_size_eval = 10
mypara.num_epochs = 40
mypara.TFnum_epochs = 20
mypara.TFlr = 1.5e-5
mypara.early_stopping = True
mypara.patience = 4
mypara.warmup = 2000

# data related
mypara.adr_pretr = (
    "./data/up150_tauxy/CMIP6_separate_model_up150m_tauxy_Nor_kb.nc"
)
mypara.interval = 4
mypara.TraindataProportion = 0.9
mypara.all_group = 13000
mypara.adr_eval = (
    "./data/up150_tauxy/SODA_ORAS_group_temp_tauxy_before1979_kb.nc"
)
mypara.needtauxy = True
mypara.input_channal = 7  # n_lev of 3D temperature
mypara.output_channal = 7
mypara.input_length = 12
mypara.output_length = 20
mypara.lev_range = (1, 8)
mypara.lon_range = (45, 165)
mypara.lat_range = (0, 51)
# nino34 region
mypara.lon_nino_relative = (49, 75)
mypara.lat_nino_relative = (15, 36)
# patch size
mypara.patch_size = (3, 4)
mypara.H0 = int((mypara.lat_range[1] - mypara.lat_range[0]) / mypara.patch_size[0])
mypara.W0 = int((mypara.lon_range[1] - mypara.lon_range[0]) / mypara.patch_size[1])
mypara.emb_spatial_size = mypara.H0 * mypara.W0

# model
mypara.model_savepath = "./model/"
mypara.seeds = 1
mypara.d_size = 256
mypara.nheads = 4
mypara.dim_feedforward = 512
mypara.dropout = 0.2
mypara.num_encoder_layers = 4
mypara.num_decoder_layers = 4
