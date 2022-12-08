import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.collections import PolyCollection
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.patch import geos_to_path
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import itertools
from copy import deepcopy
from sklearn.utils import resample


def runmean(data, n_run):
    """
    使用前面的数据做滑动平均，如n_run=3
    则使用i-2,i-1和i作为i点的滑动平均结果
    :param data: 待滑动平均的数据
    :param n_run: n点滑动平均
    :return: n点滑动平均后的序列
    """
    ll = data.shape[0]
    data_run = np.zeros([ll])
    for i in range(ll):
        if i < (n_run - 1):
            data_run[i] = np.nanmean(data[0 : i + 1])
        else:
            data_run[i] = np.nanmean(data[i - n_run + 1 : i + 1])
    return data_run


def cal_ACC(var_pred, var_true, isrunmean=False):
    """
    Args:
        var_pred: [nmon,H,W]
        var_true: [nmon,H,W]
    Returns:ACC:[H,W]
    """
    var_true2 = deepcopy(var_true)
    var_pred2 = deepcopy(var_pred)
    if isrunmean:
        var_true_rm = np.zeros(var_true2.shape)
        var_pred_rm = np.zeros(var_pred2.shape)
        for i in range(var_true2.shape[-2]):
            for j in range(var_true2.shape[-1]):
                var_true_rm[:, i, j] = runmean(var_true2[:, i, j], 3)
                var_pred_rm[:, i, j] = runmean(var_pred2[:, i, j], 3)
    else:
        var_true_rm = var_true2
        var_pred_rm = var_pred2
    mean_pre = np.mean(var_pred_rm, 0)
    mean_true = np.mean(var_true_rm, 0)
    cha_pre = var_pred_rm - mean_pre
    cha_true = var_true_rm - mean_true
    cov = np.mean(cha_pre * cha_true, 0)
    std_pre = np.sqrt(np.mean(cha_pre ** 2, 0))
    std_true = np.sqrt(np.mean(cha_true ** 2, 0))
    return cov / (std_pre * std_true)


# -----------pre_nino第二维是起报月份
def cal_ninoskill2(pre_nino_all, real_nino):
    """
    :param pre_nino_all: [n_yr,start_mon,lead_max]
    :param real_nino: [n_yr,12]
    :return: nino_skill: [12,lead_max]
    """
    lead_max = pre_nino_all.shape[2]
    nino_skill = np.zeros([12, lead_max])
    for ll in range(lead_max):
        lead = ll + 1
        dd = deepcopy(pre_nino_all[:, :, ll])
        for mm in range(12):
            bb = dd[:, mm]
            st_m = mm + 1
            terget = st_m + lead
            if 12 < terget <= 24:
                terget = terget - 12
            elif terget > 24:
                terget = terget - 24
            aa = deepcopy(real_nino[:, terget - 1])
            nino_skill[mm, ll] = np.corrcoef(aa, bb)[0, 1]
    return nino_skill


def fig_3d(
    fig,
    ax,
    x,
    y,
    z,
    var_list,
    cf_vmin,
    cf_vmax,
    cf_num,
    proj,
    cmap,
    needcbar,
    cbar_location,
    cbar_interval,
    cbar_label,
    orientation="vertical",
    title=None,
    fontsize=10,
):
    """
    Args:
        fig: fig句柄，用于向上加坐标轴
        ax: 子图ax句柄
        x: meshgrid后的lon
        y: meshgrid后的lat
        z: lev深度值列表
        var_list: [n_lev,lat,lon]
        cf_vmin: 绘制contourf时填色的最大值,也决定了colorbar的范围
        cf_vmax: 绘制contourf时填色的最小值
        cbar_interval: colorbar上色彩分级间隔
        cf_num: 绘制contourf时色彩分级数，越大contourf越细腻
        proj: 地图投影方式
        needcbar: 是否绘制colorbar
        cbar_location: cbar位置，列表形式
        cmap: colormap名称
        cbar_label: colorbar上的label
        fontsize: colorbar上数字、cbar label、ax title字号
        orientation: cbar放置方式，'horizontal' or "vertical"
        title: ax子图的title

    Returns: fig, ax

    """
    concat = lambda iterable: list(itertools.chain.from_iterable(iterable))
    if needcbar:
        # --------------设置colorbar
        # 绘制一个填色图，用于获取其colorbar范围，并用后销毁
        pp = deepcopy(var_list[0])
        aa = plt.figure(3).add_subplot(111)
        bb = aa.contourf(
            x,
            y,
            pp,
            levels=np.arange(cf_vmin, cf_vmax + 0.01, cbar_interval),
            cmap=cmap,
        )

        cbar_ax = fig.add_axes(cbar_location)
        cbar = plt.colorbar(bb, cax=cbar_ax, extend="both", orientation=orientation)
        cbar.ax.tick_params(labelsize=fontsize)
        font = {
            "family": "Arial",
            "color": "k",
            "weight": "normal",
            "size": fontsize,
        }
        cbar.set_label(cbar_label, fontdict=font)  # 设置colorbar标签和字体
        aa.set_visible(False)  # 用后隐藏
        plt.close()
        del aa, bb, pp
    # ---------------------在3D画板上绘制填色图
    for i in range(len(z)):
        ax.contourf(
            x,
            y,
            var_list[i],
            cf_num,
            vmin=cf_vmin,
            vmax=cf_vmax,
            cmap=cmap,
            zdir="z",
            offset=z[i],
        )

    # ----------新建一个图，绘制地图并迁移，用后销毁
    proj_ax = plt.figure(2).add_subplot(111, projection=proj)
    proj_ax.set_xlim(ax.get_xlim())
    proj_ax.set_ylim(ax.get_ylim())
    target_projection = proj_ax.projection
    feature = cfeature.NaturalEarthFeature("physical", "land", "110m")
    geoms = feature.geometries()
    boundary = proj_ax._get_extent_geom()
    geoms = [target_projection.project_geometry(geom, feature.crs) for geom in geoms]
    geoms2 = []
    for i in range(len(geoms)):
        if geoms[i].is_valid:
            geoms2.append(geoms[i])
    geoms = geoms2
    del geoms2
    geoms = [boundary.intersection(geom) for geom in geoms]
    paths = concat(geos_to_path(geom) for geom in geoms)  # geom转path
    polys = concat(path.to_polygons() for path in paths)  # path转poly
    # -------------将地图加入到3Dfig中
    for i in range(len(z)):
        lc = PolyCollection(polys, edgecolor="gray", facecolor="gray", closed=False)
        ax.add_collection3d(lc, zs=z[i])
        del lc
    proj_ax.spines["geo"].set_visible(False)  # 解除掉用于确定地图的子图
    plt.close()
    # --------------添加黑框
    for i in range(len(z)):
        ax.plot(
            [x[0, 0], x[0, -1], x[0, -1], x[0, 0], x[0, 0]],
            [y[0, 0], y[0, 0], y[-1, 0], y[-1, 0], y[0, 0]],
            [
                z[i],
                z[i],
                z[i],
                z[i],
                z[i],
            ],
            "-k",
            linewidth=1,
        )
    _ = ax.text(x=x[0, 0], y=y[-1, 0], z=z[0] - 9, s=title, fontsize=fontsize)
    return fig, ax


def tiansetu(
    ax,
    x,
    y,
    variate,
    cflevel,
    crlevel,
    crcolor,
    crxs,
    crdgx_fontsize,
    x_min,
    x_max,
    xtick,
    xtick_label,
    xlabel,
    y_min,
    y_max,
    ytick,
    ytick_label,
    ylabel,
    invert_y,
    cmap,
    extend,
    title,
    label_fontsize,
    gridon=False,
):
    if cflevel is not None:
        cf = ax.contourf(x, y, variate, levels=cflevel, extend=extend, cmap=cmap)
    if crlevel is not None:
        cr = ax.contour(x, y, variate, levels=crlevel, colors=crcolor, linewidths=0.5)
        cr.levels = crlevel
        if crxs is not None:
            ax.clabel(
                cr,
                cr.levels,
                fmt="%." + str(crxs) + "f",
                inline=True,
                fontsize=crdgx_fontsize,
            )

    # ax.spines["right"].set_color("none")  # 设置右边和顶边为无色
    # ax.spines["top"].set_color("none")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(ytick)  # y轴坐标刻度
    ax.set_yticklabels(ytick_label)
    ax.set_xticks(xtick)
    ax.set_xticklabels(xtick_label)
    ax.set_xlabel(xlabel, fontsize=label_fontsize + 1)
    ax.tick_params(axis="x", rotation=0, labelsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize + 1)
    ax.tick_params(axis="y", rotation=0, labelsize=label_fontsize)
    if gridon:
        ax.grid(alpha=0.4)
    if invert_y:
        ax.invert_yaxis()
    if title is not None:
        if invert_y:
            ax.text(x=x_min, y=y_min - 0.3, s=title, fontsize=label_fontsize + 1)
        else:
            ax.text(x=x_min, y=y_max + 3, s=title, fontsize=label_fontsize + 1)
    return ax, cf


def oceanmap(
    ax,
    lon,
    lat,
    verb,
    min_lon,
    max_lon,
    min_lat,
    max_lat,
    lon_iv,
    lat_iv,
    cflevel=None,
    crlist=None,
    crcolor="darkgreen",
    xs=None,
    extend="both",
    cmap="RdBu_r",
    title=None,
    fontsize_dzx=10,
    fontsize_lonlat=15,
):
    ax.set_xticks(np.arange(min_lon, max_lon, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(min_lat, max_lat, 10), crs=ccrs.PlateCarree())
    ax.tick_params(axis="x", rotation=0, labelsize=fontsize_lonlat)
    ax.tick_params(axis="y", rotation=0, labelsize=fontsize_lonlat)
    if title is not None:
        ax.set_title(title, fontsize=fontsize_lonlat)
    # zero_direction_label用来设置经度的0度加不加E和W
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    # 设置地图属性:加载国界、海岸线、河流、湖泊
    land = cfeature.NaturalEarthFeature(
        "physical", "land", "110m", color="gray", edgecolor="face"
    )
    ax.add_feature(land)
    # ax.add_feature(cfeature.COASTLINE, linewidth=1.5, zorder=1, color="k")
    # ax.add_feature(cfeature.RIVERS.with_scale("50m"), zorder=1)
    # ax.add_feature(cfeature.LAKES.with_scale("50m"), zorder=1)
    # 经纬度间隔
    x_major_locator = MultipleLocator(lon_iv)
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(lat_iv)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.plot(
        [180, 180],
        [lat[0], lat[-1]],
        color="k",
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
    )
    ax.plot(
        [lon[0], lon[-1]],
        [0, 0],
        color="k",
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
    )
    # 设置网格点属性
    # gl = ax.gridlines(
    #     crs=ccrs.PlateCarree(),
    #     draw_labels=True,
    #     linewidth=1.2,
    #     color="k",
    #     alpha=0.5,
    #     linestyle="--",
    # )
    # gl.top_labels = False  # 关闭顶端标签
    # gl.right_labels = False  # 关闭右侧标签
    # gl.bottom_labels = False
    # gl.left_labels = False
    #
    # gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度格式
    # gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度格式
    if cflevel is not None:
        cf = ax.contourf(
            lon,
            lat,
            verb,
            levels=cflevel,
            extend=extend,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
        )
    if crlist is not None:
        cr = ax.contour(
            lon,
            lat,
            verb,
            levels=crlist,
            colors=crcolor,
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
        )
        cr.levels = crlist
        if xs is not None:
            ax.clabel(
                cr,
                cr.levels,
                fmt="%." + str(xs) + "f",
                inline=True,
                fontsize=fontsize_dzx,
            )
    if cflevel is not None:
        return ax, cf
    else:
        return ax, cr


def oceanmap_pcolor(
    ax,
    lon,
    lat,
    verb,
    vmin,
    vmax,
    min_lon,
    max_lon,
    min_lat,
    max_lat,
    lon_iv,
    lat_iv,
    cmap="RdBu_r",
    title=None,
    fontsize_lonlat=15,
):
    ax.set_xticks(np.arange(min_lon, max_lon, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(min_lat, max_lat, 10), crs=ccrs.PlateCarree())
    ax.tick_params(axis="x", rotation=0, labelsize=fontsize_lonlat)
    ax.tick_params(axis="y", rotation=0, labelsize=fontsize_lonlat)
    ax.set_title(title, fontsize=fontsize_lonlat + 2)
    # zero_direction_label用来设置经度的0度加不加E和W
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    # 设置地图属性:加载国界、海岸线、河流、湖泊
    land = cfeature.NaturalEarthFeature(
        "physical", "land", "110m", color="gray", edgecolor="face"
    )
    ax.add_feature(land)
    # 经纬度间隔
    ax.xaxis.set_major_locator(MultipleLocator(lon_iv))
    ax.yaxis.set_major_locator(MultipleLocator(lat_iv))
    ax.plot(
        [180, 180],
        [lat[0], lat[-1]],
        color="k",
        linewidth=1.2,
        transform=ccrs.PlateCarree(),
    )
    ax.plot(
        [lon[0], lon[-1]],
        [0, 0],
        color="k",
        linewidth=1.2,
        transform=ccrs.PlateCarree(),
    )
    # 设置网格点属性
    # gl = ax.gridlines(
    #     crs=ccrs.PlateCarree(),
    #     draw_labels=True,
    #     linewidth=1.2,
    #     color="k",
    #     alpha=0.5,
    #     linestyle="--",
    # )
    # gl.top_labels = False  # 关闭顶端标签
    # gl.right_labels = False  # 关闭右侧标签
    # gl.bottom_labels = False
    # gl.left_labels = False
    #
    # gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度格式
    # gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度格式
    cr = ax.pcolor(
        lon,
        lat,
        verb,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        # edgecolors="k",
        # linewidths=0.1,
        transform=ccrs.PlateCarree(),
    )
    return ax, cr


def mycontour(
    ax,
    x,
    y,
    z,
    level,
    linecolor="k",
    linewidth=1,
    specifiedlevel=None,
    xs=1,
    fontcolor="k",
    fontsize=8,
    inline=True,
    needbox=True,
):
    cr = ax.contour(
        x,
        y,
        z,
        levels=level,
        colors=linecolor,
        linewidths=linewidth,
    )
    cr.levels = level
    if specifiedlevel is None:
        specifiedlevel = cr.levels
    labels = ax.clabel(
        cr,
        specifiedlevel,
        fmt="%.{}f".format(xs),
        inline=inline,
        fontsize=fontsize,
        colors=fontcolor,
    )
    if needbox:
        for l in labels:
            l.set_bbox({"fc": "w", "ec": "w"})
    return ax, cr


def cal_COR_RMSE_MAE_and_Bootstrap(
    dataA, dataB, lead_max, bootstrap_repeat, quantile_L, quantile_U
):
    from sklearn.metrics import mean_squared_error  # 均方误差
    from sklearn.metrics import mean_absolute_error  # 平方绝对误差
    from sklearn.utils import resample

    assert dataB.shape[0] == lead_max
    assert dataA.shape[0] == dataB.shape[1]
    corr = np.zeros([lead_max])
    rmse = np.zeros([lead_max])
    mae = np.zeros([lead_max])
    for l in range(lead_max):
        corr[l] = np.corrcoef(deepcopy(dataA), deepcopy(dataB[l]))[0, 1]
        rmse[l] = (mean_squared_error(deepcopy(dataA), deepcopy(dataB[l]))) ** 0.5
        mae[l] = mean_absolute_error(deepcopy(dataA), deepcopy(dataB[l]))
    corrBootS = np.zeros([lead_max, bootstrap_repeat])
    rmseBootS = np.zeros([lead_max, bootstrap_repeat])
    maeBootS = np.zeros([lead_max, bootstrap_repeat])
    for i in range(lead_max):
        for j in range(bootstrap_repeat):
            dataB1 = resample(
                deepcopy(dataB[i]),
                replace=True,
                n_samples=dataA.shape[0],
                random_state=j,
            )
            dataA1 = resample(
                deepcopy(dataA),
                replace=True,
                n_samples=dataA.shape[0],
                random_state=j,
            )
            corrBootS[i, j] = np.corrcoef(dataA1, dataB1)[0, 1]
            rmseBootS[i, j] = (mean_squared_error(dataA1, dataB1)) ** 0.5
            maeBootS[i, j] = mean_absolute_error(dataA1, dataB1)
            del dataA1, dataB1
    corrBootS_diff = corrBootS - corr[:, None]
    rmseBootS_diff = rmseBootS - rmse[:, None]
    maeBootS_diff = maeBootS - mae[:, None]
    R_LU = np.zeros([lead_max, 2])
    RMSE_LU = np.zeros([lead_max, 2])
    MAE_LU = np.zeros([lead_max, 2])
    R_LU[:, 0] = corr - np.percentile(deepcopy(corrBootS_diff), q=quantile_L, axis=1)
    R_LU[:, 1] = corr - np.percentile(deepcopy(corrBootS_diff), q=quantile_U, axis=1)
    RMSE_LU[:, 0] = rmse - np.percentile(deepcopy(rmseBootS_diff), q=quantile_L, axis=1)
    RMSE_LU[:, 1] = rmse - np.percentile(deepcopy(rmseBootS_diff), q=quantile_U, axis=1)
    MAE_LU[:, 0] = mae - np.percentile(deepcopy(maeBootS_diff), q=quantile_L, axis=1)
    MAE_LU[:, 1] = mae - np.percentile(deepcopy(maeBootS_diff), q=quantile_U, axis=1)
    return corr, R_LU, rmse, RMSE_LU, mae, MAE_LU
