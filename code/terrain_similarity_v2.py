# pip install rioxarray rasterio numpy scipy scikit-learn pyflwdir pandas matplotlib glob2 xarray numba

import os
import glob
import logging
from logging.handlers import RotatingFileHandler

import rioxarray as rxr
import xarray as xr
import numpy as np
import pandas as pd

from rioxarray.merge import merge_arrays
from scipy.stats import skew, kurtosis
from scipy.ndimage import sobel

import pyflwdir  # v0.5.10: slope is a function (numba CPUDispatcher)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt


# ----------------------
# 日志配置
# ----------------------
def setup_logger():
    """配置日志：同时输出到文件和控制台，支持日志轮转"""
    LOG_DIR = "/home/huxun/01_terr_sim/logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, "terrain_similarity.log")

    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger("terrain_similarity")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


logger = setup_logger()


# ----------------------
# 1. 配置参数
# ----------------------
SRTM_PATH = "/home/huxun/00_icing_map/data/dem/SRTM_30m_China/*.tif"
TARGET_EXTENT = (108.6147, 109.5914, 25.4937, 26.36719)  # (min_lon, max_lon, min_lat, max_lat)
WRF_RES = 200
WINDOW_SIZE_M = 20000
STEP_SIZE_M = 5000
TARGET_CRS = "EPSG:32648"
RESULT_ROOT = "/home/huxun/01_terr_sim/result"
os.makedirs(RESULT_ROOT, exist_ok=True)


# ----------------------
# 剖面曲率
# ----------------------
def calculate_profile_curvature(elev, cellsize):
    dx = sobel(elev, axis=1) / (8 * cellsize)
    dy = sobel(elev, axis=0) / (8 * cellsize)

    dxx = sobel(dx, axis=1) / (8 * cellsize)
    dyy = sobel(dy, axis=0) / (8 * cellsize)
    dxy = sobel(dx, axis=0) / (8 * cellsize)

    p2 = dy**2
    q2 = dx**2
    pq = dy * dx

    denom = (p2 + q2) ** (3 / 2)
    denom = np.where(denom == 0, np.finfo(float).eps, denom)

    curv_profile = -(p2 * dyy + 2 * pq * dxy + q2 * dxx) / denom
    return curv_profile


# ----------------------
# 2. 数据预处理（关键修复：不再用 xr.open_mfdataset 合并 GeoTIFF）
# ----------------------
def preprocess_srtm(srtm_path_pattern, target_extent, wrf_res, target_crs):
    """
    读取SRTM瓦片、合并、裁剪、投影、重采样到WRF分辨率
    修复点：
      - 用 rioxarray.merge_arrays 合并瓦片，避免 open_mfdataset 由于重复x/y坐标报错
      - 用 rio.clip_box 做经纬度裁剪
      - 用 rio.reproject(..., resolution=...) 做重采样
    """
    min_lon, max_lon, min_lat, max_lat = target_extent

    logger.info("正在获取SRTM 30m数据文件列表...")
    srtm_files = sorted(glob.glob(srtm_path_pattern))
    if not srtm_files:
        raise FileNotFoundError(f"在路径 {srtm_path_pattern} 下未找到任何.tif文件！请检查路径是否正确。")
    logger.info(f"找到 {len(srtm_files)} 个SRTM瓦片文件，开始读取合并...")

    # 逐瓦片打开成 DataArray (y,x)
    das = []
    for fp in srtm_files:
        da = rxr.open_rasterio(fp, masked=True).squeeze(drop=True)
        # 若瓦片没写CRS，默认认为SRTM为WGS84经纬度
        if da.rio.crs is None:
            da = da.rio.write_crs("EPSG:4326")
        das.append(da)

    logger.info("正在合并SRTM瓦片（merge_arrays）...")
    da_mosaic = merge_arrays(das)

    logger.info("正在裁剪目标区域...")
    da_clip = da_mosaic.rio.clip_box(minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat)

    logger.info(f"正在将数据投影到 {target_crs} ...")
    da_proj = da_clip.rio.reproject(target_crs)

    logger.info(f"正在重采样到 {wrf_res}m 分辨率...")
    da_resampled = da_proj.rio.reproject(
        target_crs,
        resolution=wrf_res,
        resampling=rxr.enums.Resampling.average,
    )

    logger.info("正在处理数据空洞和异常值（SRTM无数据值-32768）...")
    da_resampled = da_resampled.where(
        da_resampled != -32768,
        da_resampled.rolling(x=3, y=3, center=True, min_periods=1).mean(),
    )

    elev = da_resampled.values
    x_coords = da_resampled.x.values
    y_coords = da_resampled.y.values

    logger.info(f"数据预处理完成！高程数据形状：{elev.shape}（y×x）")
    return elev, x_coords, y_coords


# ----------------------
# 3. 滑动窗口
# ----------------------
def generate_sliding_windows(elev, x_coords, y_coords, window_size_m, step_size_m, wrf_res):
    window_size = int(window_size_m / wrf_res)
    step_size = int(step_size_m / wrf_res)

    y_starts = np.arange(0, elev.shape[0] - window_size + 1, step_size)
    x_starts = np.arange(0, elev.shape[1] - window_size + 1, step_size)

    window_info = []
    for y_start in y_starts:
        for x_start in x_starts:
            center_y = y_coords[y_start + window_size // 2]
            center_x = x_coords[x_start + window_size // 2]
            window_info.append((y_start, x_start, center_x, center_y))

    window_df = pd.DataFrame(window_info, columns=["y_start", "x_start", "center_x", "center_y"])
    logger.info(f"滑动窗口生成完成！共生成{len(window_df)}个20km×20km子区域")
    return window_df, window_size


# ----------------------
# 4. 特征提取（修复点：aspect 不依赖 pyflwdir，避免版本缺失）
# ----------------------
def extract_terrain_features(args):
    window_idx, window_df, elev, window_size, wrf_res = args
    try:
        y_start = int(window_df.loc[window_idx, "y_start"])
        x_start = int(window_df.loc[window_idx, "x_start"])
        window_elev = elev[y_start : y_start + window_size, x_start : x_start + window_size]

        if np.isnan(window_elev).sum() > window_elev.size * 0.1:
            return (window_idx, None)

        elev_mean = np.nanmean(window_elev)
        elev_anomaly = window_elev - elev_mean

        anom_mean = np.nanmean(elev_anomaly)
        anom_std = np.nanstd(elev_anomaly)
        anom_range = np.nanmax(elev_anomaly) - np.nanmin(elev_anomaly)
        anom_skew = skew(elev_anomaly.flatten(), nan_policy="omit")
        anom_kurt = kurtosis(elev_anomaly.flatten(), nan_policy="omit")

        # 坡度：pyflwdir 0.5.10 OK
        slope = pyflwdir.slope(window_elev, cellsize=wrf_res)

        # 坡向：用 sobel 自算（避免 pyflwdir.aspect 可能不存在）
        dx = sobel(window_elev, axis=1) / (8 * wrf_res)
        dy = sobel(window_elev, axis=0) / (8 * wrf_res)
        aspect = (np.degrees(np.arctan2(-dx, dy)) + 360) % 360

        slope_mean = np.nanmean(slope)
        slope_std = np.nanstd(slope)
        aspect_mean = np.nanmean(aspect)
        aspect_std = np.nanstd(aspect)

        # 自相关
        elev_flat1 = elev_anomaly[:, :-1].flatten()
        elev_flat2 = elev_anomaly[:, 1:].flatten()
        elev_valid = ~(np.isnan(elev_flat1) | np.isnan(elev_flat2))
        elev_corr = (
            np.corrcoef(elev_flat1[elev_valid], elev_flat2[elev_valid])[0, 1] if np.sum(elev_valid) > 10 else 0.0
        )

        slope_flat1 = slope[:, :-1].flatten()
        slope_flat2 = slope[:, 1:].flatten()
        slope_valid = ~(np.isnan(slope_flat1) | np.isnan(slope_flat2))
        slope_corr = (
            np.corrcoef(slope_flat1[slope_valid], slope_flat2[slope_valid])[0, 1] if np.sum(slope_valid) > 10 else 0.0
        )

        curv_profile = calculate_profile_curvature(window_elev, wrf_res)
        curv_mean = np.nanmean(curv_profile)

        features = np.array(
            [
                anom_mean,
                anom_std,
                anom_range,
                anom_skew,
                anom_kurt,
                slope_mean,
                slope_std,
                aspect_mean,
                aspect_std,
                elev_corr,
                slope_corr,
                curv_mean,
            ],
            dtype=float,
        )
        return (window_idx, features)

    except Exception as e:
        logger.error(f"窗口{window_idx}处理失败：{str(e)}")
        return (window_idx, None)


# ----------------------
# 5. 重叠判断
# ----------------------
def is_overlap(idx1, idx2, valid_df, window_size):
    """
    判断两个窗口是否重叠超过50%
    返回：True（重叠）/False（不重叠）
    """
    y1, x1 = valid_df.loc[idx1, ["y_start", "x_start"]]
    y2, x2 = valid_df.loc[idx2, ["y_start", "x_start"]]
    y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)

    y1_end, x1_end = y1 + window_size, x1 + window_size
    y2_end, x2_end = y2 + window_size, x2 + window_size

    overlap_y_start = max(y1, y2)
    overlap_y_end = min(y1_end, y2_end)
    overlap_x_start = max(x1, x2)
    overlap_x_end = min(x1_end, x2_end)

    if overlap_y_end <= overlap_y_start or overlap_x_end <= overlap_x_start:
        return False

    overlap_area = (overlap_y_end - overlap_y_start) * (overlap_x_end - overlap_x_start)
    window_area = window_size * window_size
    return overlap_area / window_area > 0.5


# ----------------------
# 主程序执行流程
# ----------------------
if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("开始执行地形相似度分析程序")
        logger.info(f"pyflwdir version: {getattr(pyflwdir, '__version__', 'unknown')}")
        logger.info("=" * 60)

        # 步骤1：数据预处理
        elev, x_coords, y_coords = preprocess_srtm(
            srtm_path_pattern=SRTM_PATH,
            target_extent=TARGET_EXTENT,
            wrf_res=WRF_RES,
            target_crs=TARGET_CRS
        )

        # 步骤2：生成滑动窗口
        window_df, window_size = generate_sliding_windows(
            elev=elev,
            x_coords=x_coords,
            y_coords=y_coords,
            window_size_m=WINDOW_SIZE_M,
            step_size_m=STEP_SIZE_M,
            wrf_res=WRF_RES
        )

        # 步骤3：多进程提取地形特征
        logger.info("正在提取12维地形特征（多进程加速）...")
        cpu_cores = max(1, cpu_count() - 2)
        logger.info(f"使用 {cpu_cores} 个CPU核心进行多进程计算")

        args_list = [(i, window_df, elev, window_size, WRF_RES) for i in range(len(window_df))]
        with Pool(cpu_cores) as pool:
            feature_results = pool.map(extract_terrain_features, args_list)

        valid_indices = []
        feature_matrix = []
        for idx, feat in feature_results:
            if feat is not None:
                valid_indices.append(idx)
                feature_matrix.append(feat)

        if not feature_matrix:
            raise ValueError("未提取到有效地形特征，请检查数据是否正常！")

        feature_matrix = np.array(feature_matrix)
        valid_df = window_df.loc[valid_indices].reset_index(drop=True)
        logger.info(f"地形特征提取完成！有效子区域数：{len(valid_df)}")

        # 步骤4：标准化 + PCA
        logger.info("正在标准化特征并执行PCA降维（保留95%方差）...")
        scaler = StandardScaler()
        feature_scaled = scaler.fit_transform(feature_matrix)

        pca = PCA(n_components=0.95, random_state=42)
        feature_pca = pca.fit_transform(feature_scaled)
        logger.info(f"PCA降维完成！原始12维特征→降维后{feature_pca.shape[1]}维特征")

        # 步骤5：KMeans粗筛 + 相似度计算
        logger.info("正在计算地形相似度（KMeans粗筛+余弦相似度精算）...")
        n_clusters = min(10, max(2, len(valid_df) // 2))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(feature_pca)
        valid_df["cluster"] = clusters

        best_score = 0.0
        best_pair = None

        for cluster in np.unique(clusters):
            cluster_indices = valid_df[valid_df["cluster"] == cluster].index.tolist()
            if len(cluster_indices) < 2:
                continue

            cluster_pca = feature_pca[cluster_indices]
            cos_sim = cosine_similarity(cluster_pca)

            euclid_dist = np.linalg.norm(cluster_pca[:, None] - cluster_pca, axis=2)
            euclid_dist_norm = (euclid_dist - euclid_dist.min()) / (
                euclid_dist.max() - euclid_dist.min() + np.finfo(float).eps
            )

            combined_score = cos_sim * 0.8 + (1 - euclid_dist_norm) * 0.2

            for i in range(len(cluster_indices)):
                for j in range(i + 1, len(cluster_indices)):
                    idx1 = cluster_indices[i]
                    idx2 = cluster_indices[j]

                    if is_overlap(idx1, idx2, valid_df, window_size):
                        continue

                    if combined_score[i, j] > best_score:
                        best_score = float(combined_score[i, j])
                        best_pair = (idx1, idx2)

        # 步骤6：结果输出 + 保存 + 可视化
        logger.info("\n" + "=" * 60)
        if best_pair is not None:
            idx1, idx2 = best_pair
            area1_info = valid_df.loc[idx1]
            area2_info = valid_df.loc[idx2]
            area1_center = (float(area1_info["center_x"]), float(area1_info["center_y"]))
            area2_center = (float(area2_info["center_x"]), float(area2_info["center_y"]))

            logger.info("最相似的两个20km×20km地形区域：")
            logger.info(f"区域1中心（UTM 48N）：X={area1_center[0]:.2f}m，Y={area1_center[1]:.2f}m")
            logger.info(f"区域2中心（UTM 48N）：X={area2_center[0]:.2f}m，Y={area2_center[1]:.2f}m")
            logger.info(f"综合地形相似度得分：{best_score:.4f}（满分1.0）")

            # 保存CSV
            result_df = pd.DataFrame({
                "区域名称": ["区域1", "区域2"],
                "UTM_X(米)": [area1_center[0], area2_center[0]],
                "UTM_Y(米)": [area1_center[1], area2_center[1]],
                "相似度得分": [best_score, best_score],
                "窗口起始y像素": [int(area1_info["y_start"]), int(area2_info["y_start"])],
                "窗口起始x像素": [int(area1_info["x_start"]), int(area2_info["x_start"])],
                "cluster": [int(area1_info["cluster"]), int(area2_info["cluster"])],
            })
            csv_path = os.path.join(RESULT_ROOT, "terrain_similarity_best_pair.csv")
            result_df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info(f"最优对结果已保存至：{csv_path}")

            # 可视化
            logger.info("正在生成地形相似度可视化图...")

            y1, x1 = int(area1_info["y_start"]), int(area1_info["x_start"])
            y2, x2 = int(area2_info["y_start"]), int(area2_info["x_start"])

            elev1 = elev[y1:y1 + window_size, x1:x1 + window_size]
            elev2 = elev[y2:y2 + window_size, x2:x2 + window_size]

            slope1 = pyflwdir.slope(elev1, cellsize=WRF_RES)
            slope2 = pyflwdir.slope(elev2, cellsize=WRF_RES)

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            vmin_elev = np.nanmin([np.nanmin(elev1), np.nanmin(elev2)])
            vmax_elev = np.nanmax([np.nanmax(elev1), np.nanmax(elev2)])
            im1 = axes[0, 0].imshow(elev1, cmap="terrain", vmin=vmin_elev, vmax=vmax_elev)
            axes[0, 0].set_title(f"区域1 高程（相似度：{best_score:.4f}）")
            plt.colorbar(im1, ax=axes[0, 0], label="高程（m）")

            vmin_slope = np.nanmin([np.nanmin(slope1), np.nanmin(slope2)])
            vmax_slope = np.nanmax([np.nanmax(slope1), np.nanmax(slope2)])
            im2 = axes[0, 1].imshow(slope1, cmap="viridis", vmin=vmin_slope, vmax=vmax_slope)
            axes[0, 1].set_title("区域1 坡度（度）")
            plt.colorbar(im2, ax=axes[0, 1], label="坡度（度）")

            im3 = axes[1, 0].imshow(elev2, cmap="terrain", vmin=vmin_elev, vmax=vmax_elev)
            axes[1, 0].set_title("区域2 高程")
            plt.colorbar(im3, ax=axes[1, 0], label="高程（m）")

            im4 = axes[1, 1].imshow(slope2, cmap="viridis", vmin=vmin_slope, vmax=vmax_slope)
            axes[1, 1].set_title("区域2 坡度（度）")
            plt.colorbar(im4, ax=axes[1, 1], label="坡度（度）")
            # 调整布局并保存图片
            plt.tight_layout()
            img_path = os.path.join(RESULT_ROOT, "terrain_similarity_visualization.png")
            plt.savefig(img_path, dpi=300, bbox_inches="tight")
            logger.info(f"可视化图已保存至：{img_path}")
        else:
            logger.info("未找到非重叠的相似地形区域！")
        logger.info("=" * 60)
        logger.info("地形相似度分析程序执行完成")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"\n程序执行失败：{str(e)}", exc_info=True)  # exc_info=True 输出异常堆栈信息
        raise  # 抛出异常便于调试
