# pip install rioxarray rasterio numpy scipy scikit-learn pandas matplotlib glob2 xarray numba geopandas
# 中文适配版 + 省界叠加功能

import os
import glob
import logging
from logging.handlers import RotatingFileHandler

import rioxarray as rxr
import xarray as xr
import numpy as np
import pandas as pd
import rasterio 
from rasterio.enums import Resampling

from rioxarray.merge import merge_arrays
from scipy.stats import skew, kurtosis
from scipy.ndimage import sobel

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 新增库：用于处理矢量地图
try:
    import geopandas as gpd
    from shapely.geometry import box
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# ----------------------
# 日志配置
# ----------------------
def setup_logger():
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
# 字体配置
# ----------------------
def configure_chinese_font():
    """配置 Matplotlib 使用系统安装的中文字体"""
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    logger.info("字体配置已应用，使用字体：WenQuanYi Zen Hei")

configure_chinese_font()

# ----------------------
# 1. 配置参数
# ----------------------
SRTM_PATH = "/home/huxun/00_icing_map/data/dem/SRTM_30m_China/*.tif"
TARGET_EXTENT = (108.6147, 109.5914, 25.4937, 26.36719)
SRTM_BUFFER_DEG = 0.1
WRF_RES = 200
WINDOW_SIZE_M = 20000
STEP_SIZE_M = 5000
TARGET_CRS = "EPSG:32648"
RESULT_ROOT = "/home/huxun/01_terr_sim/result"
os.makedirs(RESULT_ROOT, exist_ok=True)

# ----------------------
# 辅助函数
# ----------------------
def is_bounds_intersect(bounds1, bounds2):
    minx1, miny1, maxx1, maxy1 = bounds1
    minx2, miny2, maxx2, maxy2 = bounds2
    if maxx1 < minx2 or minx1 > maxx2 or maxy1 < miny2 or miny1 > maxy2:
        return False
    return True

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
# 新增：绘制省界函数
# ----------------------
def add_province_boundaries(ax, target_crs, extent):
    """
    下载并绘制省界和省名
    extent: [minx, maxx, miny, maxy] in target_crs
    """
    if not HAS_GEOPANDAS:
        logger.warning("未安装 geopandas，无法绘制省界。请运行 pip install geopandas")
        return

    try:
        logger.info("正在加载省级行政区划数据...")
        # 使用阿里云 DataV 的 GeoJSON 接口（包含全中国省级边界）
        china_url = "https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json"
        gdf = gpd.read_file(china_url)
        
        # 转换坐标系到 UTM
        gdf_proj = gdf.to_crs(target_crs)
        
        # 创建视图范围的几何对象用于筛选
        minx, maxx, miny, maxy = extent
        view_box = box(minx, miny, maxx, maxy)
        
        # 筛选与当前视图相交的省份（提高绘图速度）
        gdf_clip = gdf_proj[gdf_proj.intersects(view_box)]
        
        if gdf_clip.empty:
            logger.info("当前视图范围内没有省界线。")
            return

        # 绘制边界
        gdf_clip.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5, linestyle='--', alpha=0.7)
        
        # 标注省名
        for idx, row in gdf_clip.iterrows():
            # 获取该省在当前坐标系下的中心点
            centroid = row.geometry.centroid
            # 只有当中心点在视图内，或者虽然中心点不在但省份覆盖了视图时，尝试标注
            # 这里简单处理：如果中心点在图内直接标，如果在图外但有交集，找一个在图内的点标
            
            label_x, label_y = centroid.x, centroid.y
            
            # 简单的边界检查，防止字标出图外
            if minx < label_x < maxx and miny < label_y < maxy:
                ax.text(label_x, label_y, row['name'], 
                        fontsize=12, fontweight='bold', color='black', 
                        ha='center', va='center', 
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
            else:
                # 如果几何中心不在图内，尝试计算几何体与视图矩形的交集的中心
                intersection = row.geometry.intersection(view_box)
                if not intersection.is_empty:
                    vis_center = intersection.centroid
                    ax.text(vis_center.x, vis_center.y, row['name'], 
                            fontsize=12, fontweight='bold', color='black', 
                            ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

        logger.info("省界及地名标注完成。")

    except Exception as e:
        logger.warning(f"绘制省界失败 (可能是网络原因无法下载 GeoJSON): {e}")

# ----------------------
# 2. 数据预处理
# ----------------------
def preprocess_srtm(srtm_path_pattern, target_extent, buffer_deg, wrf_res, target_crs):
    min_lon, max_lon, min_lat, max_lat = target_extent
    min_lon_buf = min_lon - buffer_deg
    max_lon_buf = max_lon + buffer_deg
    min_lat_buf = min_lat - buffer_deg
    max_lat_buf = max_lat + buffer_deg
    target_bounds_buf = (min_lon_buf, min_lat_buf, max_lon_buf, max_lat_buf)
    logger.info(f"目标范围 (Buffered): {target_bounds_buf}")

    logger.info("正在获取 SRTM 文件列表...")
    srtm_files = sorted(glob.glob(srtm_path_pattern))
    if not srtm_files:
        raise FileNotFoundError(f"No .tif files found in {srtm_path_pattern}")
    
    candidate_files = []
    for fp in srtm_files:
        try:
            with rasterio.open(fp) as src:
                tile_bounds = src.bounds
                tile_crs = src.crs
            if tile_crs is None or tile_crs != rasterio.crs.CRS.from_epsg(4326):
                continue
            if is_bounds_intersect(tile_bounds, target_bounds_buf):
                candidate_files.append(fp)
        except Exception as e:
            logger.warning(f"元数据读取失败 {fp}, 跳过")
            continue

    if not candidate_files:
        raise FileNotFoundError("未找到相交的 SRTM 瓦片!")
    logger.info(f"选中 {len(candidate_files)} 个瓦片. 开始加载...")

    das = []
    for fp in candidate_files:
        da = rxr.open_rasterio(fp, masked=True).squeeze(drop=True)
        if da.rio.crs is None:
            da = da.rio.write_crs("EPSG:4326")
        das.append(da)

    logger.info("正在合并瓦片...")
    da_mosaic = merge_arrays(das)
    da_clip = da_mosaic.rio.clip_box(minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat)
    
    logger.info(f"投影至 {target_crs} 并重采样...")
    da_proj = da_clip.rio.reproject(target_crs)
    da_resampled = da_proj.rio.reproject(
        target_crs,
        resolution=wrf_res,
        resampling=Resampling.average,
    )

    da_resampled = da_resampled.where(
        da_resampled != -32768,
        da_resampled.rolling(x=3, y=3, center=True, min_periods=1).mean(),
    )

    elev = da_resampled.values
    x_coords = da_resampled.x.values
    y_coords = da_resampled.y.values

    logger.info(f"预处理完成. 形状: {elev.shape}")
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
    logger.info(f"生成了 {len(window_df)} 个滑动窗口.")
    return window_df, window_size


# ----------------------
# 4. 特征提取
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

        dx = sobel(window_elev, axis=1) / (8 * wrf_res)
        dy = sobel(window_elev, axis=0) / (8 * wrf_res)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope = np.degrees(slope_rad)
        aspect = (np.degrees(np.arctan2(-dx, dy)) + 360) % 360

        slope_mean = np.nanmean(slope)
        slope_std = np.nanstd(slope)
        aspect_mean = np.nanmean(aspect)
        aspect_std = np.nanstd(aspect)

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
            [anom_mean, anom_std, anom_range, anom_skew, anom_kurt, slope_mean, slope_std, 
             aspect_mean, aspect_std, elev_corr, slope_corr, curv_mean],
            dtype=float,
        )
        return (window_idx, features)

    except Exception as e:
        logger.error(f"Window {window_idx} failed: {type(e).__name__}: {str(e)}")
        return (window_idx, None)


# ----------------------
# 5. 重叠判断
# ----------------------
def is_overlap(idx1, idx2, valid_df, window_size):
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
# 主程序
# ----------------------
if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("开始地形相似度分析")
        logger.info("=" * 60)

        # 步骤1
        elev, x_coords, y_coords = preprocess_srtm(
            SRTM_PATH, TARGET_EXTENT, SRTM_BUFFER_DEG, WRF_RES, TARGET_CRS
        )

        # 步骤2
        window_df, window_size = generate_sliding_windows(
            elev, x_coords, y_coords, WINDOW_SIZE_M, STEP_SIZE_M, WRF_RES
        )

        # 步骤3
        logger.info("正在提取特征...")
        cpu_cores = max(1, cpu_count() - 2)
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
            raise ValueError("没有提取到有效特征!")

        feature_matrix = np.array(feature_matrix)
        valid_df = window_df.loc[valid_indices].reset_index(drop=True)
        logger.info(f"特征提取完成. 有效样本数: {len(valid_df)}")

        # 步骤4
        logger.info("PCA 降维...")
        scaler = StandardScaler()
        feature_scaled = scaler.fit_transform(feature_matrix)
        pca = PCA(n_components=0.95, random_state=42)
        feature_pca = pca.fit_transform(feature_scaled)

        # 步骤5
        logger.info("计算相似度...")
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

        # 步骤6：输出与可视化
        logger.info("\n" + "=" * 60)
        if best_pair is not None:
            idx1, idx2 = best_pair
            area1_info = valid_df.loc[idx1]
            area2_info = valid_df.loc[idx2]
            
            logger.info("找到最佳匹配区域!")
            logger.info(f"相似度得分: {best_score:.4f}")

            # 1. 保存CSV
            result_df = pd.DataFrame({
                "区域名称": ["区域 1", "区域 2"],
                "UTM_X (米)": [area1_info["center_x"], area2_info["center_x"]],
                "UTM_Y (米)": [area1_info["center_y"], area2_info["center_y"]],
                "相似度得分": [best_score, best_score]
            })
            csv_path = os.path.join(RESULT_ROOT, "terrain_similarity_best_pair.csv")
            result_df.to_csv(csv_path, index=False)

            # 2. 局部对比图
            logger.info("生成局部对比图...")
            y1, x1 = int(area1_info["y_start"]), int(area1_info["x_start"])
            y2, x2 = int(area2_info["y_start"]), int(area2_info["x_start"])
            
            elev1 = elev[y1:y1 + window_size, x1:x1 + window_size]
            elev2 = elev[y2:y2 + window_size, x2:x2 + window_size]
            
            def get_slope(arr):
                dx = sobel(arr, axis=1) / (8 * WRF_RES)
                dy = sobel(arr, axis=0) / (8 * WRF_RES)
                return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

            slope1 = get_slope(elev1)
            slope2 = get_slope(elev2)

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            vmin, vmax = np.nanmin([elev1, elev2]), np.nanmax([elev1, elev2])
            
            # Area 1 Elevation
            im1 = axes[0, 0].imshow(elev1, cmap="terrain", vmin=vmin, vmax=vmax)
            axes[0, 0].set_title(f"区域 1 高程 (相似度: {best_score:.3f})")
            plt.colorbar(im1, ax=axes[0, 0], label="高程 (米)")
            
            # Area 1 Slope
            im2 = axes[0, 1].imshow(slope1, cmap="viridis")
            axes[0, 1].set_title("区域 1 坡度")
            plt.colorbar(im2, ax=axes[0, 1], label="坡度 (度)")
            
            # Area 2 Elevation
            im3 = axes[1, 0].imshow(elev2, cmap="terrain", vmin=vmin, vmax=vmax)
            axes[1, 0].set_title("区域 2 高程")
            plt.colorbar(im3, ax=axes[1, 0], label="高程 (米)")
            
            # Area 2 Slope
            im4 = axes[1, 1].imshow(slope2, cmap="viridis")
            axes[1, 1].set_title("区域 2 坡度")
            plt.colorbar(im4, ax=axes[1, 1], label="坡度 (度)")
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_ROOT, "detail_comparison.png"), dpi=300)

            # 3. 全局位置概览图
            logger.info("生成全局概览图...")
            fig_ov, ax_ov = plt.subplots(figsize=(10, 10))
            
            extent = [x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]]
            
            im_ov = ax_ov.imshow(elev, cmap="terrain", extent=extent)
            plt.colorbar(im_ov, ax=ax_ov, label="高程 (米)", fraction=0.046, pad=0.04)
            
            # --- 新增：绘制省界 ---
            add_province_boundaries(ax_ov, TARGET_CRS, extent)
            # --------------------

            w_real = window_size * WRF_RES
            h_real = window_size * WRF_RES
            
            # Area 1
            x1_utm = x_coords[x1]
            y1_utm_top = y_coords[y1]
            y1_utm_bottom = y1_utm_top - h_real
            
            rect1 = patches.Rectangle(
                (x1_utm, y1_utm_bottom), w_real, h_real, 
                linewidth=2.5, edgecolor='red', facecolor='none', label='区域 1'
            )
            ax_ov.add_patch(rect1)
            ax_ov.text(x1_utm, y1_utm_top, " 区域 1", color='red', fontweight='bold', va='bottom')

            # Area 2
            x2_utm = x_coords[x2]
            y2_utm_top = y_coords[y2]
            y2_utm_bottom = y2_utm_top - h_real
            
            rect2 = patches.Rectangle(
                (x2_utm, y2_utm_bottom), w_real, h_real, 
                linewidth=2.5, edgecolor='blue', facecolor='none', label='区域 2'
            )
            ax_ov.add_patch(rect2)
            ax_ov.text(x2_utm, y2_utm_top, " 区域 2", color='blue', fontweight='bold', va='bottom')
            
            ax_ov.set_title("相似地形区域位置概览")
            ax_ov.set_xlabel("UTM 东向坐标 (米)")
            ax_ov.set_ylabel("UTM 北向坐标 (米)")
            
            # 限制绘图范围在 SRTM 数据范围内，防止省界画太大导致图缩小
            ax_ov.set_xlim(x_coords[0], x_coords[-1])
            ax_ov.set_ylim(y_coords[-1], y_coords[0])
            
            ax_ov.legend(loc='upper right')
            
            overview_path = os.path.join(RESULT_ROOT, "location_overview.png")
            plt.savefig(overview_path, dpi=300, bbox_inches="tight")
            logger.info(f"概览图已保存至: {overview_path}")
            
        else:
            logger.info("未找到非重叠的相似区域!")
        
        logger.info("=" * 60)
        logger.info("分析完成")

    except Exception as e:
        logger.error(f"\n执行失败: {str(e)}", exc_info=True)
        raise