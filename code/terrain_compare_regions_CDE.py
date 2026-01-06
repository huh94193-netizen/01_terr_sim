import os
import glob
import logging
import numpy as np
import rasterio
import rioxarray as rxr
from rioxarray.merge import merge_arrays
from rasterio.enums import Resampling
from scipy.stats import skew, kurtosis
from scipy.ndimage import sobel
from scipy.spatial.distance import euclidean, cityblock, chebyshev, cosine, correlation
from sklearn.preprocessing import StandardScaler
from itertools import combinations

# ----------------------
# 1. 配置参数
# ----------------------
# SRTM 数据路径 (请确认路径与您超算环境一致)
SRTM_PATH = "/home/huxun/00_icing_map/data/dem/SRTM_30m_China/*.tif"
# WRF网格分辨率 (200m)
WRF_RES = 200
# 目标投影 (UTM 48N - 广西/贵州周边常用投影)
TARGET_CRS = "EPSG:32648"

# 定义新的3个目标区域 (min_lon, max_lon, min_lat, max_lat)
REGIONS = {
    "区域C": (108.673, 109.077, 25.958, 26.322),
    "区域D": (109.154, 109.555, 25.957, 26.321),
    "区域E": (108.950, 109.351, 25.792, 26.156),
}

# ----------------------
# 日志配置
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("RegionCompare")

# ----------------------
# 核心算法函数 (源自 v7)
# ----------------------
def calculate_profile_curvature(elev, cellsize):
    """计算剖面曲率"""
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

def is_bounds_intersect(bounds1, bounds2):
    """判断范围是否相交"""
    minx1, miny1, maxx1, maxy1 = bounds1
    minx2, miny2, maxx2, maxy2 = bounds2
    if maxx1 < minx2 or minx1 > maxx2 or maxy1 < miny2 or miny1 > maxy2:
        return False
    return True

def get_region_data(name, extent, srtm_files):
    """读取并预处理区域地形数据"""
    min_lon, max_lon, min_lat, max_lat = extent
    buffer = 0.05 # 缓冲防止边缘裁剪问题
    target_bounds_buf = (min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer)
    
    # 筛选文件
    candidate_files = []
    for fp in srtm_files:
        try:
            with rasterio.open(fp) as src:
                if is_bounds_intersect(src.bounds, target_bounds_buf):
                    candidate_files.append(fp)
        except:
            continue
            
    if not candidate_files:
        logger.warning(f"{name} 未找到覆盖的SRTM文件")
        return None

    # 合并
    das = []
    for fp in candidate_files:
        da = rxr.open_rasterio(fp, masked=True).squeeze(drop=True)
        if da.rio.crs is None:
            da = da.rio.write_crs("EPSG:4326")
        das.append(da)
    
    try:
        da_mosaic = merge_arrays(das)
    except Exception as e:
        logger.error(f"{name} 合并失败: {e}")
        return None
    
    # 裁剪与重采样
    da_clip = da_mosaic.rio.clip_box(minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat)
    da_proj = da_clip.rio.reproject(TARGET_CRS)
    da_resampled = da_proj.rio.reproject(
        TARGET_CRS,
        resolution=WRF_RES,
        resampling=Resampling.average,
    )
    
    # 简单填补
    da_resampled = da_resampled.where(
        da_resampled != -32768,
        da_resampled.rolling(x=3, y=3, center=True, min_periods=1).mean(),
    )
    
    return da_resampled.values

def extract_features(elev, wrf_res):
    """提取12维统计特征"""
    if elev is None or np.isnan(elev).sum() > elev.size * 0.5:
        return None

    # 1. 高程距平
    elev_mean = np.nanmean(elev)
    elev_anomaly = elev - elev_mean
    anom_mean = np.nanmean(elev_anomaly)
    anom_std = np.nanstd(elev_anomaly)
    anom_range = np.nanmax(elev_anomaly) - np.nanmin(elev_anomaly)
    anom_skew = skew(elev_anomaly.flatten(), nan_policy="omit")
    anom_kurt = kurtosis(elev_anomaly.flatten(), nan_policy="omit")

    # 2. 坡度坡向
    dx = sobel(elev, axis=1) / (8 * wrf_res)
    dy = sobel(elev, axis=0) / (8 * wrf_res)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    aspect = (np.degrees(np.arctan2(-dx, dy)) + 360) % 360
    
    slope_mean = np.nanmean(slope)
    slope_std = np.nanstd(slope)
    aspect_mean = np.nanmean(aspect)
    aspect_std = np.nanstd(aspect)

    # 3. 自相关性
    elev_flat1 = elev_anomaly[:, :-1].flatten()
    elev_flat2 = elev_anomaly[:, 1:].flatten()
    mask_elev = ~(np.isnan(elev_flat1) | np.isnan(elev_flat2))
    # 防止完全平坦导致的除零警告
    if np.std(elev_flat1[mask_elev]) == 0 or np.std(elev_flat2[mask_elev]) == 0:
        elev_corr = 0
    else:
        elev_corr = np.corrcoef(elev_flat1[mask_elev], elev_flat2[mask_elev])[0, 1]

    slope_flat1 = slope[:, :-1].flatten()
    slope_flat2 = slope[:, 1:].flatten()
    mask_slope = ~(np.isnan(slope_flat1) | np.isnan(slope_flat2))
    if np.std(slope_flat1[mask_slope]) == 0 or np.std(slope_flat2[mask_slope]) == 0:
        slope_corr = 0
    else:
        slope_corr = np.corrcoef(slope_flat1[mask_slope], slope_flat2[mask_slope])[0, 1]

    # 4. 曲率
    curv_profile = calculate_profile_curvature(elev, wrf_res)
    curv_mean = np.nanmean(curv_profile)

    return np.array([
        anom_mean, anom_std, anom_range, anom_skew, anom_kurt,
        slope_mean, slope_std, aspect_mean, aspect_std,
        elev_corr, slope_corr, curv_mean
    ])

# ----------------------
# 主程序
# ----------------------
if __name__ == "__main__":
    logger.info("开始检索 SRTM 数据...")
    srtm_files = sorted(glob.glob(SRTM_PATH))
    if not srtm_files:
        logger.error(f"路径下未找到 .tif 文件: {SRTM_PATH}")
        exit()

    # 1. 提取特征
    region_features = {}
    feature_list = []
    region_names = []

    for name, extent in REGIONS.items():
        logger.info(f"正在处理: {name} {extent}")
        elev_data = get_region_data(name, extent, srtm_files)
        feat = extract_features(elev_data, WRF_RES)
        
        if feat is not None:
            region_features[name] = feat
            feature_list.append(feat)
            region_names.append(name)
            logger.info(f"  -> {name} 特征提取成功")
        else:
            logger.error(f"  -> {name} 特征提取失败 (可能无数据)")

    if len(feature_list) < 2:
        logger.error("有效区域少于2个，无法计算相似度。")
        exit()

    # 2. 标准化特征 (Z-Score)
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(np.array(feature_list))
    scaled_dict = {name: feats_scaled[i] for i, name in enumerate(region_names)}

    # 3. 计算5种距离
    metrics = {
        "欧氏距离 (Euclidean)": euclidean,
        "曼哈顿距离 (Manhattan)": cityblock,
        "切比雪夫距离 (Chebyshev)": chebyshev,
        "余弦距离 (Cosine)": cosine,
        "相关系数距离 (Correlation)": correlation
    }

    logger.info("=" * 60)
    logger.info(f"地形相似度计算结果 (针对区域: {', '.join(region_names)})")
    logger.info("提示: 数值越小 = 越相似")
    logger.info("=" * 60)

    pairs = list(combinations(region_names, 2))

    for metric_name, func in metrics.items():
        best_pair = None
        min_dist = float('inf')
        
        print(f"\n--- {metric_name} ---")
        for r1, r2 in pairs:
            vec1 = scaled_dict[r1]
            vec2 = scaled_dict[r2]
            dist = func(vec1, vec2)
            
            print(f"  {r1} vs {r2}: {dist:.6f}")
            
            if dist < min_dist:
                min_dist = dist
                best_pair = (r1, r2)
        
        print(f"  >> 最相似组合: {best_pair[0]} <--> {best_pair[1]} (距离: {min_dist:.6f})")