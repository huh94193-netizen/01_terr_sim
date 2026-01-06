import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy.spatial.distance import euclidean, mahalanobis, cityblock, cosine, correlation
from itertools import combinations
import glob

# 1. 区域经纬度范围
regions = {
    "区域1": (107.875, 109.125, 25.875, 27.125),
    "区域2": (108.875, 110.125, 25.875, 27.125),
    "区域3": (107.875, 109.125, 24.875, 26.125),
    "区域4": (108.875, 110.125, 24.875, 26.125),
}

# 2. SRTM数据路径和参数
srtm_path_pattern = "/home/huxun/01_terr_sim/data/srtm_30m/*.tif"

def preprocess_srtm(srtm_path_pattern, extent, target_shape=(400, 400)):
    # extent: (min_lon, max_lon, min_lat, max_lat)
    min_lon, max_lon, min_lat, max_lat = extent
    tif_files = glob.glob(srtm_path_pattern)
    for tif in tif_files:
        with rasterio.open(tif) as src:
            # 判断区域是否在tif覆盖范围内
            bounds = src.bounds
            if (min_lon >= bounds.left and max_lon <= bounds.right and
                min_lat >= bounds.bottom and max_lat <= bounds.top):
                window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
                elev = src.read(1, window=window)
                # 重采样到统一shape
                elev = elev.astype(np.float32)
                elev[elev == src.nodata] = np.nan
                elev = np.nan_to_num(elev, nan=np.nanmean(elev))
                elev_resized = np.array(
                    np.resize(elev, target_shape)
                )
                return elev_resized
    raise RuntimeError("未找到覆盖该区域的SRTM数据")

region_elevs = {}
for name, extent in regions.items():
    region_elevs[name] = preprocess_srtm(srtm_path_pattern, extent)

# 3. 特征向量化
def flatten_elev(elev):
    return elev.flatten()

region_features = {name: flatten_elev(elev) for name, elev in region_elevs.items()}

# 4. 多种距离度量
def calc_distances(vec1, vec2):
    eucl = euclidean(vec1, vec2)
    stacked = np.vstack([vec1, vec2])
    cov = np.cov(stacked, rowvar=False)
    cov_inv = np.linalg.pinv(cov)
    try:
        maha = mahalanobis(vec1, vec2, cov_inv)
    except Exception:
        maha = np.nan
    manh = cityblock(vec1, vec2)
    cosd = cosine(vec1, vec2)
    corr = correlation(vec1, vec2)
    return {
        "欧氏": eucl,
        "马氏": maha,
        "曼哈顿": manh,
        "余弦": cosd,
        "相关系数": corr
    }

# 5. 两两组合计算
results = {dist: [] for dist in ["欧氏", "马氏", "曼哈顿", "余弦", "相关系数"]}
pairs = list(combinations(region_features.keys(), 2))

for a, b in pairs:
    vec1 = region_features[a]
    vec2 = region_features[b]
    dists = calc_distances(vec1, vec2)
    for dist_name, value in dists.items():
        results[dist_name].append(((a, b), value))

# 6. 输出每种距离下最相似的区域对
for dist_name, vals in results.items():
    min_pair, min_val = min(vals, key=lambda x: x[1])
    print(f"{dist_name}距离下最相似区域对: {min_pair}, 距离值: {min_val:.4f}")