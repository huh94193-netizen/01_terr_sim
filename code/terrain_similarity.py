# 安装依赖（首次运行时执行）：
# pip install rioxarray numpy scipy scikit-learn pyflwdir pandas matplotlib glob2 xarray

import os
import glob  # 新增：用于解析通配符获取所有tif文件
import logging
from logging.handlers import RotatingFileHandler  # 日志轮转，防止文件过大
import rioxarray as rxr
import xarray as xr  # 新增：导入xarray，解决open_mfdataset所属模块问题
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import pyflwdir  # 替换原来的 from pyflwdir import Terrain
from scipy.ndimage import sobel  # 用于计算剖面曲率
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

# ----------------------
# 日志配置（新增核心部分）
# ----------------------
def setup_logger():
    """配置日志：同时输出到文件和控制台，支持日志轮转"""
    # 日志目录
    LOG_DIR = "/home/huxun/01_terr_sim/logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    # 日志文件路径
    LOG_FILE = os.path.join(LOG_DIR, "terrain_similarity.log")

    # 日志格式：时间 - 日志名 - 级别 - 文件名:行号 - 消息
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 配置文件处理器：轮转日志（10MB/文件，保留5个备份）
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,  # 保留5个备份日志
        encoding='utf-8'  # 支持中文
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    # 配置控制台处理器：输出到终端
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    # 配置根日志器
    logger = logging.getLogger("terrain_similarity")
    logger.setLevel(logging.INFO)
    # 避免重复添加处理器
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# 初始化日志器
logger = setup_logger()

# ----------------------
# 1. 核心配置参数（重点：修改结果路径和SRTM路径）
# ----------------------
# SRTM 30m数据路径（你的原始瓦片路径，支持通配符*.tif）
SRTM_PATH = "/home/huxun/00_icing_map/data/dem/SRTM_30m_China/*.tif"
# 目标区域四至（经纬度：min_lon, max_lon, min_lat, max_lat）
TARGET_EXTENT = (108.6147, 109.5914, 25.4937, 26.36719)
# WRF网格分辨率（200m，匹配模拟尺度）
WRF_RES = 200
# 地形单元大小（20km×20km）
WINDOW_SIZE_M = 20000
# 滑动窗口步长（5km，平衡覆盖度与计算量）
STEP_SIZE_M = 5000
# UTM投影（目标区域对应UTM 48N，EPSG:32648，等面积投影）
TARGET_CRS = "EPSG:32648"
# 结果保存根目录（修改为新路径）
RESULT_ROOT = "/home/huxun/01_terr_sim/result"
# 自动创建结果目录（递归创建，避免子目录不存在）
os.makedirs(RESULT_ROOT, exist_ok=True)

# ----------------------
# 新增：剖面曲率计算函数（替代Terrain.curvature(profil=True)）
# ----------------------
def calculate_profile_curvature(elev, cellsize):
    """
    计算剖面曲率（Profile Curvature）：沿最大坡度方向的曲率
    输入：高程数组、像元大小（m）
    输出：剖面曲率数组
    """
    # 计算一阶偏导数（dx：x方向，dy：y方向）
    dx = sobel(elev, axis=1) / (8 * cellsize)
    dy = sobel(elev, axis=0) / (8 * cellsize)
    # 计算二阶偏导数
    dxx = sobel(dx, axis=1) / (8 * cellsize)
    dyy = sobel(dy, axis=0) / (8 * cellsize)
    dxy = sobel(dx, axis=0) / (8 * cellsize)
    # 计算坡度的平方和
    p2 = dy ** 2
    q2 = dx ** 2
    pq = dy * dx
    # 剖面曲率公式（地形学标准公式）
    denom = (p2 + q2) ** (3/2)
    # 避免除零错误
    denom = np.where(denom == 0, np.finfo(float).eps, denom)
    curv_profile = - (p2 * dyy + 2 * pq * dxy + q2 * dxx) / denom
    return curv_profile

# ----------------------
# 2. 数据预处理函数（读取SRTM+裁剪+投影+重采样）
# ----------------------
def preprocess_srtm(srtm_path_pattern, target_extent, wrf_res, target_crs):
    """
    读取SRTM瓦片、裁剪目标区域、投影转换、重采样到WRF分辨率
    修复：支持通配符读取多个tif文件，解决文件不存在问题
    返回：高程数组、x坐标（UTM）、y坐标（UTM）
    """
    min_lon, max_lon, min_lat, max_lat = target_extent
    # 步骤1：用glob获取所有.tif文件的具体路径（解决通配符读取问题）
    logger.info("正在获取SRTM 30m数据文件列表...")
    srtm_files = glob.glob(srtm_path_pattern)
    if not srtm_files:
        raise FileNotFoundError(f"在路径 {srtm_path_pattern} 下未找到任何.tif文件！请检查路径是否正确。")
    logger.info(f"找到 {len(srtm_files)} 个SRTM瓦片文件，开始读取合并...")

    # 步骤2：批量打开所有SRTM文件并合并（用xarray的open_mfdataset实现多文件合并）
    # 修复：将rxr.open_mfdataset改为xr.open_mfdataset（open_mfdataset是xarray的函数）
    ds = xr.open_mfdataset(
        srtm_files,
        chunks={"x": 1000, "y": 1000},
        parallel=True,  # 并行读取加速
        engine="rasterio"
    )
    # 裁剪目标区域（y轴：纬度从高到低）
    ds_clip = ds.sel(
        x=slice(min_lon, max_lon),
        y=slice(max_lat, min_lat)
    )
    # 投影转换为UTM（解决经纬度距离变形问题）
    logger.info("正在将数据投影到UTM 48N（EPSG:32648）...")
    ds_proj = ds_clip.rio.reproject(target_crs)
    # 重采样到200m分辨率（均值聚合，保留地形统计特征，适配WRF尺度）
    logger.info("正在将数据重采样到200m分辨率...")
    ds_resampled = ds_proj.rio.reproject_resample(
        dst_crs=target_crs,
        resolution=wrf_res,
        resampling=rxr.enums.Resampling.average
    )
    # 处理SRTM无数据值（-32768）：3×3窗口均值填充
    logger.info("正在处理数据空洞和异常值（SRTM无数据值-32768）...")
    ds_resampled = ds_resampled.where(
        ds_resampled != -32768,
        ds_resampled.rolling(x=3, y=3, center=True).mean()
    )
    # 移除多余的维度（若有）
    ds_resampled = ds_resampled.squeeze()
    # 转为numpy数组（仅保留高程数据）
    elev = ds_resampled.values
    # 处理可能的三维数组（确保是二维）
    if len(elev.shape) == 3:
        elev = elev[0]
    # 获取重采样后的坐标（UTM米制）
    x_coords = ds_resampled.x.values
    y_coords = ds_resampled.y.values
    logger.info(f"数据预处理完成！高程数据形状：{elev.shape}（y×x，单位：像素）")
    return elev, x_coords, y_coords

# ----------------------
# 3. 滑动窗口生成函数（20km×20km子区域）
# ----------------------
def generate_sliding_windows(elev, x_coords, y_coords, window_size_m, step_size_m, wrf_res):
    """
    生成所有待分析的20km×20km子区域窗口
    返回：窗口信息DataFrame、窗口像素大小
    """
    # 窗口/步长对应的像素数（200m分辨率）
    window_size = int(window_size_m / wrf_res)  # 20000/200=100像素
    step_size = int(step_size_m / wrf_res)      # 5000/200=25像素
    # 生成窗口的左上角像素索引（避免越界）
    y_starts = np.arange(0, elev.shape[0] - window_size + 1, step_size)
    x_starts = np.arange(0, elev.shape[1] - window_size + 1, step_size)
    # 记录窗口信息（索引+中心坐标）
    window_info = []
    for y_start in y_starts:
        for x_start in x_starts:
            # 计算窗口中心的UTM坐标（米）
            center_y = y_coords[y_start + window_size // 2]
            center_x = x_coords[x_start + window_size // 2]
            window_info.append((y_start, x_start, center_x, center_y))
    # 转为DataFrame便于后续处理
    window_df = pd.DataFrame(
        window_info,
        columns=["y_start", "x_start", "center_x", "center_y"]
    )
    logger.info(f"滑动窗口生成完成！共生成{len(window_df)}个20km×20km子区域")
    return window_df, window_size

# ----------------------
# 4. 地形特征提取函数（12维WRF敏感特征）
# ----------------------
def extract_terrain_features(args):
    """
    对单个窗口提取12维地形特征（适配WRF模拟的敏感因子）
    修复：将全局变量改为参数传入（解决多进程共享变量问题）
    输入：(窗口索引, window_df, elev, window_size, WRF_RES)
    输出：(窗口索引, 12维特征向量) 或 (窗口索引, None)（无效窗口）
    """
    window_idx, window_df, elev, window_size, WRF_RES = args
    try:
        # 获取窗口的像素范围
        y_start = window_df.loc[window_idx, "y_start"]
        x_start = window_df.loc[window_idx, "x_start"]
        window_elev = elev[y_start:y_start+window_size, x_start:x_start+window_size]
        # 过滤空值占比>10%的窗口（无效数据）
        if np.isnan(window_elev).sum() > window_elev.size * 0.1:
            return (window_idx, None)
        # ----------------------
        # 步骤1：高程距平化（剥离绝对海拔，聚焦地形纹理）
        # ----------------------
        elev_mean = np.nanmean(window_elev)  # 用nanmean处理空值
        elev_anomaly = window_elev - elev_mean  # 距平=原始高程-窗口均值
        # ----------------------
        # 步骤2：提取12维核心特征
        # ----------------------
        # ① 高程距平统计特征（5维）：均值、标准差、极值差、偏度、峰度
        anom_mean = np.nanmean(elev_anomaly)
        anom_std = np.nanstd(elev_anomaly)
        anom_range = np.nanmax(elev_anomaly) - np.nanmin(elev_anomaly)
        anom_skew = skew(elev_anomaly.flatten(), nan_policy='omit')
        anom_kurt = kurtosis(elev_anomaly.flatten(), nan_policy='omit')
        # ② 地形纹理特征（4维）：坡度均值、坡度标准差（粗糙度）、坡向均值、坡向标准差
        # 替换：用pyflwdir的slope/aspect函数替代Terrain
        slope = pyflwdir.slope(window_elev, cellsize=WRF_RES)
        aspect = pyflwdir.aspect(window_elev, cellsize=WRF_RES)
        # 处理空值（用nanmean/nanstd）
        slope_mean = np.nanmean(slope)
        slope_std = np.nanstd(slope)
        aspect_mean = np.nanmean(aspect)
        aspect_std = np.nanstd(aspect)
        # ③ 空间结构特征（3维）：高程自相关、坡度自相关、剖面曲率均值
        # 高程自相关（处理空值）
        elev_flat1 = elev_anomaly[:, :-1].flatten()
        elev_flat2 = elev_anomaly[:, 1:].flatten()
        # 过滤空值后计算相关系数
        elev_valid = ~(np.isnan(elev_flat1) | np.isnan(elev_flat2))
        elev_corr = np.corrcoef(elev_flat1[elev_valid], elev_flat2[elev_valid])[0, 1] if np.sum(elev_valid) > 0 else 0.0
        # 坡度自相关（处理空值）
        slope_flat1 = slope[:, :-1].flatten()
        slope_flat2 = slope[:, 1:].flatten()
        slope_valid = ~(np.isnan(slope_flat1) | np.isnan(slope_flat2))
        slope_corr = np.corrcoef(slope_flat1[slope_valid], slope_flat2[slope_valid])[0, 1] if np.sum(slope_valid) > 0 else 0.0
        # 剖面曲率均值（替换：用自定义函数替代Terrain.curvature）
        curv_profile = calculate_profile_curvature(window_elev, WRF_RES)
        curv_mean = np.nanmean(curv_profile)
        # 合并所有特征
        features = np.array([
            anom_mean, anom_std, anom_range, anom_skew, anom_kurt,
            slope_mean, slope_std, aspect_mean, aspect_std,
            elev_corr, slope_corr, curv_mean
        ])
        return (window_idx, features)
    except Exception as e:
        logger.error(f"窗口{window_idx}处理失败：{str(e)}")
        return (window_idx, None)

# ----------------------
# 5. 非重叠判断函数（避免同一地形重复匹配）
# ----------------------
def is_overlap(idx1, idx2, valid_df, window_size):
    """
    判断两个窗口是否重叠超过50%
    返回：True（重叠）/False（不重叠）
    """
    # 获取窗口的像素范围
    y1, x1 = valid_df.loc[idx1, ["y_start", "x_start"]]
    y2, x2 = valid_df.loc[idx2, ["y_start", "x_start"]]
    y1_end, x1_end = y1 + window_size, x1 + window_size
    y2_end, x2_end = y2 + window_size, x2 + window_size
    # 计算重叠区域的坐标
    overlap_y_start = max(y1, y2)
    overlap_y_end = min(y1_end, y2_end)
    overlap_x_start = max(x1, x2)
    overlap_x_end = min(x1_end, x2_end)
    # 无重叠
    if overlap_y_end <= overlap_y_start or overlap_x_end <= overlap_x_start:
        return False
    # 计算重叠率
    overlap_area = (overlap_y_end - overlap_y_start) * (overlap_x_end - overlap_x_start)
    window_area = window_size * window_size
    return overlap_area / window_area > 0.5  # 重叠率>50%视为重叠

# ----------------------
# 主程序执行流程
# ----------------------
if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("开始执行地形相似度分析程序")
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

        # 步骤3：多进程提取地形特征（修复：将参数传入函数，解决多进程全局变量问题）
        logger.info("正在提取12维地形特征（多进程加速）...")
        cpu_cores = cpu_count() - 2  # 保留2核避免系统卡顿
        if cpu_cores < 1:
            cpu_cores = 1  # 确保至少1核
        logger.info(f"使用 {cpu_cores} 个CPU核心进行多进程计算")
        # 构造参数列表（解决多进程中全局变量无法共享的问题）
        args_list = [(i, window_df, elev, window_size, WRF_RES) for i in range(len(window_df))]
        with Pool(cpu_cores) as pool:
            feature_results = pool.map(extract_terrain_features, args_list)

        # 整理有效特征（过滤空值窗口）
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

        # 步骤4：特征标准化+PCA降维
        logger.info("正在标准化特征并执行PCA降维（保留95%方差）...")
        # 特征标准化（均值0，方差1）
        scaler = StandardScaler()
        feature_scaled = scaler.fit_transform(feature_matrix)
        # PCA降维（保留95%方差，精简维度）
        pca = PCA(n_components=0.95, random_state=42)
        feature_pca = pca.fit_transform(feature_scaled)
        logger.info(f"PCA降维完成！原始12维特征→降维后{feature_pca.shape[1]}维特征")

        # 步骤5：KMeans粗筛+余弦相似度计算（找最相似区域）
        logger.info("正在计算地形相似度（KMeans粗筛+余弦相似度精算）...")
        # KMeans粗筛（减少计算量，聚类数适配有效子区域数）
        n_clusters = min(10, len(valid_df) // 2)  # 聚类数不超过10，且至少2个样本/类
        if n_clusters < 2:
            n_clusters = 2  # 确保聚类数至少为2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(feature_pca)
        valid_df["cluster"] = clusters

        # 遍历聚类找最相似的非重叠区域
        best_score = 0.0  # 综合相似度得分（满分1.0）
        best_pair = None  # 最相似的两个区域索引

        for cluster in np.unique(clusters):
            # 获取当前聚类的所有子区域索引
            cluster_indices = valid_df[valid_df["cluster"] == cluster].index.tolist()
            if len(cluster_indices) < 2:
                continue  # 聚类内样本数不足，跳过
            # 提取当前聚类的PCA特征
            cluster_pca = feature_pca[cluster_indices]
            # 计算余弦相似度（主指标：地形模式相似性）
            cos_sim = cosine_similarity(cluster_pca)
            # 计算欧氏距离（辅指标：数值相似性）并归一化
            euclid_dist = np.linalg.norm(cluster_pca[:, None] - cluster_pca, axis=2)
            euclid_dist_norm = (euclid_dist - euclid_dist.min()) / (euclid_dist.max() - euclid_dist.min() + np.finfo(float).eps)
            # 综合相似度得分（余弦占80%，欧氏占20%）
            combined_score = cos_sim * 0.8 + (1 - euclid_dist_norm) * 0.2

            # 遍历所有子区域对，找最优解
            for i in range(len(cluster_indices)):
                for j in range(i + 1, len(cluster_indices)):
                    idx1 = cluster_indices[i]
                    idx2 = cluster_indices[j]
                    # 跳过重叠区域
                    if is_overlap(idx1, idx2, valid_df, window_size):
                        continue
                    # 更新最优对
                    if combined_score[i, j] > best_score:
                        best_score = combined_score[i, j]
                        best_pair = (idx1, idx2)

        # 步骤6：结果输出+保存+可视化
        logger.info("\n" + "=" * 60)
        if best_pair is not None:
            idx1, idx2 = best_pair
            # 获取两个区域的中心坐标（UTM 48N，米制）
            area1_info = valid_df.loc[idx1]
            area2_info = valid_df.loc[idx2]
            area1_center = (area1_info["center_x"], area1_info["center_y"])
            area2_center = (area2_info["center_x"], area2_info["center_y"])

            # 输出结果到日志/终端
            logger.info(f"最相似的两个20km×20km地形区域：")
            logger.info(f"区域1中心（UTM 48N）：X={area1_center[0]:.2f}m，Y={area1_center[1]:.2f}m")
            logger.info(f"区域2中心（UTM 48N）：X={area2_center[0]:.2f}m，Y={area2_center[1]:.2f}m")
            logger.info(f"综合地形相似度得分：{best_score:.4f}（满分1.0）")

            # 保存最优对结果到CSV文件
            result_data = {
                "区域名称": ["区域1", "区域2"],
                "UTM_X(米)": [area1_center[0], area2_center[0]],
                "UTM_Y(米)": [area1_center[1], area2_center[1]],
                "相似度得分": [best_score, best_score],
                "窗口起始y像素": [area1_info["y_start"], area2_info["y_start"]],
                "窗口起始x像素": [area1_info["x_start"], area2_info["x_start"]]
            }
            result_df = pd.DataFrame(result_data)
            csv_path = os.path.join(RESULT_ROOT, "terrain_similarity_best_pair.csv")
            result_df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info(f"\n最优对结果已保存至：{csv_path}")

            # 可视化两个区域的高程和坡度（验证相似性）
            logger.info("正在生成地形相似度可视化图...")
            # 提取区域1的高程和坡度
            y1, x1 = area1_info["y_start"], area1_info["x_start"]
            elev1 = elev[y1:y1+window_size, x1:x1+window_size]
            slope1 = pyflwdir.slope(elev1, cellsize=WRF_RES)  # 替换：用pyflwdir计算坡度
            # 提取区域2的高程和坡度
            y2, x2 = area2_info["y_start"], area2_info["x_start"]
            elev2 = elev[y2:y2+window_size, x2:x2+window_size]
            slope2 = pyflwdir.slope(elev2, cellsize=WRF_RES)  # 替换：用pyflwdir计算坡度

            # 绘制可视化图
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            # 区域1高程
            vmin_elev = np.nanmin([elev1, elev2])
            vmax_elev = np.nanmax([elev1, elev2])
            im1 = axes[0, 0].imshow(elev1, cmap="terrain", vmin=vmin_elev, vmax=vmax_elev)
            axes[0, 0].set_title(f"区域1 高程（相似度：{best_score:.4f}）")
            plt.colorbar(im1, ax=axes[0, 0], label="高程（m）")
            # 区域1坡度
            vmin_slope = np.nanmin([slope1, slope2])
            vmax_slope = np.nanmax([slope1, slope2])
            im2 = axes[0, 1].imshow(slope1, cmap="viridis", vmin=vmin_slope, vmax=vmax_slope)
            axes[0, 1].set_title("区域1 坡度（度）")
            plt.colorbar(im2, ax=axes[0, 1], label="坡度（度）")
            # 区域2高程
            im3 = axes[1, 0].imshow(elev2, cmap="terrain", vmin=vmin_elev, vmax=vmax_elev)
            axes[1, 0].set_title("区域2 高程")
            plt.colorbar(im3, ax=axes[1, 0], label="高程（m）")
            # 区域2坡度
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