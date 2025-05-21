import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import cv2

def colorize_pointcloud(depth, lidar_mask, min_distance=3, max_distance=80, radius=3):
    norm = mpl.colors.Normalize(vmin=min_distance, vmax=max_distance)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    pos = np.argwhere(lidar_mask)
    pointcloud_color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    for i in range(pos.shape[0]):
        color = tuple([int(255 * value) for value in m.to_rgba(depth[pos[i, 0], pos[i, 1]])[0:3]])
        cv2.circle(pointcloud_color, (pos[i, 1], pos[i, 0]), radius, (color[0], color[1], color[2]), -1)
    return pointcloud_color

def colorize_depth(depth, min_distance=3, max_distance=80):
    """
    深度可视化（伪彩色）。注意：depth应为归一化后的深度（0~1）。
    """
    if hasattr(depth, "cpu"):  # torch tensor
        depth = depth.cpu().detach().numpy()
    while depth.ndim > 2:
        depth = depth[0]
    depth = np.clip(depth * 150, 10, 80)

    norm = mpl.colors.Normalize(vmin=min_distance, vmax=max_distance)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    depth_color = (255 * m.to_rgba(depth)[:, :, 0:3]).astype(np.uint8)
    depth_color[depth <= 0] = [0, 0, 0]
    depth_color[np.isnan(depth)] = [0, 0, 0]
    depth_color[depth == np.inf] = [0, 0, 0]
    return depth_color

def colorize_uncertainty(uncertainty, min_value=None, max_value=None, use_percentile=True, lower_percentile=1, upper_percentile=99):
    """
    不确定性可视化（伪彩色）。支持分位数归一化，避免极端值导致全红或全蓝。
    """
    if hasattr(uncertainty, "cpu"):
        uncertainty = uncertainty.cpu().detach().numpy()
    while uncertainty.ndim > 2:
        uncertainty = uncertainty[0]
    # 分位数归一化
    if use_percentile:
        min_value = np.percentile(uncertainty, lower_percentile)
        max_value = np.percentile(uncertainty, upper_percentile)
    else:
        if min_value is None:
            min_value = np.nanmin(uncertainty)
        if max_value is None:
            max_value = np.nanmax(uncertainty)
    norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    uncertainty_color = (255 * m.to_rgba(uncertainty)[:, :, 0:3]).astype(np.uint8)
    uncertainty_color[uncertainty <= 0] = [0, 0, 0]
    uncertainty_color[np.isnan(uncertainty)] = [0, 0, 0]
    uncertainty_color[uncertainty == np.inf] = [0, 0, 0]
    return uncertainty_color