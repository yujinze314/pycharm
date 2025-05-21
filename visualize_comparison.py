import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path

# 推荐：如用Windows系统，可用微软雅黑显示中文
plt.rcParams["font.family"] = ["Microsoft YaHei"]

def readlines(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()

# 路径配置
base_dir = Path('E:/gated2depth/Gated2Depth-master/data/real/')
depth_path = Path("E:/gated2depth/Gated2Depth-master/img/test")
uncertainty_path = Path("E:/gated2depth/Gated2Depth-master/img/uncertainty")
test_filenames_path = Path('E:/gated2depth/Gated2Depth-master/splits/real_test_night.txt')

test_filenames = readlines(str(test_filenames_path))
print(f"加载了 {len(test_filenames)} 个测试文件")
if not test_filenames:
    raise ValueError("测试文件列表为空！")

for i in range(len(test_filenames)):
    try:
        id = test_filenames[i].strip()
        gated_imgs = []
        missing = False
        # 读取三帧 gated 原图并做归一化
        for gate_id in range(3):
            gate_path = base_dir / f'gated{gate_id}_10bit' / f'{id}.png'
            if not gate_path.exists():
                print(f"警告: 原图不存在 - {gate_path}")
                missing = True
                break
            img = cv2.imread(str(gate_path), cv2.IMREAD_UNCHANGED)
            # 归一化到0~1，防止显示全黑
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            gated_imgs.append(np.expand_dims(img_norm, axis=2))
        if missing:
            continue
        # 拼接成三通道
        in_img_np = np.concatenate(gated_imgs, axis=2)  # (H, W, 3)
        in_img = (in_img_np * 255).astype(np.uint8)
        in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)

        # 检查深度图是否存在
        depth_file = depth_path / f'{id}.jpg'
        if not depth_file.exists():
            print(f"警告: 深度图不存在 - {depth_file}")
            continue
        depth_map = cv2.imread(str(depth_file))
        if depth_map is None:
            print(f"警告: 无法读取深度图 - {depth_file}")
            continue
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)

        # 检查不确定性图是否存在
        uncertainty_file = uncertainty_path / f'{id}.jpg'
        if not uncertainty_file.exists():
            print(f"警告: 不确定性图不存在 - {uncertainty_file}")
            continue
        uncertainty_map = cv2.imread(str(uncertainty_file), cv2.IMREAD_UNCHANGED)
        if uncertainty_map is None:
            print(f"警告: 无法读取不确定性图 - {uncertainty_file}")
            continue
        # 如果是不确定性单通道灰度，直接归一化显示
        if uncertainty_map.ndim == 3:
            # 取灰度
            uncertainty_map = cv2.cvtColor(uncertainty_map, cv2.COLOR_BGR2GRAY)
        # 归一化到0~1
        min_val = uncertainty_map.min()
        max_val = uncertainty_map.max()
        uncertainty_norm = (uncertainty_map - min_val) / (max_val - min_val + 1e-8)
        # 可选：直方图均衡增强细节
        uncertainty_eq = cv2.equalizeHist((uncertainty_norm * 255).astype(np.uint8))
        uncertainty_eq = uncertainty_eq.astype(np.float32) / 255.0

        # 显示图像
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(in_img)
        plt.title('原始图像')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(depth_map)
        plt.title('Gated2Depth 深度图')
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(uncertainty_eq, cmap='inferno')
        plt.title('不确定性图')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        plt.close()
    except Exception as e:
        print(f"处理图像 {id} 时出错: {e}")
        continue