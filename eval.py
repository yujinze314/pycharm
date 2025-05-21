import os
import numpy as np
import torch
from datasets import read_gt_image
import unet
import cv2
import math
import visualize2D
import matplotlib.pyplot as plt

def readlines(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def read_img(img_path, num_bits=10, crop_height=512, crop_width=1024, dataset='g2d', overexpose=False):
    gated_imgs = []
    normalizer = 2 ** num_bits - 1.
    for gate_id in range(3):
        path = img_path.format(gate_id)
        assert os.path.exists(path), "No such file : %s" % path
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img[((img.shape[0] - crop_height) // 2):((img.shape[0] + crop_height) // 2),
              ((img.shape[1] - crop_width) // 2):((img.shape[1] + crop_width) // 2)]
        if overexpose:
            img = np.clip(img * 2, 0, normalizer)
        img = img.copy()
        img[img > 2 ** 10 - 1] = normalizer
        img = np.float32(img / normalizer)
        gated_imgs.append(np.expand_dims(img, axis=2))
    img = np.concatenate(gated_imgs, axis=2)
    return img

def threshold(y1, y2, thr=1.25):
    max_ratio = np.maximum(y1 / y2, y2 / y1)
    return np.mean(max_ratio < thr, dtype=np.float64) * 100.

def rmse(y1, y2):
    diff = y1 - y2
    return math.sqrt(np.mean(diff * diff, dtype=np.float64))

def ard(y1, y2):
    return np.mean(np.abs(y1 - y2) / y2, dtype=np.float64)

def mae(y1, y2):
    return np.mean(np.abs(y1 - y2), dtype=np.float64)

def result(output, input, lidar_mask):
    output = output[lidar_mask].cpu().detach().numpy()
    input = input[lidar_mask].cpu().detach().numpy()
    output = np.clip(output * 150, 3, 80)
    input = np.clip(input * 150, 3, 80)
    Rmse = rmse(output, input)
    Ard = ard(output, input)
    Mae = mae(output, input)
    Threshold1 = threshold(output, input, thr=1.25)
    Threshold2 = threshold(output, input, thr=1.25 ** 2)
    Threshold3 = threshold(output, input, thr=1.25 ** 3)
    return Rmse, Ard, Mae, Threshold1, Threshold2, Threshold3

def eval(models, test_filenames, device):
    models["Encoder"].train()
    models["Decoder"].train()
    max_distance = 150
    min_distance = 3
    base_dir = 'E:/gated2depth/Gated2Depth-master/data/real/'
    Results = []
    num_samples = 10

    for i in range(len(test_filenames)):
        print("{}/{}".format(i, len(test_filenames)))
        id = test_filenames[i].strip()
        input_np, lidar_mask_np = read_gt_image(base_dir, id, 'real')
        gate_dir = os.path.join(base_dir, 'gated{}_10bit', '{}.png'.format(id))

        # 正常曝光输入
        in_img_np = read_img(gate_dir)
        # 曝光过度输入
        in_img_np_over = read_img(gate_dir, overexpose=True)

        in_img = torch.tensor(in_img_np, dtype=torch.float32).unsqueeze(0).to(device=device)
        in_img = in_img.permute(0, 3, 1, 2)
        in_img_over = torch.tensor(in_img_np_over, dtype=torch.float32).unsqueeze(0).to(device=device)
        in_img_over = in_img_over.permute(0, 3, 1, 2)

        outputs = []
        for _ in range(num_samples):
            with torch.no_grad():
                output_dict = models["Decoder"](models["Encoder"](in_img))
                output = output_dict["output", 0]
                outputs.append(output.cpu())
        outputs = torch.stack(outputs, dim=0)
        mean_output = outputs.mean(dim=0)[0]
        std_output = outputs.std(dim=0)[0]

        outputs_over = []
        for _ in range(num_samples):
            with torch.no_grad():
                output_dict = models["Decoder"](models["Encoder"](in_img_over))
                output = output_dict["output", 0]
                outputs_over.append(output.cpu())
        outputs_over = torch.stack(outputs_over, dim=0)
        mean_output_over = outputs_over.mean(dim=0)[0]
        std_output_over = outputs_over.std(dim=0)[0]

        print("std_output min/max:", std_output.min().item(), std_output.max().item())
        print("std_output_over min/max:", std_output_over.min().item(), std_output_over.max().item())

        # 可视化前画直方图，辅助调试
        plt.figure()
        plt.hist(std_output.cpu().numpy().flatten(), bins=100)
        plt.title(f"Uncertainty histogram: {id}")
        plt.savefig(f"E:/gated2depth/Gated2Depth-master/img/uncertainty/{id}_hist.png")
        plt.close()

        # --- 保存可视化 ---
        depth_map_normal = visualize2D.colorize_depth(mean_output, min_distance=3, max_distance=80)
        uncertainty_map_normal = visualize2D.colorize_uncertainty(std_output)  # 默认99/1分位归一化
        depth_map_over = visualize2D.colorize_depth(mean_output_over, min_distance=3, max_distance=80)
        uncertainty_map_over = visualize2D.colorize_uncertainty(std_output_over)

        plt.figure(figsize=(20, 4))
        plt.subplot(151)
        plt.imshow((in_img_np * 255).astype(np.uint8))
        plt.title('Original')
        plt.axis('off')
        plt.subplot(152)
        plt.imshow(depth_map_normal)
        plt.title('Depth Normal')
        plt.axis('off')
        plt.subplot(153)
        plt.imshow(uncertainty_map_normal)
        plt.title('Uncertainty Normal')
        plt.axis('off')
        plt.subplot(154)
        plt.imshow(depth_map_over)
        plt.title('Depth Overexposed')
        plt.axis('off')
        plt.subplot(155)
        plt.imshow(uncertainty_map_over)
        plt.title('Uncertainty Overexposed')
        plt.axis('off')
        plt.tight_layout()
        compare_path = "E:/gated2depth/Gated2Depth-master/img/compare"
        os.makedirs(compare_path, exist_ok=True)
        plt.savefig(os.path.join(compare_path, f"{id}_compare.jpg"))
        plt.close()

        # 保存单独的深度/不确定性图
        crop_size_h = 46
        crop_size_w = 22
        depth_map_color = depth_map_normal[crop_size_h:(depth_map_normal.shape[0] - crop_size_h),
                          crop_size_w: (depth_map_normal.shape[1] - crop_size_w)]
        depth_path = "E:/gated2depth/Gated2Depth-master/img/test"
        os.makedirs(depth_path, exist_ok=True)
        cv2.imwrite(os.path.join(depth_path, '{}.jpg'.format(id)), depth_map_color.astype(np.uint8))

        uncertainty_map_color = uncertainty_map_normal
        uncertainty_path = "E:/gated2depth/Gated2Depth-master/img/uncertainty"
        os.makedirs(uncertainty_path, exist_ok=True)
        cv2.imwrite(os.path.join(uncertainty_path, '{}.jpg'.format(id)), uncertainty_map_color.astype(np.uint8))

        input_tensor = torch.tensor(input_np, dtype=torch.float32)
        lidar_mask_tensor = torch.tensor(lidar_mask_np, dtype=torch.bool)
        Result = result(mean_output, input_tensor, lidar_mask_tensor)
        print("Result:", Result)
        Results.append(Result)

    res = np.array(Results).mean(0)
    print("rmse={:.2f}  ard={:.2f}  mae={:.2f} delta1={:.2f}  delta2={:.2f}  delta3={:.2f}".format(
        res[0], res[1], res[2], res[3], res[4], res[5]))

if __name__ == '__main__':
    data_fpath = 'E:/gated2depth/Gated2Depth-master/splits/'
    fpath = os.path.join(data_fpath, "real_{}_night.txt")
    test_filenames = readlines(fpath.format("test"))
    test_filenames = test_filenames[:10]  # 只处理前10张图片
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    models["Encoder"] = unet.Encoder()
    models["Decoder"] = unet.Decoder()
    Encoderpath = "E:/gated2depth/Gated2Depth-master/models/Encoder.pth"
    Decoderpath = "E:/gated2depth/Gated2Depth-master/models/Decoder.pth"
    models["Encoder"].load_state_dict(torch.load(Encoderpath, map_location=lambda storage, loc: storage))
    models["Decoder"].load_state_dict(torch.load(Decoderpath, map_location=lambda storage, loc: storage))
    models["Encoder"].to(device=device)
    models["Decoder"].to(device=device)
    eval(models, test_filenames, device)
