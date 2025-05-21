import numpy as np
import os
import cv2
import random
import torch

import torch.utils.data as data
from torchvision import transforms


def read_gt_image(base_dir, img_id, data_type, depth_normalizer=150.0, min_distance=3, max_distance=150.0,
                  scale_images=False,
                  scaled_img_width=None,
                  crop_size_h=104, crop_size_w=128,
                  scaled_img_height=None, raw_values_only=False):
    if data_type == 'real':
        depth_lidar1 = np.load(os.path.join(base_dir, "depth_hdl64_gated_compressed", img_id + '.npz'))['arr_0']
        depth_lidar1 = depth_lidar1[crop_size_h:(depth_lidar1.shape[0] - crop_size_h),
                       crop_size_w:(depth_lidar1.shape[1] - crop_size_w)]
        if raw_values_only:
            return depth_lidar1, None

        # gt_mask=false(512*1024)
        gt_mask = (depth_lidar1 > 0.)
        depth_lidar1 = np.float32(np.clip(depth_lidar1, min_distance, max_distance) / depth_normalizer)
        depth_lidar1 = np.expand_dims(depth_lidar1, axis=0)
        gt_mask = np.expand_dims(gt_mask, axis=0)
        # (1,512,1024)???
        # print(depth_lidar1.shape)
        depth_lidar1 = torch.from_numpy(depth_lidar1)
        # print(depth_lidar1.shape)

        return depth_lidar1, gt_mask

    img = np.load(os.path.join(base_dir, 'depth_compressed', img_id + '.npz'))['arr_0']

    if raw_values_only:
        return img, None

    img = np.clip(img, min_distance, max_distance) / max_distance

    if scale_images:
        img = cv2.resize(img, dsize=(scaled_img_width, scaled_img_height), interpolation=cv2.INTER_AREA)

    return np.expand_dims(np.expand_dims(img, axis=2), axis=0), None

def read_gated_image(base_dir, img_id, num_bits=10, data_type='real',
                     scale_images=False, scaled_img_width=None, crop_size_h=104, crop_size_w=128,
                     scaled_img_height=None):
    gated_imgs = []
    normalizer = 2 ** num_bits - 1.

    for gate_id in range(3):
        # 统一路径拼接方式
        gate_dir = os.path.join(base_dir, f'gated{gate_id}_10bit')
        path = os.path.join(gate_dir, f'{img_id}.png')

        # 检查文件是否存在，如果不存在则记录日志并跳过
        if not os.path.exists(path):
            print(f"Warning: File not found - {path}")
            return None  # 返回 None 表示该样本无效

        # 加载图像
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # 处理图像
        if data_type == 'real':
            img = img[crop_size_h:(img.shape[0] - crop_size_h),
                  crop_size_w:(img.shape[1] - crop_size_w)]
            img = img.copy()
            img[img > 2 ** 10 - 1] = normalizer

        img = np.float32(img / normalizer)
        gated_imgs.append(np.expand_dims(img, axis=2))

    img = np.concatenate(gated_imgs, axis=2)

    # 缩放图像（如果需要）
    if scale_images:
        img = cv2.resize(img, dsize=(scaled_img_width, scaled_img_height), interpolation=cv2.INTER_AREA)
    return img

# def read_gated_image(base_dir, img_id, num_bits=10, data_type='real',
#                      scale_images=False, scaled_img_width=None, crop_size_h=104, crop_size_w=128,
#                      scaled_img_height=None):
#     gated_imgs = []
#     normalizer = 2 ** num_bits - 1.
#
#     for gate_id in range(3):
#         gate_dir = os.path.join(base_dir, 'gated%d_10bit' % gate_id)
#         path = os.path.join(gate_dir, img_id + '.png')
#         assert os.path.exists(path), "No such file : %s" % path
#         img = cv2.imread(os.path.join(gate_dir, img_id + '.png'), cv2.IMREAD_UNCHANGED)
#         # print(img.shape)
#         if data_type == 'real':
#             img = img[crop_size_h:(img.shape[0] - crop_size_h),
#                   crop_size_w:(img.shape[1] - crop_size_w)]
#             img = img.copy()
#             img[img > 2 ** 10 - 1] = normalizer
#
#         img = np.float32(img / normalizer)
#         gated_imgs.append(np.expand_dims(img, axis=2))
#     #        print(img.shape)
#     img = np.concatenate(gated_imgs, axis=2)
#     #    print(img.shape)
#     if scale_images:
#         img = cv2.resize(img, dsize=(scaled_img_width, scaled_img_height), interpolation=cv2.INTER_AREA)
#     return img


def pre(base_dir, img_id):
    gate_dir = os.path.join(base_dir, 'gated{}_10bit'.format(2))
    path = os.path.join(gate_dir, img_id + '.png')
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.float32(img)
    return img


class Gated2DepthDataset(data.Dataset):

    def __init__(self, gated_dir, filenames,
                 height, width, num_scales, depth_normalizer=150.0,
                 frame_idxs=[0],
                 is_train=False):

        assert frame_idxs == [0], "Gated2depth dataset has no temporal frames"
        self.depth_normalizer = depth_normalizer
        self.load_depth = self.check_depth()
        self.depth_loader = read_gt_image
        self.loader = read_gated_image

        self.root_dir = gated_dir
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales

        self.full_res_shape = (1280, 720)
        self.crop_size_h, self.crop_size_w = int((self.full_res_shape[1] - self.height) / 2), int(
            (self.full_res_shape[0] - self.width) / 2),

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.to_tensor = transforms.ToTensor()

        self.K = np.array([[1.81, 0.0, 0.52, 0.0],
                           [0.0, 3.23, 0.36, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        self.pre = pre

    def __getitem__(self, index):

        inputs = {}
        # do_flip  = fasle or true
        do_flip = self.is_train and random.random() > 0.5

        # line = self.filenames[index].split()
        line = self.filenames[index].split('\n')
        frame_index = line[0]
        inputs["pre"] = self.get_pre(frame_index)
        # there is no temporal data for gated2depth dataset
        # photo implement random inversion
        inputs[("gated", 0, -1)] = self.get_gated(frame_index, do_flip)
        # (512,1024,3)
        inputs["depth_gt"], inputs["mask"] = self.get_depth(frame_index, do_flip)
        # (1,512,1024)
        # print(inputs[("gated", 0, -1)].shape)
        # print(inputs["depth_gt"].shape)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        color_aug = (lambda x: x)
        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("gated", i, -1)]
            del inputs[("gated_aug", i, -1)]

        # for k in list(inputs):
        #     print("inputs[{}] shape = {}".format(k, inputs[k].shape))

        return inputs

    def __len__(self):
        # 返回s集大小
        return len(self.filenames)

    def preprocess(self, inputs, color_aug):

        for k in list(inputs):
            frame = inputs[k]
            if "gated" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    # inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
                    s = 2 ** i
                    scaled_img_width, scaled_img_height = self.width // s, self.height // s
                    inputs[(n, im, i)] = cv2.resize(inputs[(n, im, i - 1)], dsize=(scaled_img_width, scaled_img_height),
                                                    interpolation=cv2.INTER_AREA)

        for k in list(inputs):
            f = inputs[k]
            if "gated" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)  # 数据转换为张量，并归一化到0～1
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def get_depth(self, frame_index, do_flip):
        depth_gt, mask = self.depth_loader(self.root_dir, frame_index, 'real', depth_normalizer=self.depth_normalizer)
        # if do_flip:
        #     depth_gt = np.fliplr(depth_gt).copy()
        return depth_gt, mask

    def get_gated(self, frame_index, do_flip):
        gated = self.loader(self.root_dir, frame_index)

        # if do_flip:
        #     gated = np.fliplr(gated).copy() #翻转

        return gated

    def check_depth(self):
        return True  # Gated2Depth dataset has lidar data

    def get_pre(self, frame_index):
        gated = self.pre(self.root_dir, frame_index)
        return gated


if __name__ == "__main__":
    gated_dir = "data/real"
    f = open("splits/real_train_night.txt", "r")
    filenames = f.readlines()
    height = 512
    width = 1024
    num_scales = 1
    is_train = True
    depth_normalizer = 150.0
    frame_idxs = [0]
    train_dataset = Gated2DepthDataset(gated_dir, filenames, height, width, num_scales, depth_normalizer, frame_idxs,
                                       is_train)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,
                                               num_workers=12,
                                               pin_memory=True, drop_last=True)

    for batch_id, inputs in enumerate(train_loader):
        for k in list(inputs):
            f = inputs[k]
            # print(f.shape)
        break
