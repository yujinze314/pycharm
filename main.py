import os
import numpy as np
import torch
import torch.optim as optim
import unet
import cv2
from datasets import Gated2DepthDataset
import torch.utils.data as data


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def Loss(output, input, lidar_mask, in_img, pre_img):
    dis = output - input
    l1_loss = dis[lidar_mask].abs().mean()
    # smooth_loss = get_smooth_loss(output,in_img)
    # return 0.85 * l1_loss + 0.15 * smooth_loss
    return l1_loss


def read_img(img_path,
             num_bits=10,
             crop_height=512, crop_width=1024, dataset='g2d'):
    gated_imgs = []
    normalizer = 2 ** num_bits - 1.

    for gate_id in range(3):
        path = img_path.format(gate_id)
        assert os.path.exists(path), "No such file : %s" % path
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img[((img.shape[0] - crop_height) // 2):((img.shape[0] + crop_height) // 2),
              ((img.shape[1] - crop_width) // 2):((img.shape[1] + crop_width) // 2)]
        img = img.copy()
        img[img > 2 ** 10 - 1] = normalizer
        img = np.float32(img / normalizer)
        gated_imgs.append(np.expand_dims(img, axis=2))
    img = np.concatenate(gated_imgs, axis=2)
    return img


def train(models, train_filenames):
    models["Encoder"].train()
    models["Decoder"].train()
    base_dir = 'E:/gated2depth/Gated2Depth-master/data/real/'
    height = 512
    width = 1024
    model_optimizer_Encoder = optim.Adam(models["Encoder"].parameters(), lr=1e-4)
    model_optimizer_Decoder = optim.Adam(models["Decoder"].parameters(), lr=1e-4)

    train_dataset = Gated2DepthDataset(base_dir,
                                       train_filenames,
                                       height, width,
                                       num_scales=4,
                                       depth_normalizer=150.0,
                                       frame_idxs=[0],
                                       is_train=True)

    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True,
                                   num_workers=0,
                                   pin_memory=True, drop_last=True)
    for epoch in range(2):
        i = 0
        for batch_id, inputs in enumerate(train_loader):
            # 将数据拷贝到device中
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(device=device)
            output = models["Decoder"](models["Encoder"](inputs[("gated_aug", 0, 0)]))
            loss = Loss(output["output", 0], inputs["depth_gt"], inputs["mask"], inputs[("gated_aug", 0, 0)],
                        inputs["pre"])

            model_optimizer_Encoder.zero_grad()
            model_optimizer_Decoder.zero_grad()
            loss.backward()
            model_optimizer_Encoder.step()
            model_optimizer_Decoder.step()
            print('epoch:{0}  [{1}/{2}]  loss:{3:.3f}'.format(epoch, i + 1, len(train_filenames), loss))
            i = i + 1
    return models


if __name__ == '__main__':
    data_fpath = 'E:\\gated2depth\\Gated2Depth-master\\splits\\data'
    # 数据加载
    fpath = os.path.join(data_fpath, "{}_files.txt")
    train_filenames = readlines(fpath.format("train"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    models["Encoder"] = unet.Encoder()
    models["Decoder"] = unet.Decoder()
    models["Encoder"].to(device=device)
    models["Decoder"].to(device=device)

    model = train(models, train_filenames)  # train for one epoch
    model_dir = "models/"
    torch.save(model["Encoder"].state_dict(), os.path.join(model_dir, 'Encoder.pth'))
    torch.save(model["Decoder"].state_dict(), os.path.join(model_dir, 'Decoder.pth'))