# different loss funciton
import torch
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np


def gradient_loss(gen_frames, gt_frames):
    # kernel_x = [[0, 0, 0],
    #             [-1., 1., 0],
    #             [0, 0, 0]]
    #
    # kernel_y = [[0, 0, 0],
    #             [0, 1., 0],
    #             [0, -1., 0]]

    # different kernels
    kernel_x = [[-1., -2., -1.],
                [0, 0, 0],
                [1., 2., 1.]]

    kernel_y = [[-1., 0, 1.],
                [-2., 0, 2.],
                [-1., 0, 1.]]
    min_batch = gen_frames.size()[0]
    channels = gen_frames.size()[1]
    out_channel = channels
    kernel_x = torch.FloatTensor(kernel_x).expand(out_channel, channels, 3, 3).cuda()
    kernel_y = torch.FloatTensor(kernel_y).expand(out_channel, channels, 3, 3).cuda()
    # weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    # weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    gen_dx = torch.abs(F.conv2d(gen_frames, kernel_x, stride=1, padding=1))
    gen_dy = torch.abs(F.conv2d(gen_frames, kernel_y, stride=1, padding=1))
    gt_dx = torch.abs(F.conv2d(gt_frames, kernel_x, stride=1, padding=1))
    gt_dy = torch.abs(F.conv2d(gt_frames, kernel_y, stride=1, padding=1))
    grad_diff_x = torch.abs(gt_dx - gen_dx)
    grad_diff_y = torch.abs(gt_dy - gen_dy)
    # condense into one tensor and avg
    return torch.mean(grad_diff_x + grad_diff_y)


def mse_loss(gen_frames, gt_frames):
    loss = nn.MSELoss().cuda()
    return loss(gen_frames, gt_frames)


def l1_loss(gen_frames, gt_frames):
    loss = nn.L1Loss().cuda()
    return loss(gen_frames, gt_frames)


def log_mse_loss(gen_frames, gt_frames, esp=1e-5):
    # log_mse = (torch.log(gen_frames + esp) - torch.log(gt_frames + esp)) ** 2
    # print(log_mse)
    # print(torch.mean(log_mse))
    # return torch.mean(log_mse)
    mse = nn.MSELoss().cuda()
    log_mse = mse(torch.log(gen_frames + esp), torch.log(gt_frames + esp))
    return log_mse


def color_loss(gen_frames, gt_frames):
    # loss_color = (get_hue_value(gen_frames) - get_hue_value(gt_frames)) ** 2
    # return torch.mean(loss_color)
    mse = nn.MSELoss().cuda()
    loss_color = mse(get_hue_value(gen_frames), get_hue_value(gt_frames))
    return loss_color


def cosin_loss(gen_frames, gt_frames):
    loss = torch.nn.CosineSimilarity(dim=1, eps=1e-20)
    return torch.mean(1 - loss(gen_frames, gt_frames))


# 将hdr图像从RGB空间----> HSV空间，得到H通道的值
def get_hue_value(img):
    H = img

    img_max, index_max = torch.max(img, 1)
    img_min, index_min = torch.min(img, 1)

    temp = torch.zeros(img_max.shape).cuda()
    img_r = img[:, 0, :, :]
    img_g = img[:, 1, :, :]
    img_b = img[:, 2, :, :]
    d = img_max - img_min
    index_nz = d != 0
    img_g_sel = img_g[index_nz]
    img_b_sel = img_b[index_nz]
    d_sel = d[index_nz]

    temp[index_nz] = 60 * (img_g_sel - img_b_sel) / d_sel  #
    img_hue = temp
    # index = (index_max==0)*(img_g<img_b) #
    # img_hue[index] = temp[index]+360
    index2 = (index_max == 1) * index_nz
    img_b_sel2 = img_b[index2]
    img_r_sel2 = img_r[index2]
    d_sel2 = d[index2]
    img_hue[index2] = 60 * (img_b_sel2 - img_r_sel2) / d_sel2 + 120

    index3 = (index_max == 2) * index_nz
    img_r_sel3 = img_r[index3]
    img_g_sel3 = img_g[index3]
    d_sel3 = d[index3]

    img_hue[index3] = 60 * (img_r_sel3 - img_g_sel3) / d_sel3 + 240
    zeros = torch.zeros_like(img_hue).cuda()
    zeros[img_hue < 0] = 360
    img_hue = img_hue + zeros
    # print(img_hue.shape)
    return img_hue / 360


# single MSE loss
def loss(gen_frames, gt_frames):
    return mse_loss(gen_frames, gt_frames)


# MSE + gradient
def loss1(gen_frames, gt_frames, alpha=0.01):
    gradient = gradient_loss(gen_frames, gt_frames)
    mse = mse_loss(gen_frames, gt_frames)
    return  mse + alpha * gradient


#  log_mse + color
def loss2(gen_frames, gt_frames, alpha=0.01):
    log_mse = log_mse_loss(gen_frames, gt_frames)
    loss_color = color_loss(gen_frames, gt_frames)
    return log_mse + alpha * loss_color


# L1loss + cosin
def loss3(gen_frames, gt_frames, alpha=0.5):
    l1 = l1_loss(gen_frames, gt_frames)
    cosin = cosin_loss(gen_frames, gt_frames)
    return l1 + alpha * cosin


def cv2torch(np_img):
    rgb = np_img[:, :, (2, 1, 0)].astype(np.float32)
    return torch.from_numpy(rgb.swapaxes(1, 2).swapaxes(0, 1))

# img1 = cv2.imread('hdr/train1/1_0.hdr', flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
# img2 = cv2.imread('hdr/train1/1_1.hdr', flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
# img1 = cv2torch(img1)  # 转为(3,256,256) tensor
# img2 = cv2torch(img2)
# log_mse = log_mse_loss(img1, img2)
# color = color_loss(img1, img2)
# print(log_mse, 'color: ', color)
