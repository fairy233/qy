# different loss funciton
import torch
from torch import nn
import torch.nn.functional as F


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
    loss = nn.MSELoss()
    return torch.mean(loss(gen_frames, gt_frames))


def loss(gen_frames, gt_frames):
    return mse_loss(gen_frames, gt_frames)


def loss1(gen_frames, gt_frames):
    gradient = gradient_loss(gen_frames, gt_frames)
    mse = mse_loss(gen_frames, gt_frames)
    return gradient + mse

# loss2 = nn.MSELoss().cuda() + gradient_loss_()
# loss3 = nn.MSELoss().cuda() + pu_ssim_loss_()


# def pu_ssim_loss (gen_frames, gt_frames):
# return 0