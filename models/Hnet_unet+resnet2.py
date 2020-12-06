import torch
import torch.nn as nn
from models.ResBlock import ResidualBlock


class HNet(nn.Module):
    def __init__(self, colordim=6, nhf=64):
        super(HNet, self).__init__()

        # 特征提取
        self.downsamp1 = ResidualBlock(colordim, nhf)
        self.downsamp2 = ResidualBlock(nhf, nhf * 2)
        self.downsamp3 = ResidualBlock(nhf * 2, nhf * 4)
        self.downsamp4 = ResidualBlock(nhf * 4, nhf * 8)  # 特征维度512
        self.downsamp5 = nn.Sequential(
            nn.Conv2d(nhf * 8, nhf * 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(nhf * 16),  # 添加了BN层
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(nhf * 16, nhf * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(nhf * 8),
            nn.LeakyReLU(inplace=True)
        )  # 特征维度512 nhf * 8

        self.upsamp4 = DoubleConv(nhf * 16, nhf * 4)
        self.upsamp3 = DoubleConv(nhf * 8, nhf * 2)
        self.upsamp3 = DoubleConv(nhf * 4, nhf)
        self.upsamp1 = DoubleConv(nhf * 2, nhf // 2)

        self.conv = nn.Conv2d(nhf // 2, 3, 1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.downsamp1(x)
        x2 = MaxPool(x1)

        x2 = self.downsamp2(x2)
        x3 = MaxPool(x2)

        x3 = self.downsamp3(x3)
        x4 = MaxPool(x3)

        x4 = self.downsamp4(x4)
        x5 = MaxPool(x4)

        x5 = self.downsamp5(x5)

        x4_1 = UpSample(x5)
        x4_1 = self.upsamp4(torch.cat((x4_1, x4), 1))

        x3_1 = UpSample(x4_1)
        x3_1 = self.upsamp4(torch.cat((x3_1, x3), 1))

        x2_1 = UpSample(x3_1)
        x2_1 = self.upsamp4(torch.cat((x2_1, x2), 1))

        x1_1 = UpSample(x2_1)
        x1_1 = self.upsamp4(torch.cat((x1_1, x1), 1))

        out = self.conv(x1_1)

        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1),
            nn.BatchNorm2d(in_ch // 2),  # 添加了BN层
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch // 2, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()
        # TODO： 参数注意一下， 要size减半
        self.max = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

    def forward(self, x):
        out = self.max(x)
        return out


class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpSample, self).__init__()
        # TODO： 参数注意一下， 要size变2倍         两种上采样的方法
        # self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        out = self.up(x)
        return out