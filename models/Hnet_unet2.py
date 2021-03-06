import torch
import torch.nn as nn
from models.ResBlock import ResidualBlock
from models.init_weights import init_weights


class HNet(nn.Module):
    def __init__(self, colordim=6, nhf=64):
        super(HNet, self).__init__()

        # 特征提取
        self.downsamp1 = DoubleConv(colordim, nhf, nhf)
        self.downsamp2 = DoubleConv(nhf, nhf * 2, nhf * 2)
        self.downsamp3 = DoubleConv(nhf * 2, nhf * 4, nhf * 4)
        self.downsamp4 = DoubleConv(nhf * 4, nhf * 8, nhf * 8)  # 特征维度512
        self.downsamp5 = DoubleConv(nhf * 8, nhf * 16, nhf * 8)  # 特征维度512 nhf * 8

        self.upsamp4 = DoubleConv(nhf * 16, nhf * 8, nhf * 4)
        self.upsamp3 = DoubleConv(nhf * 8, nhf * 4, nhf * 2)
        self.upsamp2 = DoubleConv(nhf * 4, nhf * 2, nhf)
        self.upsamp1 = DoubleConv(nhf * 2, nhf, int(nhf/2))

        self.max = nn.MaxPool2d(2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv2d(int(nhf / 2), 3, 1, stride=1, padding=0)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.downsamp1(x)
        x2 = self.max(x1)

        x2 = self.downsamp2(x2)
        x3 = self.max(x2)

        x3 = self.downsamp3(x3)
        x4 = self.max(x3)

        x4 = self.downsamp4(x4)
        x5 = self.max(x4)

        x5 = self.downsamp5(x5)

        x4_1 = self.up(x5)
        x4_1 = self.upsamp4(torch.cat((x4_1, x4), 1))

        x3_1 = self.up(x4_1)
        x3_1 = self.upsamp3(torch.cat((x3_1, x3), 1))

        x2_1 = self.up(x3_1)
        x2_1 = self.upsamp2(torch.cat((x2_1, x2), 1))

        x1_1 = self.up(x2_1)
        x1_1 = self.upsamp1(torch.cat((x1_1, x1), 1))

        out = self.conv(x1_1)
        out = self.activation(out)

        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, inner, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, inner, 3, stride=1, padding=1),
            nn.BatchNorm2d(inner),  # 添加了BN层
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inner, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
        for m in self.children():  # self.children()存储网络结构的子层模块，一层一层
            init_weights(m, init_type='kaiming')  # 对每一层参数进行初始化

    def forward(self, x):
        out = self.conv(x)
        return out


