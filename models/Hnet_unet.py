import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ResBlock import ResidualBlock

# TODO: 网络模型改小，1、卷积核的特征数减半， 2、下采样层数减少一层  目前采取1
# TODO: 网络结构打算采用 unet+resnet, 在每一个连接层上加残差网络，
class HNet(nn.Module):
    def __init__(self, colordim=3):
        super(HNet, self).__init__()

        # cover 特征提取
        self.downsamp1_c = nn.Sequential(
            DoubleConv(3, 32),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)  # size 减半
        )
        self.downsamp2_c = nn.Sequential(
            DoubleConv(32, 64),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        )
        self.downsamp3_c = nn.Sequential(
            DoubleConv(64, 128),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

        )
        self.downsamp4_c = nn.Sequential(
            DoubleConv(128, 256),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        )
        self.downsamp5_c = nn.Sequential(
            DoubleConv(256, 512),
        )

        # secret 特征提取
        self.downsamp1_s = nn.Sequential(
            DoubleConv(3, 32),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        )
        self.downsamp2_s = nn.Sequential(
            DoubleConv(32, 64),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        )
        self.downsamp3_s = nn.Sequential(
            DoubleConv(64, 128),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

        )
        self.downsamp4_s = nn.Sequential(
            DoubleConv(128, 256),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        )

        self.downsamp5_s = nn.Sequential(
            DoubleConv(256, 512),  # 特征维度512
        )

        # 把cover和secret的特征concat之后，512+512变为1024维度
        # 特征要再降维到512维度,size大小不变
        self.downsamp_feature = nn.Conv2d(1024, 512, 1, padding=1)

        # 上采样+卷积
        self.upsamp1 = nn.UpsamplingBilinear2d(scale_factor=2),
        self.upsamp1_conv = DoubleConv(1024, 512)

        self.upsamp2 = nn.UpsamplingBilinear2d(scale_factor=2),
        self.upsamp2_conv = DoubleConv(512, 256)

        self.upsamp3 = nn.UpsamplingBilinear2d(scale_factor=2),
        self.upsamp3_conv = DoubleConv(256, 128)

        self.upsamp4 = nn.UpsamplingBilinear2d(scale_factor=2),
        self.upsamp4_conv = DoubleConv(128, 64)

        self.upsamp4_conv1 = nn.Conv2d(64, 3, 1, padding=1)

    def forward(self, x):
        c = x[:, :3, :, :]
        s = x[:, 3:6, :, :]

        c1 = self.downsamp1_c(c)
        c2 = self.downsamp2_c(c1)
        c3 = self.downsamp3_c(c2)
        c4 = self.downsamp4_c(c3)
        c5 = self.downsamp5_c(c4)  # 特征维度512

        s1 = self.downsamp1_s(s)
        s1 = self.downsamp2_s(s1)
        s1 = self.downsamp3_s(s1)
        s1 = self.downsamp4_s(s1)
        s1 = self.downsamp5_s(s1)  # 特征维度512

        # 将cover和stego特征合并
        feature = torch.cat([c5, s1], dim=1)  # 特征维度512+512为1024
        feature = self.downsamp_feature(feature),  # 将特征维度从1024降维到512

        feature = self.upsamp1(feature)
        stego = torch.cat((feature, c4), 1)
        stego = self.upsamp1_conv(stego)

        stego = self.upsamp2(stego)
        stego = torch.cat((stego, c3), 1)
        stego = self.upsamp2_conv(stego)

        stego = self.upsamp3(stego)
        stego = torch.cat((stego, c2), 1)
        stego = self.upsamp3_conv(stego)

        stego = self.upsamp4(stego)
        stego = torch.cat((stego, c1), 1)
        stego = self.upsamp4_conv(stego)

        return stego


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),  # 添加了BN层
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU()
        )

    def forward(self, x):
        out = self.conv(x)
        return out