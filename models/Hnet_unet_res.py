import torch
import torch.nn as nn
from models.ResBlock import ResidualBlock

# TODO: 网络模型改小，1、卷积核的特征数减半， 2、下采样层数减少一层  目前采取1
# TODO: 网络结构打算采用 unet+resnet, 在每一个连接层上加残差网络，
class HNet(nn.Module):
    def __init__(self, colordim=3):
        super(HNet, self).__init__()

        # cover 特征提取
        self.downsamp1_c = ResidualBlock(3, 32)
        self.max = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)  # size 减半

        self.downsamp2_c = ResidualBlock(32, 64)

        self.downsamp3_c = ResidualBlock(64, 128)

        self.downsamp4_c = ResidualBlock(128, 256)  # 特征维度256

        # secret 特征提取
        self.downsamp1_s = ResidualBlock(3, 32)

        self.downsamp2_s = ResidualBlock(32, 64)

        self.downsamp3_s = ResidualBlock(64, 128)

        self.downsamp4_s = ResidualBlock(128, 256)  # 特征维度256

        self.downsamp5 = nn.Sequential(
            ResidualBlock(512, 512),  # 特征维度512
        )

        # 把cover和secret的特征concat之后，256+256变为512维度
        self.downsamp_feature = ResidualBlock(512, 256)

        # 上采样+卷积 nn.ConvTranspose2d()
        # self.upsamp1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.upsamp1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsamp1_conv = DoubleConv(512, 128)

        # self.upsamp2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upsamp2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsamp2_conv = DoubleConv(256, 64)

        # self.upsamp3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upsamp3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsamp3_conv = DoubleConv(128, 32)

        # self.upsamp4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upsamp4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsamp4_conv = DoubleConv(64, 16)

        self.conv = nn.Conv2d(16, 3, 1, padding=0)

    def forward(self, x):
        c = x[:, :3, :, :]
        s = x[:, 3:6, :, :]

        c1 = self.downsamp1_c(c)
        c1_1 = self.max(c1)
        c2 = self.downsamp2_c(c1_1)
        c2_1 = self.max(c2)
        c3 = self.downsamp3_c(c2_1)
        c3_1 = self.max(c3)
        c4 = self.downsamp4_c(c3_1)
        c4_1 = self.max(c4)

        s1 = self.downsamp1_s(s)
        s1 = self.max(s1)
        s1 = self.downsamp2_s(s1)
        s1 = self.max(s1)
        s1 = self.downsamp3_s(s1)
        s1 = self.max(s1)
        s1 = self.downsamp4_s(s1)
        s1 = self.max(s1)

        # 将cover和stego特征合并
        feature = torch.cat([c4_1, s1], dim=1)  # 特征维度256+256为512, (4,512,20,20)
        feature = self.downsamp_feature(feature)  # 将特征维度从512降维到256  (4,256,20,20)
        feature = self.upsamp1(feature)  # (4,256,40,40)

        ste = torch.cat((feature, c4), 1)  # (4,512,40,40)
        ste = self.upsamp1_conv(ste)  # (4,128,40,40)

        ste = self.upsamp2(ste)   # (4,128,80,80)
        ste = torch.cat((ste, c3), 1)     # (4,256,80,80)
        ste = self.upsamp2_conv(ste)  # (4,64,80,80)

        ste = self.upsamp3(ste)  # (4,64,160,160)
        ste = torch.cat((ste, c2), 1)  # (4,128,160,160)
        ste = self.upsamp3_conv(ste)  # (4,32,160,160)

        ste = self.upsamp4(ste)  # (4,32,320,320)
        ste = torch.cat((ste, c1), 1)   # (4,64,320,320)

        ste = self.upsamp4_conv(ste)   # (4,16,320,320)
        ste = self.conv(ste)  # (4,3,320,320)
        return ste


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1),
            nn.BatchNorm2d(in_ch // 2),  # 添加了BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out