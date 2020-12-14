# encoding: utf-8
import torch.nn as nn
import torch

# 5个卷积层，conv size/2， k = 2, s = 2, p = 0
# 5个反卷积层， deconv size*2 k= 2, s = 2, p = 0
# TODO: + 1conv 3*3 , filters change


class HNet(nn.Module):
    def __init__(self, colordim=6):
        super(HNet, self).__init__()
        # filters = [50, 100, 200, 400]  # 特征的通道数
        filters = [64, 128, 256, 512]  # 特征的通道数

        # 特征提取
        self.layer1 = nn.Sequential(
            nn.Conv2d(colordim, filters[0], kernel_size=2, stride=2, padding=0),  # size/2
            nn.LeakyReLU(inplace=True),  # inplace-选择是否进行覆盖运算
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=2, stride=2, padding=0),  # size/2
            nn.BatchNorm2d(filters[1]),
            nn.LeakyReLU(inplace=True),  # inplace-选择是否进行覆盖运算
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=2, stride=2, padding=0),  # size/2
            nn.BatchNorm2d(filters[2]),
            nn.LeakyReLU(inplace=True),  # inplace-选择是否进行覆盖运算
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], kernel_size=2, stride=2, padding=0),  # size/2
            nn.BatchNorm2d(filters[3]),  # 400
            nn.LeakyReLU(inplace=True),  # inplace-选择是否进行覆盖运算
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(filters[3], filters[3], kernel_size=2, stride=2, padding=0),  # size/2
            nn.ReLU(inplace=True),  # inplace-选择是否进行覆盖运算
        )

        # 上采样
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(filters[3], filters[3], kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(filters[3]),  # 400
        )
        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(filters[3] * 2, filters[2], kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(filters[2]),  # 400
        )
        self.layer8 = nn.Sequential(
            nn.ConvTranspose2d(filters[2] * 2, filters[1], kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(filters[1]),  # 400
        )
        self.layer9 = nn.Sequential(
            nn.ConvTranspose2d(filters[1] * 2, filters[0], kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(filters[0]),  # 400
        )
        self.layer10 = nn.Sequential(
            nn.ConvTranspose2d(filters[0] * 2, filters[0], kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(filters[0]),  # 400
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(filters[0], 3, kernel_size=3, stride=1, padding=1),  # size
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        x6 = self.layer6(x5)
        x6 = self.relu(torch.cat([x6, x4], dim=1))

        x7 = self.layer7(x6)
        x7 = self.relu(torch.cat([x7, x3], dim=1))

        x8 = self.layer8(x7)
        x8 = self.relu(torch.cat([x8, x2], dim=1))

        x9 = self.layer9(x8)
        x9 = self.relu(torch.cat([x9, x1], dim=1))

        x9 = self.relu(self.layer10(x9))
        x9 = self.layer11(x9)

        return x9
