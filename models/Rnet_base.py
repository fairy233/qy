# encoding: utf-8
import torch.nn as nn
# 论文： Simultaneous convolutional neural network for highly efficient image steganography
# 6个卷积层，size不变， k = 3, s = 3, p = 1

# nhf  特征数量
class RNet(nn.Module):
    def __init__(self, colordim=3,output_function=nn.Sigmoid):
        super(RNet, self).__init__()
        filters = [64, 128, 256, 128, 64]  # 特征的通道数
        self.main = nn.Sequential(
            nn.Conv2d(colordim, filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True),

            nn.Conv2d(filters[0], filters[1], 3, 1, 1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(True),

            nn.Conv2d(filters[1], filters[2], 3, 1, 1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(True),

            nn.Conv2d(filters[2], filters[3], 3, 1, 1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(True),

            nn.Conv2d(filters[3], filters[4], 3, 1, 1),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(True),

            nn.Conv2d(filters[4], colordim, 3, 1, 1),
            output_function()
        )

    def forward(self, x):
        output = self.main(x)
        return output

