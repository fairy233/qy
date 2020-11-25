# encoding: utf-8
import torch.nn as nn


# nhf  特征数量
class RNet(nn.Module):
    def __init__(self, colordim=3, nhf=64, output_function=nn.Sigmoid):
        super(RNet, self).__init__()
            
        self.main = nn.Sequential(
            nn.Conv2d(colordim, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            
            nn.Conv2d(nhf, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            
            nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),
            nn.BatchNorm2d(nhf*4),
            nn.ReLU(True),
            
            nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            
            nn.Conv2d(nhf * 2, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            
            nn.Conv2d(nhf, colordim, 3, 1, 1),
            
            output_function()
        )

    def forward(self, x):
        output = self.main(x)
        return output