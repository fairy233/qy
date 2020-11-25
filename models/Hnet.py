import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class HNet(nn.Module):
    def __init__(self, colordim =3):
        super(HNet, self).__init__()
        # cover要输入的网络  N1
        self.convN1_L1_1 = nn.Conv2d(colordim, 64, 3, padding=1)  # input of (n,n,1), output of (n-2,n-2,64)
        self.convN1_L1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.InN1_L1 = nn.InstanceNorm2d(64)

        self.convN1_L2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.convN1_L2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.InN1_L2 = nn.InstanceNorm2d(128)

        self.convN1_L3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.convN1_L3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.convN1_L3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.InN1_L3 = nn.InstanceNorm2d(256)

        self.convN1_L4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.convN1_L4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.convN1_L4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.InN1_L4 = nn.InstanceNorm2d(512)
        
        
#        self.convN1_L5_1 = nn.Conv2d(512, 512, 3, padding=1)
#        self.convN1_L5_2 = nn.Conv2d(512, 512, 3, padding=1)
#        self.InN1_L5 = nn.InstanceNorm2d(512)

#--------------------------------------------------------------
        # secret要输入的网络  N2
        self.convN2_L1_1 = nn.Conv2d(colordim, 64, 3, padding=1)  # input of (n,n,1), output of (n-2,n-2,64)
        self.convN2_L1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.InN2_L1 = nn.InstanceNorm2d(64)

        self.convN2_L2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.convN2_L2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.InN2_L2 = nn.InstanceNorm2d(128)

        self.convN2_L3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.convN2_L3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.convN2_L3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.InN2_L3 = nn.InstanceNorm2d(256)

        self.convN2_L4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.convN2_L4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.convN2_L4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.InN2_L4 = nn.InstanceNorm2d(512)

#        self.convN2_L5_1 = nn.Conv2d(512, 512, 3, padding=1)
#        self.convN2_L5_2 = nn.Conv2d(512, 512, 3, padding=1)
#        self.InN2_L5 = nn.InstanceNorm2d(512)
        
        
#--------------------------------------------------------------
        self.convL5_1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.convL5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.InL5 = nn.InstanceNorm2d(512)
        
        
#--------------------------------------------------------------
#       上采样过程

        self.upconvL8 = nn.Conv2d(1024, 512, 1)
        self.convL8 = nn.Conv2d(512, 256, 3, padding=1)
        self.InL8 = nn.InstanceNorm2d(256)

        self.upconvL9 = nn.Conv2d(512, 256, 1)
        self.convL9 = nn.Conv2d(256, 128, 3, padding=1)
        self.InL9 = nn.InstanceNorm2d(128)

        self.upconvL10 = nn.Conv2d(256, 128, 1)
        self.convL10 = nn.Conv2d(128, 64, 3, padding=1)
        self.InL10 = nn.InstanceNorm2d(64)

        self.upconvL11 = nn.Conv2d(128, 64, 1)
        self.InL11 = nn.InstanceNorm2d(64)
        
        self.upconvL12 = nn.Conv2d(64, 3, 1)
        self.InL12 = nn.InstanceNorm2d(3)
        
        self.upconvL13 = nn.Conv2d(6, 3, 1)
        self.InL13 = nn.InstanceNorm2d(3)

#------------------------------------------------------------------------------        
        self.convC = nn.Conv2d(colordim*3,colordim,1)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        # self.avgpool = nn.AvgPool2d(2, stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2) 
        
 
    def forward(self, x0):
#        输入x0，是对cover和secret图像合并为6通道之后进行输入，需要对其进行分解
        c = x0[:,:3,:,:]
        s = x0[:,3:6,:,:]
#-----------------------------------------------------------------------------------   

        c1 = F.leaky_relu(self.InN1_L1(self.convN1_L1_2(F.leaky_relu(self.convN1_L1_1(c))))) # 64, 320 
        c2 = F.leaky_relu(self.InN1_L2(self.convN1_L2_2(F.leaky_relu(self.convN1_L2_1(self.maxpool(c1)))))) # 128, 160
        c3 = F.leaky_relu(self.InN1_L3(self.convN1_L3_3(F.leaky_relu(self.convN1_L3_2(F.leaky_relu(self.convN1_L3_1(self.maxpool(c2)))))))) # 256, 80 
        c4 = F.leaky_relu(self.InN1_L4(self.convN1_L4_3(F.leaky_relu(self.convN1_L4_2(F.leaky_relu(self.convN1_L4_1(self.maxpool(c3))))))))# 512, 40
        
        
        s1 = F.leaky_relu(self.InN2_L1(self.convN2_L1_2(F.leaky_relu(self.convN1_L1_1(s))))) # 64, 320 
        s2 = F.leaky_relu(self.InN2_L2(self.convN2_L2_2(F.leaky_relu(self.convN1_L2_1(self.maxpool(s1)))))) # 128, 160
        s3 = F.leaky_relu(self.InN2_L3(self.convN2_L3_3(F.leaky_relu(self.convN1_L3_2(F.leaky_relu(self.convN2_L3_1(self.maxpool(s2)))))))) # 256, 80 
        s4 = F.leaky_relu(self.InN2_L4(self.convN2_L4_3(F.leaky_relu(self.convN1_L4_2(F.leaky_relu(self.convN2_L4_1(self.maxpool(s3))))))))# 512, 40

        
#-----------------------------------------------------------------------------------
        # 将cover和secret提取出来的特征合并， 按照第一维度合并
        feature = torch.cat([c4, s4], dim=1) # 1024 40
        # pdb.set_trace()
#-----------------------------------------------------------------------------------
        
        stego = F.leaky_relu(self.InL5(self.convL5_2(F.leaky_relu(self.convL5_1(self.maxpool(feature))))))  #    20
   
#-----------------------------------------------------------------------------------   

        stego = F.leaky_relu(self.InL8(self.convL8(self.upconvL8(torch.cat((self.upsample(stego), c4),1))))) # 256, 40
        stego = F.leaky_relu(self.InL9(self.convL9(self.upconvL9(torch.cat((self.upsample(stego), c3),1)))))# 128, 80
        stego = F.leaky_relu(self.InL10(self.convL10(self.upconvL10(torch.cat((self.upsample(stego), c2),1))))) # 64, 160
        stego = F.leaky_relu(self.InL11(self.upconvL11(torch.cat((self.upsample(stego), c1),1)))) # 64, 320
        stego = F.leaky_relu(self.InL12(self.upconvL12(stego))) # 3, 320
        stego = F.relu(self.upconvL13(torch.cat((stego, c),1))) # 3, 320

#----------------------------------------------------------------------------------- 

        return stego

 


