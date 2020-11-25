# # encoding: utf-8
#
# """
# Adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
#
# """
#
# import functools
#
# import torch
# import torch.nn as nn
#
#
# # Defines the Unet generator.
# # |num_downs|: number of downsamplings in UNet. For example,
# # if |num_downs| == 7, image of size 128x128 will become of size 1x1
# # at the bottleneck
# # 有7层下采样，最终会将128*128的图像变为尺寸为1*1
#
# class UnetGenerator(nn.Module):
#     def __init__(self, input_nc, output_nc, num_downs, ngf=64,
#                  norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Sigmoid):
#         super(UnetGenerator, self).__init__()
#         # construct unet structure
#         unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
#         for i in range(num_downs - 5):
#         unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
#         unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, output_function=output_function)
#
#         self.model = unet_block
#
#     def forward(self, input):
#         return self.model(input)
#
#
# # Defines the submodule with skip connection.
# # X -------------------identity---------------------- X
# #   |-- downsampling -- |submodule| -- upsampling --|
#
# class UnetSkipConnectionBlock(nn.Module):
#     def __init__(self, outer_nc, inner_nc, input_nc=None,submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Sigmoid):
#         super(UnetSkipConnectionBlock, self).__init__()
#         self.outermost = outermost
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#         if input_nc is None:
#             input_nc = outer_nc
#         downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
#                              stride=2, padding=1, bias=use_bias)
#         downrelu = nn.LeakyReLU(0.2, True)
#         downnorm = norm_layer(inner_nc)
#         uprelu = nn.ReLU(True)
#         upnorm = norm_layer(outer_nc)
#
#         if outermost:
#             upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1)
#             down = [downconv]
#             if output_function == nn.Tanh:
#                 up = [uprelu, upconv, nn.Tanh()]
#             else:
#                 up = [uprelu, upconv, nn.Sigmoid()]
#             model = down + [submodule] + up
#         elif innermost:
#             upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downrelu, downconv]
#             up = [uprelu, upconv, upnorm]
#             model = down + up
#         else:
#             upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downrelu, downconv, downnorm]
#             up = [uprelu, upconv, upnorm]
#
#             if use_dropout:
#                 model = down + [submodule] + up + [nn.Dropout(0.5)]
#             else:
#                 model = down + [submodule] + up
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, x):
#         if self.outermost:
#             return self.model(x)
#         else:
#             return torch.cat([x, self.model(x)], 1)
#         # cat函数，将两个张量tensor拼接在一起，最后一个参数是维度，0是竖着拼接，1是横着拼接
#         #outermost innermost 来控制是否要这个残差层，是否要加该层的输入
#
#
#
# #        import torch
# #import torch.nn as nn
# #import torch.nn.functional as F
# #import numpy as np
# #from torchsummary import summary
# #
# ##  Conv2d参数： in_channels, out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
# ## LeakyReLU_2参数： 控制负斜率的角度，默认是0.01， inplace，是否进行覆盖运算，
# ## BatchNorm2d: 对数据进行归一化处理，使数据在输入激活函数之前，不会因为数据值过大而导致网络性能不稳定
# ## 参数： num_features: 特征的数量， eps: 为了计算稳定性添加的值，默认为1e-5,  momentum: 估计参数， affine: 布尔值，设为true时，会给定学习的系数矩阵伽马和贝塔
# ## convTranspose2d 反卷积参数： in_channels, out, kernel_size, stride=1, padding=0输入边补0的层数，高宽都增加2padding, output_padding=0,输出边补0的层数，高宽都增加padding groups=1,从输入通道到输出通道的阻塞连接数 bias=True添加偏置
# #class HNet(nn.Module):
# #    def __init__(self, colordim=6): # 输入通道数为6，对cover和secret图像进行预处理,把两个图像合并为一个6通道的图像
# #        # 我的想法：  cover和secret图像都是3通道，分别输入，得到两个特征向量，然后将两个特征向量合并为一个
# #        super(HNet, self).__init__()
# #        self.conv2d_1 = nn.Conv2d(colordim, 50, 3, stride=2, padding=1,)# 步长为2，就可以包含unet中的池化操作
# #        self.LeakyReLU_2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
# #
# #        self.conv2d_3 = nn.Conv2d(50, 100, 3, stride=2, padding=1)
# #        self.BatchNorm2d_4 = nn.BatchNorm2d(n,affine=True)
# #        self.LeakyReLU_5 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
# #
# #        self.conv2d_6 = nn.Conv2d(100, 200, 3, stride=2, padding=1)
# #        self.BatchNorm2d_7 = nn.BatchNorm2d(n, affine=True)
# #        self.LeakyReLU_8 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
# #
# #        self.conv2d_9 = nn.Conv2d(200, 400, 3, stride=2, padding=1)
# #        self.BatchNorm2d_10 = nn.BatchNorm2d(n, affine=True)
# #        self.LeakyReLU_11 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
# #
# #        self.conv2d_12 = nn.Conv2d(50, 100, 3, stride=2, padding=1)
# #        self.ReLU_13 = nn.ReLU(inplace=False)
# #
# #        # --------------------------------------------------------------
# #
# #        # 这里参数还没修改！！
# #        self.convTranspose2d_14 = nn.ConvTranspose2d(50, 100, 3, stride=2, padding=1)
# #        self.BatchNorm2d_15 = nn.BatchNorm2d(n, affine=True)
# #        self.SkipConnection_16 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
# #        self.ReLU_17 = nn.ReLU(inplace=False)
# #
# #        self.convTranspose2d_18 = nn.ConvTranspose2d(50, 100, 3, stride=2, padding=1)
# #        self.BatchNorm2d_19 = nn.BatchNorm2d(n, affine=True)
# #        self.SkipConnection_20 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
# #        self.ReLU_21 = nn.ReLU(inplace=False)
# #
# #        self.convTranspose2d_22 = nn.ConvTranspose2d(50, 100, 3, stride=2, padding=1)
# #        self.BatchNorm2d_23 = nn.BatchNorm2d(n, affine=True)
# #        self.SkipConnection_24 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
# #        self.ReLU_25 = nn.ReLU(inplace=False)
# #
# #        self.convTranspose2d_26 = nn.ConvTranspose2d(50, 100, 3, stride=2, padding=1)
# #        self.BatchNorm2d_27 = nn.BatchNorm2d(n, affine=True)
# #        self.SkipConnection_28 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
# #        self.ReLU_29 = nn.ReLU(inplace=False)
# #
# #        self.convTranspose2d_30 = nn.ConvTranspose2d(50, 100, 3, stride=2, padding=1)
# #        self.Sigmoid_31 = nn.Sigmoid()
# #        self.SkipConnection_32 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
# #
# #
# #
# #
# #def conv3x3(in_planes, out_planes, stride=1):
# #    """3x3 convolution with padding"""
# #    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride)
# #
# #
# #def conv1x1(in_planes, out_planes, stride=1):
# #    """1x1 convolution without padding"""
# #    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)
# #
# #
# #def up_conv2x2(in_planes, out_planes):
# #    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2)
# #
# #
# #def max_pool2x2():
# #    return nn.MaxPool2d(kernel_size=2, stride=2)
# #
# #
# #class UNet(nn.Module):
# #    def __init__(self, class_num=1000):
# #        super(UNet, self).__init__()
# #        # downsample stage
# #        self.conv_1 = nn.Sequential(conv3x3(1, 64), conv3x3(64, 64))
# #        self.conv_2 = nn.Sequential(conv3x3(64, 128), conv3x3(128, 128))
# #        self.conv_3 = nn.Sequential(conv3x3(128, 256), conv3x3(256, 256))
# #        self.conv_4 = nn.Sequential(conv3x3(256, 512), conv3x3(512, 512))
# #        self.conv_5 = nn.Sequential(conv3x3(512, 1024), conv3x3(1024, 1024))
# #        self.maxpool = max_pool2x2()
# #
# #        # upsample stage
# #        # up_conv_4 corresponds conv_4
# #        self.up_conv_4 = nn.Sequential(up_conv2x2(1024, 512))
# #        # conv the cat(stage_4,up_conv_4) from 1024 to 512
# #        self.conv_6 = nn.Sequential(conv3x3(1024, 512), conv3x3(512, 512))
# #        # up_conv_3 corresponds conv_3
# #        self.up_conv_3 = nn.Sequential(up_conv2x2(512, 256))
# #        # conv the cat(stage_3,up_conv_3) from 512 to 256
# #        self.conv_7 = nn.Sequential(conv3x3(512, 256), conv3x3(256, 256))
# #        # up_conv_2 corresponds conv_2
# #        self.up_conv_2 = nn.Sequential(up_conv2x2(256, 128))
# #        # conv the cat(stage_2,up_conv_2) from 256 to 128
# #        self.conv_8 = nn.Sequential(conv3x3(256, 128), conv3x3(128, 128))
# #        # up_conv_1 corresponds conv_1
# #        self.up_conv_1 = nn.Sequential(up_conv2x2(128, 64))
# #        # conv the cat(stage_1,up_conv_1) from 128 to 64
# #        self.conv_9 = nn.Sequential(conv3x3(128, 64), conv3x3(64, 64))
# #        # output
# #        self.result = conv1x1(64, 2)
# #
# #    def _concat(self, tensor1, tensor2):
# #        # concat 2 tensor by the channel axes
# #        tensor1, tensor2 = (tensor1, tensor2) if tensor1.size()[3] >= tensor2.size()[3] else (tensor2, tensor1)
# #        crop_val = int((tensor1.size()[3] - tensor2.size()[3]) / 2)
# #        tensor1 = tensor1[:, :, crop_val:tensor1.size()[3] - crop_val
# #        , crop_val:tensor1.size()[3] - crop_val]
# #        return torch.cat((tensor1, tensor2), 1)
# #
# #    def forward(self, x):
# #        # get 4 stage conv output
# #        stage_1 = self.conv_1(x)
# #        stage_2 = self.conv_2(self.maxpool(stage_1))
# #        stage_3 = self.conv_3(self.maxpool(stage_2))
# #        stage_4 = self.conv_4(self.maxpool(stage_3))
# #
# #        # get up_conv_4 and concat with stage_4
# #        up_in_4 = self.conv_5(self.maxpool(stage_4))
# #        up_stage_4 = self.up_conv_4(up_in_4)
# #        up_stage_4 = self._concat(stage_4, up_stage_4)
# #        # get up_conv_3 and concat with stage_3
# #        up_in_3 = self.conv_6(up_stage_4)
# #        up_stage_3 = self.up_conv_3(up_in_3)
# #        up_stage_3 = self._concat(stage_3, up_stage_3)
# #        # get up_conv_2 and concat with stage_2
# #        up_in_2 = self.conv_7(up_stage_3)
# #        up_stage_2 = self.up_conv_2(up_in_2)
# #        up_stage_2 = self._concat(stage_2, up_stage_2)
# #        # get up_conv_1 and concat with stage_1
# #        up_in_1 = self.conv_8(up_stage_2)
# #        up_stage_1 = self.up_conv_1(up_in_1)
# #        up_stage_1 = self._concat(stage_1, up_stage_1)
# #
# #        # last conv to channel 2
# #        out = self.conv_9(up_stage_1)
# #        # result
# #        out = self.result(out)
# #        return out