import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import dropout, gelu
import numpy as np
import argparse
import math
from SmaAtUnet.unet_parts import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from SmaAtUnet.Unet_eff_att import EfficientAttention
from SmaAtUnet.layers import DepthwiseSeparableConv
from SmaAtUnet.deform_attention import DeformableAttention2D

class lianglu_att(nn.Module):
    def __init__(self,
                 qkv_bias: bool = False,
                 kernels_per_layer=1
                 ):
        super(lianglu_att, self).__init__()
        self.conv256 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.lin = nn.Linear(256, 256, bias=qkv_bias)
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(256, 256, kernel_size=3, kernels_per_layer=kernels_per_layer,
                                   padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(256, 256, kernel_size=3, kernels_per_layer=kernels_per_layer,
                                   padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.def_att = DeformableAttention2D(dim=256)
        self.eff_att = EfficientAttention(dim=256)


    def forward(self, x):
        #分两路,按通道划分
        b, C, h, w = x.shape
        c = C / 2
        x_gao = self.conv256(x)#(8,256,16,16)
        x_att = self.conv256(x)

        # x_gao_1 = rearrange(x_gao, 'b c h w -> b (h w) c')#(8,256,256)
        # x_gao_1 = self.lin(x_gao_1)#(8,256,256)
        # x_gao_1 = rearrange(x_gao_1, 'b (h w) c -> b c h w', h=16, w=16)
        # x_gao_1 = self.double_conv(x_gao_1)

        x_gao = self.def_att(x_gao)

        x_di = self.eff_att(x_att)

        x_two = torch.cat((x_gao, x_di), dim=1)

        return x_two

class UNet_lianglu(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True):
        super(UNet_lianglu, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.two_att = lianglu_att()
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #x5进行self-attention
        x5_two_att = self.two_att(x5)
        x = self.up1(x5_two_att, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits