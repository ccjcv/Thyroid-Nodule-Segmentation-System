import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import dropout, gelu
import numpy as np
import argparse
import math
from SmaAtUnet.unet_parts import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from SmaAtUnet.deform_attention_2 import DeformableAttention2D
class UNet_enc_def_bian(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = False):
        super(UNet_enc_def_bian, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.def_att_5 = DeformableAttention2D(dim=1024)
        self.conv1_5 = nn.Conv2d(1024, 1, kernel_size=1)
        self.up_bian_5 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)

        # self.def_att_4 = def_att_4(512)
        # self.conv1_4 = nn.Conv2d(512, 1, kernel_size=1)
        # self.up_bian_4 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        #
        # self.def_att_3 = def_att_3(256)
        # self.conv1_3 = nn.Conv2d(256, 1, kernel_size=1)
        # self.up_bian_3 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x3_eff_att = self.def_att_3(x3)
        # x3_conv1 = self.conv1_3(x3_eff_att)
        # x3_bian = self.up_bian_3(x3_conv1)  # 用来作监督的
        # x3_cheng = torch.mul(x3, x3_conv1)
        # x3_jia = x3_cheng + x3  # Boundary Enhanced Feature

        x4 = self.down3(x3)
        # x4_eff_att = self.def_att_4(x4)
        # x4_conv1 = self.conv1_4(x4_eff_att)
        # x4_bian = self.up_bian_4(x4_conv1)  # 用来作监督的
        # x4_cheng = torch.mul(x4, x4_conv1)
        # x4_jia = x4_cheng + x4  # Boundary Enhanced Feature

        x5 = self.down4(x4)
        #x5进行self-attention
        x5_eff_att = self.def_att_5(x5)
        x5_conv1 = self.conv1_5(x5_eff_att)
        x5_bian = self.up_bian_5(x5_conv1)#用来作监督的
        x5_cheng = torch.mul(x5, x5_conv1)
        x5_jia = x5_cheng + x5#Boundary Enhanced Feature

        x = self.up1(x5_jia, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        bian_last = x5_bian
        return logits, bian_last