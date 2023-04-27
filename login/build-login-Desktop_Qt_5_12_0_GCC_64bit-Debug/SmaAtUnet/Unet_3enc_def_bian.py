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
class def_att_5(nn.Module):
    def __init__(self, dim=1024, num_heads=8, qkv_bias: bool = False,
                 dropout_p: float = 0.0, sr_ratio: int = 2):
        super(def_att_5, self).__init__()
        self.SlayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.def_att = DeformableAttention2D(dim=dim)

    def forward(self, x):
        b, c, h, w = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')  # (8,16*16,512)
        x = self.SlayerNorm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x_zi = x.view(b, c, 2, 8, 2, 8)
        x_zi = x_zi.permute(2, 4, 0, 1, 3, 5)#(2,2,8,512,8,8)
        a1 = x_zi[0][0]#(8,512,8,8)
        a2 = x_zi[0][1]
        a3 = x_zi[1][0]
        a4 = x_zi[1][1]
        list = []
        for i in range(2):
            for j in range(2):
                b, c, h, w = x_zi[i][j].size()
                x5_def_att = self.def_att(x_zi[i][j])
                list.append(x5_def_att)
        result_1 = list#四个tensor,(8,64,512)
        result = []
        for i in range(4):
            #result_2 = rearrange(result_1[i], 'b (h w) c -> b c h w', h=8, w=8)
            result.append(result_1[i])#4个tensor,(8,512,8,8)
        shang_liangfutu = torch.cat((result[0], result[1]), dim = 3)#(8,512,8,16)
        xia_liangfutu = torch.cat((result[2], result[3]), dim = 3)#(8,512,8,16)
        si_fu_tu = torch.cat((shang_liangfutu, xia_liangfutu), dim = 2)#(8,512,16,16)
        return si_fu_tu
class def_att_4(nn.Module):
    def __init__(self, dim=512, num_heads=8, qkv_bias: bool = False,
                 dropout_p: float = 0.0, sr_ratio: int = 2):
        super(def_att_4, self).__init__()
        self.SlayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.def_att = DeformableAttention2D(dim=dim)


    def forward(self, x):
        b, c, h, w = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')  # (8,16*16,512)
        x = self.SlayerNorm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x_zi = x.view(b, c, 4, 8, 4, 8)
        x_zi = x_zi.permute(2, 4, 0, 1, 3, 5)#(2,2,8,512,8,8)

        list = []
        for i in range(4):
            for j in range(4):
                b, c, h, w = x_zi[i][j].size()
                x4_def_att = self.def_att(x_zi[i][j])
                list.append(x4_def_att)
        result_1 = list#四个tensor,(8,64,512)
        result = []
        for i in range(16):
            #result_2 = rearrange(result_1[i], 'b (h w) c -> b c h w', h=8, w=8)
            result.append(result_1[i])#4个tensor,(8,512,8,8)
        #每行相加
        hang_1 = torch.cat((result[0], result[1], result[2], result[3]), dim = 3)#(8,512,8,32)
        hang_2 = torch.cat((result[4], result[5], result[6], result[7]), dim = 3)#(8,512,8,32)
        hang_3 = torch.cat((result[8], result[9], result[10], result[11]), dim=3)  # (8,512,8,32)
        hang_4 = torch.cat((result[12], result[13], result[14], result[15]), dim=3)  # (8,512,8,32)
        last = torch.cat((hang_1, hang_2, hang_3, hang_4), dim=2)#(8,512,16,16)
        return last
class def_att_3(nn.Module):
    def __init__(self, dim=256, num_heads=8, qkv_bias: bool = False,
                 dropout_p: float = 0.0, sr_ratio: int = 2):
        super(def_att_3, self).__init__()
        self.SlayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.def_att = DeformableAttention2D(dim=dim)

    def forward(self, x):
        b, c, h, w = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')  # (8,16*16,512)
        x = self.SlayerNorm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x_zi = x.view(b, c, 8, 8, 8, 8)
        x_zi = x_zi.permute(2, 4, 0, 1, 3, 5)#(2,2,8,512,8,8)

        list = []
        for i in range(8):
            for j in range(8):
                b, c, h, w = x_zi[i][j].size()
                x3_def_att = self.def_att(x_zi[i][j])
                list.append(x3_def_att)
        result_1 = list#四个tensor,(8,64,512)
        result = []
        for i in range(64):
            #result_2 = rearrange(result_1[i], 'b (h w) c -> b c h w', h=8, w=8)
            result.append(result_1[i])#4个tensor,(8,512,8,8)
        #每行相加
        hang_1 = torch.cat((result[0], result[1], result[2], result[3],
                            result[4], result[5], result[6], result[7]), dim=3)#(8,512,8,64)
        hang_2 = torch.cat((result[8], result[9], result[10], result[11],
                            result[12], result[13], result[14], result[15]), dim=3)#(8,512,8,64)
        hang_3 = torch.cat((result[16], result[17], result[18], result[19],
                            result[20], result[21], result[22], result[23]), dim=3)  # (8,512,8,64)
        hang_4 = torch.cat((result[24], result[25], result[26], result[27],
                            result[28], result[29], result[30], result[31]), dim=3)  # (8,512,8,64)
        hang_5 = torch.cat((result[32], result[33], result[34], result[35],
                            result[36], result[37], result[38], result[39]), dim=3)  # (8,512,8,64)
        hang_6 = torch.cat((result[40], result[41], result[42], result[43],
                            result[44], result[45], result[46], result[47]), dim=3)  # (8,512,8,64)
        hang_7 = torch.cat((result[48], result[49], result[50], result[51],
                            result[52], result[53], result[54], result[55]), dim=3)  # (8,512,8,64)
        hang_8 = torch.cat((result[56], result[57], result[58], result[59],
                            result[60], result[61], result[62], result[63]), dim=3)  # (8,512,8,64)
        last = torch.cat((hang_1, hang_2, hang_3, hang_4,
                          hang_5, hang_6, hang_7, hang_8), dim=2)#(8,512,16,16)
        return last
class UNet_3enc_def_bian(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = False):
        super(UNet_3enc_def_bian, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.def_att_5 = def_att_5(dim=1024)
        self.conv1_5 = nn.Conv2d(1024, 1, kernel_size=1)
        self.up_bian_5 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)

        self.def_att_4 = def_att_4(512)
        self.conv1_4 = nn.Conv2d(512, 1, kernel_size=1)
        self.up_bian_4 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

        self.def_att_3 = def_att_3(256)
        self.conv1_3 = nn.Conv2d(256, 1, kernel_size=1)
        self.up_bian_3 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3_eff_att = self.def_att_3(x3)
        x3_conv1 = self.conv1_3(x3_eff_att)
        x3_bian = self.up_bian_3(x3_conv1)  # 用来作监督的
        x3_cheng = torch.mul(x3, x3_conv1)
        x3_jia = x3_cheng + x3  # Boundary Enhanced Feature

        x4 = self.down3(x3_jia)
        x4_eff_att = self.def_att_4(x4)
        x4_conv1 = self.conv1_4(x4_eff_att)
        x4_bian = self.up_bian_4(x4_conv1)  # 用来作监督的
        x4_cheng = torch.mul(x4, x4_conv1)
        x4_jia = x4_cheng + x4  # Boundary Enhanced Feature

        x5 = self.down4(x4_jia)
        #x5进行self-attention
        x5_eff_att = self.def_att_5(x5)
        x5_conv1 = self.conv1_5(x5_eff_att)
        x5_bian = self.up_bian_5(x5_conv1)#用来作监督的
        x5_cheng = torch.mul(x5, x5_conv1)
        x5_jia = x5_cheng + x5#Boundary Enhanced Feature

        x = self.up1(x5_jia, x4_jia)
        x = self.up2(x, x3_jia)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        bian_last = x5_bian + x4_bian + x3_bian
        return logits, bian_last