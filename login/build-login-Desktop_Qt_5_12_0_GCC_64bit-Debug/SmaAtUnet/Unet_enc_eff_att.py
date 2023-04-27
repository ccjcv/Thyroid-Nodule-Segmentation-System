import argparse
import math
import numpy as np
from SmaAtUnet.unet_parts import *
from SmaAtUnet.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from SmaAtUnet.layers import CBAM
from SmaAtUnet.CoordAttention import CoordAtt
from SmaAtUnet.Unet_eff_att import EfficientAttention
class MultiHeadDense(nn.Module):
    def __init__(self, d, bias=False):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
        if bias:
            raise NotImplementedError()
            self.bias = Parameter(torch.Tensor(d, d))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x:[b, h*w, d]
        b, wh, d = x.size()
        x = torch.bmm(x, self.weight.repeat(b, 1, 1))
        # x = F.linear(x, self.weight, self.bias)
        return x
class att_can(nn.Module):
    def __init__(self, in_ch):
        super(att_can, self).__init__()
        self.bn_l = nn.BatchNorm2d(in_ch)
        self.eff_att = EfficientAttention(dim=in_ch)
    def forward(self, x):
        residue = x
        x1 = self.bn_l(x)
        eff_att = self.eff_att(x1)
        out = eff_att + residue
        return out



class UNet_enc_eff_att(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True):
        super(UNet_enc_eff_att, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.att_can_1 = att_can(128)
        self.down2 = Down(128, 256)
        self.att_can_2 = att_can(256)
        self.down3 = Down(256, 512)
        self.att_can_3 = att_can(512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
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
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits