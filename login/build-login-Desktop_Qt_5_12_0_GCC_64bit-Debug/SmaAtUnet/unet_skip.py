import argparse
from SmaAtUnet.unet_parts import *
from SmaAtUnet.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from SmaAtUnet.layers import CBAM
from SmaAtUnet.CoordAttention import CoordAtt
from SmaAtUnet.external_attention_2 import External_attention
from SmaAtUnet.deform_attention_2 import DeformableAttention2D
from SmaAtUnet.semantic_attention import SemanticAttention
class UNet_skip_CA(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True,
                 reduction_ratio: int = 16):
        super(UNet_skip_CA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.inc = DoubleConv(self.n_channels, 64)
        self.ca1 = CoordAtt(64, 64, reduction=reduction_ratio)
        self.down1 = Down(64, 128)
        self.ca2 = CoordAtt(128, 128, reduction=reduction_ratio)
        self.down2 = Down(128, 256)
        self.ca3 = CoordAtt(256, 256, reduction=reduction_ratio)
        self.down3 = Down(256, 512)
        self.ca4 = CoordAtt(512, 512, reduction=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1_ = self.ca1(x1)
        x2 = self.down1(x1)
        x2_ = self.ca2(x2)
        x3 = self.down2(x2)
        x3_ = self.ca3(x3)
        x4 = self.down3(x3)
        x4_ = self.ca4(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4_)
        x = self.up2(x, x3_)
        x = self.up3(x, x2_)
        x = self.up4(x, x1_)
        logits = self.outc(x)
        return logits
from SmaAtUnet.deform_conv import DeformConv2d


class UNet_skip_deform_conv(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True,
                 reduction_ratio: int = 16):
        super(UNet_skip_deform_conv, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.inc = DoubleConv(self.n_channels, 64)
        self.red1 = nn.Conv2d(64, 32, 1)
        self.def1 = DeformConv2d(32, 32)
        self.rec1 = nn.Conv2d(32, 64, 1)
        self.down1 = Down(64, 128)
        self.red2 = nn.Conv2d(128, 64, 1)
        self.def2 = DeformConv2d(64, 64)
        self.rec2 = nn.Conv2d(64, 128, 1)
        self.down2 = Down(128, 256)
        self.red3 = nn.Conv2d(256, 128, 1)
        self.def3 = DeformConv2d(128, 128)
        self.rec3 = nn.Conv2d(128, 256, 1)
        self.down3 = Down(256, 512)
        self.red4 = nn.Conv2d(512, 256, 1)
        self.def4 = DeformConv2d(256, 256)
        self.rec4 = nn.Conv2d(256, 512, 1)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1_ = self.rec1(self.def1(self.red1(x1)))
        x2 = self.down1(x1)
        x2_ = self.rec2(self.def2(self.red2(x2)))
        x3 = self.down2(x2)
        x3_ = self.rec3(self.def3(self.red3(x3)))
        x4 = self.down3(x3)
        x4_ = self.rec4(self.def4(self.red4(x4)))
        x5 = self.down4(x4)
        x = self.up1(x5, x4_)
        x = self.up2(x, x3_)
        x = self.up3(x, x2_)
        x = self.up4(x, x1_)
        logits = self.outc(x)
        return logits

class UNet_skip_CBAM(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True,
                 reduction_ratio: int = 16):
        super(UNet_skip_CBAM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.inc = DoubleConv(self.n_channels, 64)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = Down(64, 128)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = Down(128, 256)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = Down(256, 512)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1_ = self.cbam1(x1)
        x2 = self.down1(x1)
        x2_ = self.cbam2(x2)
        x3 = self.down2(x2)
        x3_ = self.cbam3(x3)
        x4 = self.down3(x3)
        x4_ = self.cbam4(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4_)
        x = self.up2(x, x3_)
        x = self.up3(x, x2_)
        x = self.up4(x, x1_)
        logits = self.outc(x)
        return logits
class UNet_skip_deform_attention(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True,
                 reduction_ratio: int = 16):
        super(UNet_skip_deform_attention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.inc = DoubleConv(self.n_channels, 64)
        self.de_att1 = DeformableAttention2D(dim=64)
        self.down1 = Down(64, 128)
        self.de_att2 = DeformableAttention2D(dim=128)
        self.down2 = Down(128, 256)
        self.de_att3 = DeformableAttention2D(dim=256)
        self.down3 = Down(256, 512)
        self.de_att4 = DeformableAttention2D(dim=512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1_ = self.de_att1(x1)
        x2 = self.down1(x1)
        x2_ = self.de_att2(x2)
        x3 = self.down2(x2)
        x3_ = self.de_att3(x3)
        x4 = self.down3(x3)
        x4_ = self.de_att4(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4_)
        x = self.up2(x, x3_)
        x = self.up3(x, x2_)
        x = self.up4(x, x1_)
        logits = self.outc(x)
        return logits

class UNet_bot_deform_attention(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True):
        super(UNet_bot_deform_attention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.da = DeformableAttention2D(dim=512)
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
        x5 = self.da(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNet_bot_semantic_attention(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True):
        super(UNet_bot_semantic_attention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.sem_att = SemanticAttention(512, n_cls=n_classes)
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
        cls, x5 = self.sem_att(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, cls

class UNet_bot_ea(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True):
        super(UNet_bot_ea, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.ea = External_attention(c=512)
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
        x5 = self.ea(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits