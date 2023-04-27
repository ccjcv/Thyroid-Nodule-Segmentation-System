import argparse
from SmaAtUnet.unet_parts import *
from SmaAtUnet.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from SmaAtUnet.layers import CBAM
from SmaAtUnet.CoordAttention import CoordAtt
# from Seg_EA.zhuyili import External_attention


class UNet(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
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


class UNet_Attention(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True,
                 reduction_ratio: int = 16):
        super(UNet_Attention, self).__init__()
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
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits


class UNetDS(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True,
                 kernels_per_layer: int = 1):
        super(UNetDS, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        kernels_per_layer = kernels_per_layer

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

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


class UNetDS_Attention(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True,
                 reduction_ratio: int = 16,
                 kernels_per_layer = 1):
        super(UNetDS_Attention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio
        kernels_per_layer = kernels_per_layer

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits


class UNetDS_Attention_4CBAMs(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True,
                 reduction_ratio: int = 16,
                 kernels_per_layer: int = 1):
        super(UNetDS_Attention_4CBAMs, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio
        kernels_per_layer = kernels_per_layer

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits


class UNet_CA(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True,
                 reduction_ratio: int = 16):
        super(UNet_CA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.inc = DoubleConv(self.n_channels, 64)
        #self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.ca1 = CoordAtt(64, 64, reduction=reduction_ratio)
        self.down1 = Down(64, 128)
        #self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.ca2 = CoordAtt(128, 128, reduction=reduction_ratio)
        self.down2 = Down(128, 256)
        #self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.ca3 = CoordAtt(256, 256, reduction=reduction_ratio)
        self.down3 = Down(256, 512)
        #self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        self.ca4 = CoordAtt(512, 512, reduction=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        #self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.ca5 = CoordAtt(512, 512, reduction=reduction_ratio)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        #x1Att = self.cbam1(x1)
        x1Att = self.ca1(x1)
        x2 = self.down1(x1)
        #x2Att = self.cbam2(x2)
        x2Att = self.ca2(x2)
        x3 = self.down2(x2)
        #x3Att = self.cbam3(x3)
        x3Att = self.ca3(x3)
        x4 = self.down3(x3)
        #x4Att = self.cbam4(x4)
        x4Att = self.ca4(x4)
        x5 = self.down4(x4)
        #x5Att = self.cbam5(x5)
        x5Att = self.ca5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits
