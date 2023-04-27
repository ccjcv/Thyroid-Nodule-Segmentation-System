import argparse
from SmaAtUnet.unet_parts import *
from SoftPool import soft_pool2d, SoftPool2d
from SmaAtUnet.deform_conv import DeformConv2d
from SmaAtUnet.deform_attention_2 import DeformableAttention2D
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelAtt(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=2, pool_types=['avg', 'max','soft']):
        super(ChannelAtt, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU()
            #nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
        self.incr = nn.Linear(gate_channels // reduction_ratio, gate_channels)

    def forward(self, x):
        channel_att_sum = None
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        avgpoolmlp = self.mlp(avg_pool)
        maxpoolmlp=self.mlp(max_pool)
        pooladd = avgpoolmlp+maxpoolmlp

        self.pool = SoftPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        soft_pool = self.mlp(self.pool(x))
        weightPool = soft_pool * pooladd
        # channel_att_sum = self.mlp(weightPool)
        channel_att_sum = self.incr(weightPool)
        Att = torch.sigmoid(channel_att_sum)
        Att = Att.unsqueeze(2)
        Att = Att.unsqueeze(3)
        Att = Att.expand_as(x)
        return Att
class Up_(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        # if bilinear:
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        # else:
        #     self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        #     self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
class UNet_bot_tong_def_conv(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = False):
        super(UNet_bot_tong_def_conv, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.dcn_4 = DeformConv2d(inc=512, outc=512, kernel_size=3, padding=0, stride=1,modulation=True)
        #self.def_att_4 = DeformableAttention2D(dim=1024)
        # bn=nn.BatchNorm2d(outchannel[i_layer])
        self.channel_att_4 = ChannelAtt(gate_channels=512, reduction_ratio=2, pool_types=['avg', 'max'])

        self.up1_ = Up_(128, 64, self.bilinear)
        self.up2_ = Up_(256, 128 // factor, self.bilinear)
        self.up3_ = Up_(512, 256 // factor, self.bilinear)
        self.up4_ = Up_(1024, 512 // factor, self.bilinear)

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

        x2_ = self.up1_(x2)  # (8,64,256,256)
        x3_ = self.up2_(x3)  # (8,128,128,128)
        x4_ = self.up3_(x4)  # (8,256,64,64)
        x5_ = self.up4_(x5)

        x4_channel = self.channel_att_4(x4)
        x5_def = self.dcn_4(x5_)

        x4_cheng = torch.mul(x4_channel, x5_def)
        x4_jia = x4_cheng + x4_channel + x4
        x = self.up1(x5, x4_jia)
        # x4_ = self.fang_chihua_3(x4_)
        # x3_chi = self.tiao_chihua_3(x3)
        # x3 = x3 + self.a1 * x3_chi + self.a2 * x4_
        x3 = x3 + x4_
        x = self.up2(x, x3)
        x2 = x2 + x3_
        x = self.up3(x, x2)
        x1 = x1 + x2_
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits