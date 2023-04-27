import argparse
from SmaAtUnet.unet_parts import *
from einops import rearrange
from SmaAtUnet.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from SoftPool import soft_pool2d, SoftPool2d
import adapool_cuda
from adaPool import AdaPool2d
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
        self.soft_pool = SoftPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        avgpoolmlp = self.mlp(avg_pool)
        maxpoolmlp=self.mlp(max_pool)
        softpoolmlp = self.mlp(self.soft_pool(x))
        pooladd = avgpoolmlp+maxpoolmlp+softpoolmlp

        #self.pool = SoftPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # self.pool = AdaPool2d(kernel_size=(x.size(2), x.size(3)), beta=(1, 1), stride=(x.size(2), x.size(3)),
        #                       device='cuda')
        # ada_pool = self.mlp(self.pool(x))
        # weightPool = ada_pool * pooladd
        weightPool = pooladd
        # channel_att_sum = self.mlp(weightPool)
        channel_att_sum = self.incr(weightPool)
        Att = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return Att
class Up_(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        # if bilinear:
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=1)
        # else:
        #     self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        #     self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
class UNet(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = False):
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

        self.up1_ = Up_(128, 64)
        self.up2_ = Up_(256, 128)
        self.up3_ = Up_(512, 256)
        self.up4_ = Up_(1024, 512)

        #self.dcn_5 = DeformConv2d(inc=1024, outc=1024, kernel_size=3, padding=0, stride=1, modulation=True)
        self.def_att_5 = DeformableAttention2D(dim=1024)
        # bn=nn.BatchNorm2d(outchannel[i_layer])
        self.channel_att_4 = ChannelAtt(gate_channels=512, reduction_ratio=2, pool_types=['avg', 'max'])

        self.dcn_4 = DeformConv2d(inc=512, outc=512, kernel_size=3, padding=0, stride=1, modulation=True)
        #self.def_att_4 = DeformableAttention2D(dim=512)
        # bn=nn.BatchNorm2d(outchannel[i_layer])
        self.channel_att_3 = ChannelAtt(gate_channels=256, reduction_ratio=2, pool_types=['avg', 'max'])

        self.weight_level_256_4 = nn.Conv2d(256, 4, 1)
        self.weight_level_512_4 = nn.Conv2d(512, 4, 1)
        self.weight_level_128_4 = nn.Conv2d(128, 4, 1)
        self.weight_level_64_4 = nn.Conv2d(64, 4, 1)
        self.weight_level_8_2 = nn.Conv2d(8, 2, 1)

        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)

        x2_1 = self.up1_(x2)
        weight_1_channel = self.weight_level_64_4(x1)
        weight_1_cheng = self.weight_level_64_4(x2_1)
        weight_1_jia = torch.cat((weight_1_channel, weight_1_cheng), 1)
        level_weight_1_jia = self.weight_level_8_2(weight_1_jia)
        level_weight_1_jia = F.softmax(level_weight_1_jia, dim=1)
        fused_x1_jia = x1 * level_weight_1_jia[:, 0:1, :, :] + \
                       x2_1 * level_weight_1_jia[:, 1:2, :, :]

        x3 = self.down2(x2)
        x3_2 = self.up2_(x3)
        weight_2_channel = self.weight_level_128_4(x2)
        weight_2_cheng = self.weight_level_128_4(x3_2)
        weight_2_jia = torch.cat((weight_2_channel, weight_2_cheng), 1)
        level_weight_2_jia = self.weight_level_8_2(weight_2_jia)
        level_weight_2_jia = F.softmax(level_weight_2_jia, dim=1)
        fused_x2_jia = x2 * level_weight_2_jia[:, 0:1, :, :] + \
                       x3_2 * level_weight_2_jia[:, 1:2, :, :]


        x4 = self.down3(x3)

        x3_channel = self.channel_att_3(x3)#1#(256,64,64)
        #x4_def = self.def_att_4(x4)
        x4_def = self.dcn_4(x4)
        x4_def_3 = self.up3_(x4_def)
        x3_cheng = torch.mul(x3_channel, x4_def_3)#2
        weight_3_channel = self.weight_level_256_4(x3_channel)
        weight_3_cheng = self.weight_level_256_4(x3_cheng)
        weight_3_jia = torch.cat((weight_3_channel, weight_3_cheng), 1)
        level_weight_3_jia = self.weight_level_8_2(weight_3_jia)
        level_weight_3_jia = F.softmax(level_weight_3_jia, dim=1)
        fused_x3_jia = x3_channel * level_weight_3_jia[:, 0:1, :, :] + \
                               x3_cheng * level_weight_3_jia[:, 1:2, :, :] + x3


        # x4_3 = self.up3_(x4)
        # weight_3_channel = self.weight_level_256_4(x3)
        # weight_3_cheng = self.weight_level_256_4(x4_3)
        # weight_3_jia = torch.cat((weight_3_channel, weight_3_cheng), 1)
        # level_weight_3_jia = self.weight_level_8_2(weight_3_jia)
        # level_weight_3_jia = F.softmax(level_weight_3_jia, dim=1)
        # fused_x3_jia = x3 * level_weight_3_jia[:, 0:1, :, :] + \
        #                x4_3 * level_weight_3_jia[:, 1:2, :, :]

        x5 = self.down4(x4)

        x4_channel = self.channel_att_4(x4)  # 1#(512,32,32)
        x5_def = self.def_att_5(x5)
        x5_def_4 = self.up4_(x5_def)
        x4_cheng = torch.mul(x4_channel, x5_def_4)  # 2
        weight_4_channel = self.weight_level_512_4(x4_channel)
        weight_4_cheng = self.weight_level_512_4(x4_cheng)
        weight_4_jia = torch.cat((weight_4_channel, weight_4_cheng), 1)
        level_weight_4_jia = self.weight_level_8_2(weight_4_jia)
        level_weight_4_jia = F.softmax(level_weight_4_jia, dim=1)
        fused_x4_jia = x4_channel * level_weight_4_jia[:, 0:1, :, :] + \
                       x4_cheng * level_weight_4_jia[:, 1:2, :, :] + x4



        x = self.up1(x5, fused_x4_jia)
        x = self.up2(x, fused_x3_jia)
        x = self.up3(x, fused_x2_jia)
        x = self.up4(x, fused_x1_jia)
        logits = self.outc(x)
        return logits