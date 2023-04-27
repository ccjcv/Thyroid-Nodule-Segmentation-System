import argparse
from SmaAtUnet.unet_parts import *
from SmaAtUnet.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from UTNET.conv_trans_utils import *
from UTNET.unet_utils import up_block, down_block

class UNet_skip_can_att(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True,
                 base_chan = 64):
        super(UNet_skip_can_att, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_chan = base_chan

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, self.bilinear)
        # self.up2 = Up(512, 256 // factor, self.bilinear)
        # self.up3 = Up(256, 128 // factor, self.bilinear)
        # self.up4 = Up(128, 64, self.bilinear)

        self.up1 = up_block_trans(8 * base_chan, 8 * base_chan, num_block=0, bottleneck=False,
                                  heads=4, dim_head=8 * base_chan // 4, attn_drop=0.1,
                                  proj_drop=0.1, reduce_size=8, projection='interp', rel_pos=True)
        self.up2 = up_block_trans(8 * base_chan, 4 * base_chan, num_block=0, bottleneck=False, heads=4,
                                  dim_head=4 * base_chan // 4, attn_drop=0.1, proj_drop=0.1,
                                  reduce_size=8, projection='interp', rel_pos=True)
        self.up3 = up_block_trans(4 * base_chan, 2 * base_chan, num_block=0, bottleneck=False, heads=4,
                                  dim_head=2 * base_chan // 4, attn_drop=0.1, proj_drop=0.1,
                                  reduce_size=8, projection='interp', rel_pos=True)
        # self.up4 = up_block_trans(2 * base_chan, base_chan, num_block=0, bottleneck=False, heads=4,
        #                           dim_head=base_chan // 4, attn_drop=0.1, proj_drop=0.1,
        #                           reduce_size=8, projection='interp', rel_pos=True)
        self.up4 = up_block(2 * base_chan, base_chan, scale=(2, 2), num_block=2)
        #self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)#(8,512,16,16)
        #跳跃连接
        x = self.up1(x5, x4)#(8,512,32,32)
        x = self.up2(x, x3)#(8,256,64,64)
        x = self.up3(x, x2)#(8,128,128,128)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits