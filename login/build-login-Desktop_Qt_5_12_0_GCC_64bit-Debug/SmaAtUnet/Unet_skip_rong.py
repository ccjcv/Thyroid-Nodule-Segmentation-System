import argparse
from SmaAtUnet.unet_parts import *
from SmaAtUnet.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from SmaAtUnet.Unet_bot_aspp_att import Aspp_Attention
from einops import rearrange
import numpy as np
import math
from torch.nn.functional import dropout, gelu
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000
                         **(torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y,
                             device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                          dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2),
                          device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)
class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias, stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out
class Aspp_Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 dropout_p: float = 0.0, sr_ratio: int = 2):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f'expected dim {dim} to be a multiple of num_heads {num_heads}.')

        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout_p = dropout_p

        self.pe = PositionalEncodingPermute2D(dim)

        self.sr_ratio = sr_ratio
        #segformer中的和PVT v1中的mhsa
        # if sr_ratio > 1:
        #     sr_ratio_tuple = (sr_ratio, sr_ratio)
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio_tuple, stride=sr_ratio_tuple)
        #     self.norm = nn.LayerNorm(dim)

        #PVT v2中的mhsa
        self.pool_1 = nn.AdaptiveAvgPool2d(8)
        self.pool_2 = nn.AdaptiveAvgPool2d(4)
        self.pool_3 = nn.AdaptiveAvgPool2d(2)
        self.pool_4 = nn.AdaptiveAvgPool2d(1)
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.dsc = depthwise_separable_conv(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        b, c, h, w = x.size()
        pe = self.pe(x)
        x = x + pe#(8,512,16,16),(b,c,h,w)
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # [b, h*w, d]
        q = self.q(x)
        q = rearrange(q, ('b hw (m c) -> b m hw c'), m=self.num_heads)

        #PVT v1中的
        # if self.sr_ratio > 1:
        #     x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)#(8,512,16,16)
        #     x = self.sr(x)#(8,512,8,8)
        #     x = rearrange(x, 'b c h w -> b (h w) c')#(8,64,512)
        #     x = self.norm(x)

        #PVT v2中的
        x_ = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # (8,512,16,16)
        #进行多尺度池化
        x_1 = self.pool_1(x_)#(8,512,8,8)
        x_1 = self.dsc(x_1)
        x_1 = rearrange(x_1, 'b c h w -> b (h w) c')
        x_2 = self.pool_2(x_)#(8,512,4,4)
        x_2 = self.dsc(x_2)
        x_2 = rearrange(x_2, 'b c h w -> b (h w) c')
        x_3 = self.pool_3(x_)#(8,512,2,2)
        x_3 = self.dsc(x_3)
        x_3 = rearrange(x_3, 'b c h w -> b (h w) c')
        x_4 = self.pool_4(x_)#(8,512,1,1)
        x_4 = self.dsc(x_4)
        x_4 = rearrange(x_4, 'b c h w -> b (h w) c')
        x = torch.cat((x_1, x_2, x_3, x_4), dim = 1)#(8,85,512)
        # x = self.sr(x)  # (8,512,8,8)
        # x = rearrange(x, 'b c h w -> b (h w) c')  # (8,64,512)
        x = self.norm(x)
        x = self.act(x)

        x = self.kv(x)#(8,85,1024)
        x = rearrange(x, 'b d (a m c) -> a b m d c', a=2, m=self.num_heads)#(2,8,8,85,64)
        k, v = x.unbind(0)#(8,8,85,64)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = rearrange(x, 'b m hw c -> b hw (m c)')
        x = self.proj(x)
        x = dropout(x, p=self.dropout_p, training=self.training)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x
class tiao_Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 dropout_p: float = 0.0, sr_ratio: int = 2):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f'expected dim {dim} to be a multiple of num_heads {num_heads}.')

        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout_p = dropout_p

        self.pe = PositionalEncodingPermute2D(dim)

        self.sr_ratio = sr_ratio
        #segformer中的和PVT v1中的mhsa
        # if sr_ratio > 1:
        #     sr_ratio_tuple = (sr_ratio, sr_ratio)
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio_tuple, stride=sr_ratio_tuple)
        #     self.norm = nn.LayerNorm(dim)

        #PVT v2中的mhsa
        # self.pool_1 = nn.AdaptiveAvgPool2d(8)
        # self.pool_2 = nn.AdaptiveAvgPool2d(4)
        # self.pool_3 = nn.AdaptiveAvgPool2d(2)
        # self.pool_4 = nn.AdaptiveAvgPool2d(1)
        self.pool_shu = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_heng = nn.AdaptiveAvgPool2d((1, None))

        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        b, c, h, w = x.size()
        pe = self.pe(x)
        x = x + pe#(8,512,16,16),(b,c,h,w)
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # [b, h*w, d]
        q = self.q(x)
        q = rearrange(q, ('b hw (m c) -> b m hw c'), m=self.num_heads)

        #PVT v1中的
        # if self.sr_ratio > 1:
        #     x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)#(8,512,16,16)
        #     x = self.sr(x)#(8,512,8,8)
        #     x = rearrange(x, 'b c h w -> b (h w) c')#(8,64,512)
        #     x = self.norm(x)

        #PVT v2中的
        x_ = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # (8,512,16,16)
        #进行多尺度池化
        x_shu = self.pool_shu(x_)#(8,512,16,1)
        x_shu = self.sr(x_shu)#(8,512,16,1)
        x_shu = rearrange(x_shu, 'b c h w -> b (h w) c')
        x_heng = self.pool_heng(x_)#(8,512,1,16)
        x_heng = self.sr(x_heng)
        x_heng = rearrange(x_heng, 'b c h w -> b (h w) c')
        # x_1 = self.pool_1(x_)#(8,512,8,8)
        # x_1 = self.sr(x_1)
        # x_1 = x_1.view(b, 64, c)
        # x_2 = self.pool_2(x_)#(8,512,4,4)
        # x_2 = self.sr(x_2)
        # x_2 = x_2.view(b, 16, c)
        # x_3 = self.pool_3(x_)#(8,512,2,2)
        # x_3 = self.sr(x_3)
        # x_3 = x_3.view(b, 4, c)
        # x_4 = self.pool_4(x_)#(8,512,1,1)
        # x_4 = self.sr(x_4)
        # x_4 = x_4.view(b, 1, c)
        x = torch.cat((x_shu, x_heng), dim=1)#(8,85,512)
        # x = self.sr(x)  # (8,512,8,8)
        # x = rearrange(x, 'b c h w -> b (h w) c')  # (8,64,512)
        x = self.norm(x)
        x = self.act(x)

        x = self.kv(x)
        x = rearrange(x, 'b d (a m c) -> a b m d c', a=2, m=self.num_heads)
        k, v = x.unbind(0)#(8,8,85,64)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = rearrange(x, 'b m hw c -> b hw (m c)')
        x = self.proj(x)
        x = dropout(x, p=self.dropout_p, training=self.training)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x

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
class fang_chihua_module(nn.Module):
    def __init__(self, dim):
        super(fang_chihua_module, self).__init__()
        self.SlayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.fang_Attention = Aspp_Attention(dim)
    def forward(self, x):
        m = x  # (8,512,16,16), (B, N, H)
        b, c, h, w = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')#(8,16*16,512)
        x = self.SlayerNorm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        x_fang = self.fang_Attention(x)
        return x_fang
class tiao_chihua_module(nn.Module):
    def __init__(self, dim):
        super(tiao_chihua_module, self).__init__()
        self.SlayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.tiao_Attention = tiao_Attention(dim)


    def forward(self, x):
        m = x  # (8,512,16,16), (B, N, H)
        b, c, h, w = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')#(8,16*16,512)
        x = self.SlayerNorm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        x_tiao = self.tiao_Attention(x)

        return x_tiao
class UNet_skip_rong(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = False):
        super(UNet_skip_rong, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.a1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.a2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.a3 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.a4 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1_ = Up_(128, 64, self.bilinear)
        self.up2_ = Up_(256, 128 // factor, self.bilinear)
        self.up3_ = Up_(512, 256 // factor, self.bilinear)
        self.up4_ = Up_(1024, 512 // factor, self.bilinear)

        self.fang_chihua_4 = fang_chihua_module(dim=512)
        self.tiao_chihua_4 = tiao_chihua_module(dim=512)
        self.fang_chihua_3 = fang_chihua_module(dim=256)
        self.tiao_chihua_3 = tiao_chihua_module(dim=256)


        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)#(8,128,128,128)
        x3 = self.down2(x2)#(8,256,64,64)
        x4 = self.down3(x3)#(8,512,32,32)
        x5 = self.down4(x4)

        x2_ = self.up1_(x2)#(8,64,256,256)
        x3_ = self.up2_(x3)#(8,128,128,128)
        x4_ = self.up3_(x4)#(8,256,64,64)
        x5_ = self.up4_(x5)

        x5_ = self.fang_chihua_4(x5_)
        x4_chi = self.tiao_chihua_4(x4)
        x4 = x4 + self.a1 * x4_chi + self.a2 * x5_
        x = self.up1(x5, x4)
        x4_ = self.fang_chihua_3(x4_)
        x3_chi = self.tiao_chihua_3(x3)
        x3 = x3 + self.a3 * x3_chi + self.a4 * x4_
        x = self.up2(x, x3)
        x2 = x2 + x3_
        x = self.up3(x, x2)
        x1 = x1 + x2_
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits