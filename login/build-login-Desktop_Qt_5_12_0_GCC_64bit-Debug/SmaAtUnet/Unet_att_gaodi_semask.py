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
# from UTNET.conv_trans_utils import *
# from UTNET.unet_utils import up_block, down_block
from SmaAtUnet.Unet_mhsa import MultiHeadSelfAttention
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
class EfficientAttention(nn.Module):
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
        self.pool = nn.AdaptiveAvgPool2d(8)
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
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # (8,512,16,16)
        x = self.pool(x)
        x = self.sr(x)  # (8,512,8,8)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (8,64,512)
        x = self.norm(x)
        x = self.act(x)

        x = self.kv(x)
        x = rearrange(x, 'b d (a m c) -> a b m d c', a=2, m=self.num_heads)
        k, v = x.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = rearrange(x, 'b m hw c -> b hw (m c)')
        x = self.proj(x)
        x = dropout(x, p=self.dropout_p, training=self.training)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x
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
        canshu = self.weight.repeat(b, 1, 1)
        x = torch.bmm(x, canshu)
        # x = F.linear(x, self.weight, self.bias)
        return x
class Semask_Attention(nn.Module):
    def __init__(self, num_classes=2, channel=512):
        super(Semask_Attention, self).__init__()
        #self.query = MultiHeadDense(num_classes, bias=False)
        self.query = nn.Linear(channel, num_classes)
        #self.key = MultiHeadDense(num_classes, bias=False)
        self.key = nn.Linear(channel, num_classes)
        self.value = MultiHeadDense(channel, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        # pe = self.positional_encoding_2d(c, h, w)
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # [b, h*w, d],[b, 256, 512]
        #将C变成num-classes
        Q = self.query(x)
        K = self.key(x)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)))  # [b, h*w, h*w]
        V = self.value(x)
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(b, c, h, w)
        Q_out = Q.permute(0, 2, 1).reshape(b, 2, h, w)
        return x, Q_out
class att_can(nn.Module):
    def __init__(self, in_ch):
        super(att_can, self).__init__()
        self.bn_l = nn.BatchNorm2d(in_ch)

        self.eff_att = EfficientAttention(dim=in_ch)

        self.bn2 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
    def forward(self, x):
        residue = x
        x1 = self.bn_l(x)
        eff_att = self.eff_att(x1)

        out = eff_att + residue
        residue = out

        out = self.bn2(out)
        out = self.relu(out)
        out = self.mlp(out)

        out += residue
        return out

class UNet_att_gaodi_semask(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True,
                 base_chan = 64):
        super(UNet_att_gaodi_semask, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_chan = base_chan

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.att_can_1 = att_can(128)
        self.down2 = Down(128, 256)
        self.att_can_2 = att_can(256)
        self.down3 = Down(256, 512)
        self.att_can_3 = att_can(512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.m_att = MultiHeadSelfAttention(512)
        self.eff_att = EfficientAttention(512)
        self.a1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.a2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.semask = Semask_Attention()
        self.up_semask = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)
        # self.up1 = up_block_trans(8 * base_chan, 8 * base_chan, num_block=0, bottleneck=False,
        #                           heads=8, dim_head=8 * base_chan // 8, attn_drop=0.1,
        #                           proj_drop=0.1, reduce_size=8, projection='interp', rel_pos=True)
        # self.up2 = up_block_trans(8 * base_chan, 4 * base_chan, num_block=0, bottleneck=False, heads=8,
        #                           dim_head=4 * base_chan // 8, attn_drop=0.1, proj_drop=0.1,
        #                           reduce_size=8, projection='interp', rel_pos=True)
        # self.up3 = up_block_trans(4 * base_chan, 2 * base_chan, num_block=0, bottleneck=False, heads=8,
        #                           dim_head=2 * base_chan // 8, attn_drop=0.1, proj_drop=0.1,
        #                           reduce_size=8, projection='interp', rel_pos=True)
        # self.up4 = up_block(2 * base_chan, base_chan, scale=(2, 2), num_block=2)


        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.att_can_1(x2)
        x3 = self.down2(x2)
        x3 = self.att_can_2(x3)
        x4 = self.down3(x3)
        x4 = self.att_can_3(x4)
        x5 = self.down4(x4)
        #x5进行self-attention
        x5_m_att = self.m_att(x5)
        x5_eff_att = self.eff_att(x5)
        x5_three = x5 + self.a1 * x5_m_att + self.a2 * x5_eff_att
        x5_semash, Q_out = self.semask(x5_three)
        Q_out = self.up_semask(Q_out)
        x = self.up1(x5_semash, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, Q_out