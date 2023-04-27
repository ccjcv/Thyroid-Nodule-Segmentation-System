import argparse
from SmaAtUnet.unet_parts import *
from SmaAtUnet.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from SmaAtUnet.layers import CBAM
from SmaAtUnet.CoordAttention import CoordAtt
from SmaAtUnet.external_attention_2 import External_attention
from SmaAtUnet.deform_attention_2 import DeformableAttention2D
from SmaAtUnet.semantic_attention import SemanticAttention
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import math
from torch.nn.functional import dropout, gelu
# from SoftPool import soft_pool2d, SoftPool2dS
from SmaAtUnet.deform_conv import DeformConv2d
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
        # self.soft_pool = SoftPool2d(kernel_size=(8,8), stride=(8, 8))
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
        avg_pool = F.avg_pool2d(x, (8, 8), stride=(8, 8))
        max_pool = F.max_pool2d(x, (8, 8), stride=(8, 8))
        # soft_pool = self.soft_pool(x.contiguous())
        #x = self.pool(x)
        x = avg_pool + max_pool
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
class MEAttention(nn.Module):
    def __init__(self, dim):
        super(MEAttention, self).__init__()
        self.num_heads = 8
        self.coef = 4
        self.query_liner = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.coef * self.num_heads
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

        self.proj = nn.Linear(dim * self.coef, dim)

    def forward(self, x):#(b,c,h,w)
        x = rearrange(x, 'b c h w -> b (h w) c')
        B, N, C = x.shape
        x = self.query_liner(x)
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1,
                                                     3)  #(1, 32, 225, 32)

        attn = self.linear_0(x)

        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))

        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B, N, -1)

        x = self.proj(x)#(24,256,512)
        x = rearrange(x, 'b (h w) c -> b c h w', h=16, w=16)

        return x

class Def_EAmodule(nn.Module):
    def __init__(self, dim=1024):
        super(Def_EAmodule, self).__init__()
        self.SlayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.ElayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.dcn_bot = DeformConv2d(inc=512, outc=512, kernel_size=3, padding=0, stride=1, modulation=True)
        # self.Def_Attention = DeformableAttention2D(dim=512)
        self.eff_att = EfficientAttention(512)
        self.E_Attention = MEAttention(dim)

    def forward(self, x):
        m = x  # (8,512,16,16), (B, N, H)
        b, c, h, w = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')#(8,16*16,512)
        x = self.SlayerNorm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=16, w=16)#(8,1024,16,16)
        x_gao = x[:,:512,:,:]
        x_di = x[:,512:,:,:]
        # x_win = x_gao.view(b, 512, 2, 8, 2, 8)
        # x_win = x_win.permute(2, 4, 0, 1, 3, 5)  # (2,2,8,512,8,8)
        # list = []
        # for i in range(2):
        #     for j in range(2):
        #         z = self.Def_Attention(x_win[i][j])
        #         list.append(z)
        # result_1 = list  # 四个tensor,(8,64,512)
        # result = []
        # for i in range(4):
        #     result.append(result_1[i])  # 4个tensor,(8,512,8,8)
        # shang_liangfutu = torch.cat((result[0], result[1]), dim=3)  # (8,512,8,16)
        # xia_liangfutu = torch.cat((result[2], result[3]), dim=3)  # (8,512,8,16)
        # si_fu_tu = torch.cat((shang_liangfutu, xia_liangfutu), dim=2)  # (8,512,16,16)

        x_dcn = self.dcn_bot(x_gao)
        x_gao_he = x_dcn

        x_eff = self.eff_att(x_di)

        # x = self.Def_Attention(x)
        # x = m + x
        x = torch.cat((x_gao_he, x_eff), dim = 1)
        x = x + m

        m = x
        x = rearrange(x, 'b c h w -> b (h w) c')  # (8,16*16,512)
        x = self.ElayerNorm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=16, w=16)

        x = self.E_Attention(x)
        x = m + x

        return x
class UNet_bot_def_ea(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = False):
        super(UNet_bot_def_ea, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.def_ea = Def_EAmodule(1024)
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
        #x5进行self-attention
        x5_def_ea = self.def_ea(x5)
        x = self.up1(x5_def_ea, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits