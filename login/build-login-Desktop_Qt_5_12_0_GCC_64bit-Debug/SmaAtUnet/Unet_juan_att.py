import argparse
import math
import numpy as np
from SmaAtUnet.unet_parts import *
from einops import rearrange
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
class bihui1(nn.Module):
    def __init__(self, dim=512, num_heads=8, qkv_bias: bool = False,
                 dropout_p: float = 0.0, sr_ratio: int = 2):
        super(bihui1, self).__init__()
        self.pe = PositionalEncodingPermute2D(dim)
        self.pool_8 = nn.AdaptiveAvgPool2d((8,8))
        self.conv = nn.Conv2d(512, 512, kernel_size=4, padding=1, stride=2)

        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout_p = dropout_p
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        b, c, h, w = x.size()
        pe = self.pe(x)
        x = x + pe  # (8,512,16,16),(b,c,h,w)

        x_conv = self.conv(x)#(8,512,8,8)
        x_pool_8 = self.pool_8(x)#(8,512,8,8)

        x_conv = rearrange(x_conv, 'b c h w -> b (h w) c')#(8,64,512)

        x_zi = x.view(b, 512, 2, 8, 2, 8)
        x_zi = x_zi.permute(2, 4, 0, 1, 3, 5)#(2,2,8,512,8,8)

        a1 = x_zi[0][0]#(8,512,8,8)
        a2 = x_zi[0][1]
        a3 = x_zi[1][0]
        a4 = x_zi[1][1]
        list = []
        for i in range(2):
            for j in range(2):
                b, c, h, w = x_zi[i][j].size()
                a = x_zi[i][j].reshape(b, c, h * w).permute(0, 2, 1)  # [b, h*w, d]
                q = self.q(a)  # (8,64,512)
                # q = rearrange(q, ('b hw (m c) -> b m hw c'), m=self.num_heads)

                # PVT v2中的
                a = rearrange(a, 'b (h w) c -> b c h w', h=h, w=w)  # (8,512,8,8)
                # a1 = self.pool(a1)
                a = self.sr(a)  # (8,512,8,8)
                a = rearrange(a, 'b c h w -> b (h w) c')  # (8,64,512)
                a = self.norm(a)
                a = self.act(a)

                a = self.kv(a)  # (8,64,1024)
                # a1 = rearrange(a1, 'b d (a m c) -> a b m d c', a=2, m=self.num_heads)#(2,8,8,16,64)
                a = rearrange(a, 'b d (a c) -> a b d c', a=2)  # (2,8,64,512)
                k, v = a.unbind(0)  # (8,64,512)

                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)  # (8,64,64)
                # 本来是和v乘
                # x = attn @ v#(8,64,512)
                x = attn @ x_conv
                list.append(x)
        result_1 = list#四个tensor,(8,64,512)
        result = []
        for i in range(4):
            result_2 = rearrange(result_1[i], 'b (h w) c -> b c h w', h=8, w=8)
            result.append(result_2)#4个tensor,(8,512,8,8)
        shang_liangfutu = torch.cat((result[0], result[1]), dim = 3)#(8,512,8,16)
        xia_liangfutu = torch.cat((result[2], result[3]), dim = 3)#(8,512,8,16)
        si_fu_tu = torch.cat((shang_liangfutu, xia_liangfutu), dim = 2)#(8,512,16,16)
        return si_fu_tu
class UNet_bot_juan_att(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True):
        super(UNet_bot_juan_att, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.bihui1 = bihui1(512)
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
        x5_bihui1 = self.bihui1(x5)#(8,512,16,16)
        x = self.up1(x5_bihui1, x4)#(8,256,32,32)
        x = self.up2(x, x3)#(8,128,64,64)
        x = self.up3(x, x2)#(8,64,128,128)
        x = self.up4(x, x1)#(8,64,256,256)
        logits = self.outc(x)
        return logits