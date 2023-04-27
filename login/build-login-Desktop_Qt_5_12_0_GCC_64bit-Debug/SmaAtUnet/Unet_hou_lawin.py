import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import dropout, gelu
import numpy as np
import argparse
import math
from SmaAtUnet.unet_parts import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmcv.cnn import ConvModule, NonLocal2d
from mmseg.ops import resize
class MLP(nn.Module):
    """
    Linear Embedding: github.com/NVlabs/SegFormer
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class PatchEmbed(nn.Module):
    """
    Patch Embedding: github.com/SwinTransformer/
    """

    def __init__(self, proj_type='pool', patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj_type = proj_type
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if proj_type == 'conv':
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
                                  groups=patch_size * patch_size)
        elif proj_type == 'pool':
            self.proj = nn.ModuleList([nn.MaxPool2d(kernel_size=patch_size, stride=patch_size),
                                       nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)])
        else:
            raise NotImplementedError(f'{proj_type} is not currently supported.')

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        if self.proj_type == 'conv':
            x = self.proj(x)  # B C Wh Ww
        else:
            x = 0.5 * (self.proj[0](x) + self.proj[1](x))

        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x

class LawinAttn(NonLocal2d):
    def __init__(self, *arg, head=1,
                 patch_size=None, **kwargs):
        super().__init__(*arg, **kwargs)
        self.head = head
        self.patch_size = patch_size

        if self.head != 1:
            self.position_mixing = nn.ModuleList(
                [nn.Linear(patch_size * patch_size, patch_size * patch_size) for _ in range(self.head)])

    def forward(self, query, context):
        # x: [N, C, H, W]

        n = context.size(0)
        n, c, h, w = context.shape

        if self.head != 1:
            context = context.reshape(n, c, -1)
            context_mlp = []
            for hd in range(self.head):
                context_crt = context[:, (c // self.head) * (hd):(c // self.head) * (hd + 1), :]
                context_mlp.append(self.position_mixing[hd](context_crt))

            context_mlp = torch.cat(context_mlp, dim=1)
            context = context + context_mlp
            context = context.reshape(n, c, h, w)

        # g_x: [N, HxW, C]
        g_x = self.g(context).view(n, self.inter_channels, -1)
        g_x = rearrange(g_x, 'b (h dim) n -> (b h) dim n', h=self.head)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        if self.mode == 'gaussian':
            theta_x = query.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(context).view(n, self.in_channels, -1)
            else:
                phi_x = context.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(query).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(context).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(query).view(n, self.inter_channels, -1)
            theta_x = rearrange(theta_x, 'b (h dim) n -> (b h) dim n', h=self.head)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(context).view(n, self.inter_channels, -1)
            phi_x = rearrange(phi_x, 'b (h dim) n -> (b h) dim n', h=self.head)

        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        y = rearrange(y, '(b h) n dim -> b n (h dim)', h=self.head)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *query.size()[2:])

        output = query + self.conv_out(y)

        return output
class UNet_hou_lawin(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = True,
                 embed_dim=768, use_scale=True, reduction=2):
        super(UNet_hou_lawin, self).__init__()
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

        self.conv_cfg = dict(type='Conv2d')
        self.norm_cfg = dict(type='BN')
        self.act_cfg = dict(type='ReLU')
        self.in_channels = [64, 128, 320, 512]
        self.align_corners = False

        self.lawin_8 = LawinAttn(in_channels=512, reduction=reduction, use_scale=use_scale, conv_cfg=self.conv_cfg,
                                 norm_cfg=self.norm_cfg, mode='embedded_gaussian', head=64, patch_size=8)
        self.lawin_4 = LawinAttn(in_channels=512, reduction=reduction, use_scale=use_scale, conv_cfg=self.conv_cfg,
                                 norm_cfg=self.norm_cfg, mode='embedded_gaussian', head=16, patch_size=8)
        self.lawin_2 = LawinAttn(in_channels=512, reduction=reduction, use_scale=use_scale, conv_cfg=self.conv_cfg,
                                 norm_cfg=self.norm_cfg, mode='embedded_gaussian', head=4, patch_size=8)
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        ConvModule(512, 512, 1,
                                                   conv_cfg=self.conv_cfg,
                                                   norm_cfg=self.norm_cfg,
                                                   act_cfg=self.act_cfg))

        self.linear_c1 = MLP(input_dim=256, embed_dim=768)
        self.linear_c2 = MLP(input_dim=128, embed_dim=768)
        self.linear_c3 = MLP(input_dim=64, embed_dim=768)
        self.linear_fuse = ConvModule(
            in_channels=768 * 3,
            out_channels=512,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.ds_8 = PatchEmbed(proj_type='pool', patch_size=8, in_chans=512, embed_dim=512, norm_layer=nn.LayerNorm)
        self.ds_4 = PatchEmbed(proj_type='pool', patch_size=4, in_chans=512, embed_dim=512, norm_layer=nn.LayerNorm)
        self.ds_2 = PatchEmbed(proj_type='pool', patch_size=2, in_chans=512, embed_dim=512, norm_layer=nn.LayerNorm)


        self.outc = OutConv(64, self.n_classes)
    def get_context(self, x, patch_size):
        n, _, h, w = x.shape
        context = []
        for i, r in enumerate([8, 4, 2]):
            _context = F.unfold(x, kernel_size=patch_size * r, stride=patch_size, padding=int((r - 1) / 2 * patch_size))
            _context = rearrange(_context, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', ph=patch_size * r,
                                 pw=patch_size * r, nh=h // patch_size, nw=w // patch_size)
            context.append(getattr(self, f'ds_{r}')(_context))

        return context

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x1_ = self.up1(x5, x4)#(8,256,32,32)
        x2_ = self.up2(x1_, x3)#(8,128,64,64)
        x3_ = self.up3(x2_, x2)#(8,64,128,128)
        x4_ = self.up4(x3_, x1)#(8,32,256,256)

        n, _, h, w = x3_.shape

        _c1 = self.linear_c1(x1_).permute(0, 2, 1).reshape(n, -1, x1_.shape[2], x1_.shape[3])#(8,768,32,32)
        _c1 = resize(_c1, size=x3_.size()[2:], mode='bilinear', align_corners=False)  # (8,768,128,128))

        _c2 = self.linear_c2(x2_).permute(0, 2, 1).reshape(n, -1, x2_.shape[2], x2_.shape[3])  # (8,768,64,64)
        _c2 = resize(_c2, size=x3_.size()[2:], mode='bilinear', align_corners=False)  # (8,768,128,128)

        _c3 = self.linear_c3(x3_).permute(0, 2, 1).reshape(n, -1, x3_.shape[2], x3_.shape[3])  # (8,768,128,128)
        _c = self.linear_fuse(torch.cat([_c1, _c2, _c3], dim=1))  # (8,512,128,128)

        n, _, h, w = _c.shape

        ############### Lawin attention spatial pyramid pooling ###########
        patch_size = 8
        context = self.get_context(_c, patch_size)  # 3 list,(128,512,8,8)
        query = F.unfold(_c, kernel_size=patch_size, stride=patch_size)  # (8,32768,16)
        query = rearrange(query, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', ph=patch_size, pw=patch_size,
                          nh=h // patch_size, nw=w // patch_size)  # (128,512,8,8)

        output = []
        output.append(self.short_path(_c))  # (8,512,32,32)
        m = self.image_pool(_c)  # (8,512,1,1)
        output.append(resize(m,
                             size=(h, w),
                             mode='bilinear',
                             align_corners=self.align_corners))  # 2ge, (8,512,32,32)

        for i, r in enumerate([8, 4, 2]):
            _output = getattr(self, f'lawin_{r}')(query, context[i])
            _output = rearrange(_output, '(b nh nw) c ph pw -> b c (nh ph) (nw pw)', ph=patch_size, pw=patch_size,
                                nh=h // patch_size, nw=w // patch_size)
            output.append(_output)

        output = self.cat(torch.cat(output, dim=1))  # (8,512,32,32)
        logits = self.outc(x)
        return output