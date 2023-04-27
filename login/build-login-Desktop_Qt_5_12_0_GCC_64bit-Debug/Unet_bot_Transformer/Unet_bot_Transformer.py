import argparse
from SmaAtUnet.unet_parts import *
from einops import rearrange
from AA_TransUnet.encoder import ViT
from AA_TransUnet.decoder import SingleConv, UpDS
class UNet(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 2,
                 bilinear: bool = False,
                 patch_size=2,
                 vit_transformer_dim=1024,
                 img_dim=256,
                 vit_blocks=1,
                 vit_heads=8,
                 vit_dim_linear_mhsa_block=3072,
                 vit_transformer=None,
                 vit_channels=None
                 ):
        super(UNet, self).__init__()

        self.inplanes = 128
        self.patch_size = patch_size
        self.vit_transformer_dim = vit_transformer_dim
        vit_channels = self.inplanes * 8 if vit_channels is None else vit_channels
        self.img_dim_vit = img_dim // 16
        assert (self.img_dim_vit % patch_size == 0), "Vit patch_dim not divisible"

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.vit = ViT(img_dim=self.img_dim_vit,
                       in_channels=vit_channels,  # input features' channels (encoder)
                       patch_dim=patch_size,
                       # transformer inside dimension that input features will be projected
                       # out will be [batch, dim_out_vit_tokens, dim ]
                       dim=vit_transformer_dim,
                       blocks=vit_blocks,
                       heads=vit_heads,
                       dim_linear_block=vit_dim_linear_mhsa_block,
                       classification=False) if vit_transformer is None else vit_transformer

        # to project patches back - undoes vit's patchification
        token_dim = vit_channels * (patch_size ** 2)
        self.project_patches_back = nn.Linear(vit_transformer_dim, token_dim)
        # upsampling path
        self.vit_conv = SingleConv(in_ch=vit_channels, out_ch=1024)

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

        y = self.vit(x5)  # out shape of number_of_patches, vit_transformer_dim

        # from [number_of_patches, vit_transformer_dim] -> [number_of_patches, token_dim]
        y = self.project_patches_back(y)

        # from [batch, number_of_patches, token_dim] -> [batch, channels, img_dim_vit, img_dim_vit]
        y = rearrange(y, 'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',
                      x=self.img_dim_vit // self.patch_size, y=self.img_dim_vit // self.patch_size,
                      patch_x=self.patch_size, patch_y=self.patch_size)

        y = self.vit_conv(y)

        x = self.up1(y, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits