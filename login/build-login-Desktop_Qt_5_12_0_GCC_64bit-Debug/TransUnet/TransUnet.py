import torch
from TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import numpy as np

def get_transNet(num_classes):
    img_size = 256
    vit_patches_size = 16
    vit_name = 'R50-ViT-B_16'

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = num_classes
    config_vit.n_skip = 3
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    net = ViT_seg(config_vit, img_size=img_size, num_classes=num_classes)
    # net.load_from(weights=np.load(config_vit.pretrained_path))
    return net


if __name__ == '__main__':
    net = get_transNet(2)
    img = torch.randn((2, 3, 512, 512))
    segments = net(img)
    print(segments.size())
    # for edge in edges:
    #     print(edge.size())