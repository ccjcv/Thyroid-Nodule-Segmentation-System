import torch
import torch.nn as nn
class SemanticAttention(nn.Module):
    """ ClassMasking
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, dim, n_cls):

        super().__init__()
        self.dim = dim
        self.n_cls = n_cls
        self.softmax = nn.Softmax(dim=-1)

        self.mlp_cls_q = nn.Linear(self.dim, self.n_cls)
        self.mlp_cls_k = nn.Linear(self.dim, self.n_cls)

        self.mlp_v = nn.Linear(self.dim, self.dim)

        self.mlp_res = nn.Linear(self.dim, self.dim)

        self.proj_drop = nn.Dropout(0.1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.init_weight()

    def forward(self, x):
        """ Forward function.
        Args:
            x: input features with shape of (B, N, C)
        returns:
            class_seg_map: (B, N, K)
            gated feats: (B, N, C)
        """

        seg_map = self.mlp_cls_q(x)
        seg_ft = self.mlp_cls_k(x)

        feats = self.mlp_v(x)

        seg_score = seg_map @ seg_ft.transpose(-2, -1)
        seg_score = self.softmax(seg_score)

        feats = seg_score @ feats
        feats = self.mlp_res(feats)
        feats = self.proj_drop(feats)

        feat_map = self.gamma * feats + x

        return seg_map, feat_map

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Linear):
                nn.init.kaiming_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
            elif isinstance(ly, nn.LayerNorm):
                nn.init.constant_(ly.bias, 0)
                nn.init.constant_(ly.weight, 1.0)

        nn.init.zeros_(self.mlp_res.weight)
        if not self.mlp_res.bias is None: nn.init.constant_(self.mlp_res.bias, 0)