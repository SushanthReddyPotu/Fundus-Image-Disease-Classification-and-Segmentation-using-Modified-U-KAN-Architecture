import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from kan import KANLinear


# =========================
# KAN Layer
# =========================
class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        grid_size = 5
        spline_order = 3

        if not no_kan:
            self.fc1 = KANLinear(in_features, hidden_features,
                                 grid_size=grid_size, spline_order=spline_order)
            self.fc2 = KANLinear(hidden_features, out_features,
                                 grid_size=grid_size, spline_order=spline_order)
            self.fc3 = KANLinear(hidden_features, out_features,
                                 grid_size=grid_size, spline_order=spline_order)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, H, W):
        B, N, C = x.shape

        x = self.fc1(x.reshape(B * N, C)).reshape(B, N, C)
        x = self.dwconv_1(x, H, W)

        x = self.fc2(x.reshape(B * N, C)).reshape(B, N, C)
        x = self.dwconv_2(x, H, W)

        x = self.fc3(x.reshape(B * N, C)).reshape(B, N, C)
        x = self.dwconv_3(x, H, W)

        return x


# =========================
# KAN Block
# =========================
class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm2 = norm_layer(dim)
        self.layer = KANLayer(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, H, W):
        return x + self.drop_path(self.layer(self.norm2(x), H, W))


# =========================
# Depthwise Conv Blocks
# =========================
class DW_bn_relu(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.relu(self.bn(self.dwconv(x)))
        return x.flatten(2).transpose(1, 2)


# =========================
# Patch Embedding
# =========================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=3,
                 stride=2, in_chans=3, embed_dim=768):
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# =========================
# Basic Conv Blocks
# =========================
class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# =========================
# UKAN Multi-Task Model
# =========================
class UKAN(nn.Module):
    def __init__(self,
                 num_classes,
                 input_channels=3,
                 embed_dims=[256, 320, 512],
                 cls_classes=4,
                 **kwargs):

        super().__init__()

        base = embed_dims[0]

        # Encoder
        self.encoder1 = ConvLayer(input_channels, base // 8)
        self.encoder2 = ConvLayer(base // 8, base // 4)
        self.encoder3 = ConvLayer(base // 4, base)

        self.patch_embed3 = PatchEmbed(in_chans=base, embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.block1 = nn.ModuleList([KANBlock(embed_dims[1])])
        self.block2 = nn.ModuleList([KANBlock(embed_dims[2])])

        self.norm3 = nn.LayerNorm(embed_dims[1])
        self.norm4 = nn.LayerNorm(embed_dims[2])

        # Decoder
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer(embed_dims[1], base)
        self.decoder3 = D_ConvLayer(base, base // 4)
        self.decoder4 = D_ConvLayer(base // 4, base // 8)
        self.decoder5 = D_ConvLayer(base // 8, base // 8)

        self.final = nn.Conv2d(base // 8, num_classes, 1)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dims[2], 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, cls_classes)
        )

    def forward(self, x):
        B = x.shape[0]

        # Encoder
        t1 = F.relu(F.max_pool2d(self.encoder1(x), 2))
        t2 = F.relu(F.max_pool2d(self.encoder2(t1), 2))
        t3 = F.relu(F.max_pool2d(self.encoder3(t2), 2))

        out, H, W = self.patch_embed3(t3)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        t4 = out

        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # Classification branch
        cls_out = self.classifier(out)

        # Decoder
        out = F.interpolate(self.decoder1(out), scale_factor=2)
        out = out + t4

        out = F.interpolate(self.decoder2(out), scale_factor=2)
        out = out + t3

        out = F.interpolate(self.decoder3(out), scale_factor=2)
        out = out + t2

        out = F.interpolate(self.decoder4(out), scale_factor=2)
        out = out + t1

        out = F.interpolate(self.decoder5(out), scale_factor=2)

        seg_out = self.final(out)

        return seg_out, cls_out


__all__ = ['UKAN']
