import torch
import torch.nn as nn
from typing import Dict, Tuple, Union

IMAGE_SIZE = 224


def conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11Encoder(nn.Module):
    """
    VGG11 from scratch (Simonyan & Zisserman 2014).
    BatchNorm after each Conv stabilizes training and allows higher LR.
    Pools are separate from blocks so skip connections capture
    pre-pool feature maps at full spatial resolution for U-Net.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.block1 = nn.Sequential(conv_bn_relu(in_channels, 64))
        self.pool1  = nn.MaxPool2d(2, 2)
        self.block2 = nn.Sequential(conv_bn_relu(64, 128))
        self.pool2  = nn.MaxPool2d(2, 2)
        self.block3 = nn.Sequential(
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
        )
        self.pool3  = nn.MaxPool2d(2, 2)
        self.block4 = nn.Sequential(
            conv_bn_relu(256, 512),
            conv_bn_relu(512, 512),
        )
        self.pool4  = nn.MaxPool2d(2, 2)
        self.block5 = nn.Sequential(
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
        )
        self.pool5  = nn.MaxPool2d(2, 2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        return_features=False → bottleneck (B,512,7,7)
        return_features=True  → (bottleneck, dict of b1..b5)
        Skip maps are taken BEFORE pooling for maximum spatial detail.
        """
        f1 = self.block1(x)        # (B, 64,  224, 224)
        f2 = self.block2(self.pool1(f1))   # (B, 128, 112, 112)
        f3 = self.block3(self.pool2(f2))   # (B, 256,  56,  56)
        f4 = self.block4(self.pool3(f3))   # (B, 512,  28,  28)
        f5 = self.block5(self.pool4(f4))   # (B, 512,  14,  14)
        bn = self.pool5(f5)                # (B, 512,   7,   7)

        if return_features:
            return bn, {"b1": f1, "b2": f2, "b3": f3, "b4": f4, "b5": f5}
        return bn


VGG11 = VGG11Encoder
