import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.layers import CustomDropout


class VGG11Localizer(nn.Module):
    def __init__(self, pretrained_vgg: VGG11 = None, freeze_blocks: int = 2,
                 img_size: int = 224):
        super().__init__()
        self.img_size = img_size

        if pretrained_vgg is not None:
            self.block1 = pretrained_vgg.block1
            self.block2 = pretrained_vgg.block2
            self.block3 = pretrained_vgg.block3
            self.block4 = pretrained_vgg.block4
            self.block5 = pretrained_vgg.block5
        else:
            tmp = VGG11()
            self.block1 = tmp.block1
            self.block2 = tmp.block2
            self.block3 = tmp.block3
            self.block4 = tmp.block4
            self.block5 = tmp.block5

        # Only freeze first 2 blocks (low-level edges/textures)
        # Fine-tune blocks 3,4,5 for better localization
        for i, block in enumerate([self.block1, self.block2, self.block3,
                                    self.block4, self.block5], start=1):
            if i <= freeze_blocks:
                for param in block.parameters():
                    param.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Regression head — Sigmoid output for normalized [0,1] coords
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.3),        # use CustomDropout, not nn.Dropout
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.3),
            nn.Linear(256, 4),
            nn.Sigmoid()                 # output in [0,1] — multiply by img_size to get pixels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x * self.img_size    # back to pixel space [0, 224]
