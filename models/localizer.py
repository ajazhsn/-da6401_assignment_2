import torch
import torch.nn as nn
from models.vgg11 import VGG11


class VGG11Localizer(nn.Module):
    """
    Bounding box regressor using VGG11 convolutional backbone as encoder.
    Output: [x_center, y_center, width, height] in pixel coordinates.

    Freezing strategy: We freeze blocks 1-3 (low-level features: edges, textures)
    and fine-tune blocks 4-5 (high-level semantic features). This provides a good
    balance — low-level features are generic and transfer well, while later layers
    need adaptation for localization (different task from classification).
    """

    def __init__(self, pretrained_vgg: VGG11 = None, freeze_blocks: int = 3,
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

        # Freeze early blocks
        for i, block in enumerate([self.block1, self.block2, self.block3,
                                   self.block4, self.block5], start=1):
            if i <= freeze_blocks:
                for param in block.parameters():
                    param.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.ReLU(inplace=True),   # outputs are pixel coords ≥ 0
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
        return x   # [x_center, y_center, width, height] in pixels
