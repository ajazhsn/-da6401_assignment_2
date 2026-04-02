import torch
import torch.nn as nn
from models.vgg11 import VGG11


def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11UNet(nn.Module):
    """
    U-Net style segmentation using VGG11 as contracting path.
    - Transposed convolutions for upsampling (no bilinear/unpooling).
    - Skip connections via channel concatenation at each scale.
    - Output: 3-class trimap (background, foreground, border).

    Loss justification: Combined BCE + Dice loss. BCE provides stable pixel-wise
    gradients; Dice directly optimizes the evaluation metric and handles class
    imbalance between foreground and background pixels.
    """

    def __init__(self, pretrained_vgg: VGG11 = None, num_classes: int = 3,
                 freeze_encoder: bool = False):
        super().__init__()

        if pretrained_vgg is not None:
            self.enc1 = pretrained_vgg.block1   # out: 64,  112x112
            self.enc2 = pretrained_vgg.block2   # out: 128,  56x56
            self.enc3 = pretrained_vgg.block3   # out: 256,  28x28
            self.enc4 = pretrained_vgg.block4   # out: 512,  14x14
            self.enc5 = pretrained_vgg.block5   # out: 512,   7x7
        else:
            tmp = VGG11()
            self.enc1 = tmp.block1
            self.enc2 = tmp.block2
            self.enc3 = tmp.block3
            self.enc4 = tmp.block4
            self.enc5 = tmp.block5

        if freeze_encoder:
            for block in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
                for p in block.parameters():
                    p.requires_grad = False

        # Decoder — TransposedConv upsampling + skip connections
        # up5: 7→14, concatenate with enc4 (512), so in=512+512=1024
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = double_conv(512 + 512, 512)

        # up4: 14→28, concatenate with enc3 (256), so in=512+256=768
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = double_conv(256 + 256, 256)

        # up3: 28→56, concatenate with enc2 (128), so in=256+128=384
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = double_conv(128 + 128, 128)

        # up2: 56→112, concatenate with enc1 (64), so in=128+64=192
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = double_conv(64 + 64, 64)

        # up1: 112→224 (restore original resolution)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = double_conv(32, 32)

        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder (note: VGG blocks include MaxPool)
        e1 = self.enc1(x)   # 64,  112
        e2 = self.enc2(e1)  # 128,  56
        e3 = self.enc3(e2)  # 256,  28
        e4 = self.enc4(e3)  # 512,  14
        e5 = self.enc5(e4)  # 512,   7

        # Decoder with skip connections
        d5 = self.up5(e5)                          # 512, 14
        d5 = self.dec5(torch.cat([d5, e4], dim=1))  # 512, 14

        d4 = self.up4(d5)                          # 256, 28
        d4 = self.dec4(torch.cat([d4, e3], dim=1))  # 256, 28

        d3 = self.up3(d4)                          # 128, 56
        d3 = self.dec3(torch.cat([d3, e2], dim=1))  # 128, 56

        d2 = self.up2(d3)                          # 64, 112
        d2 = self.dec2(torch.cat([d2, e1], dim=1))  # 64, 112

        d1 = self.up1(d2)                          # 32, 224
        d1 = self.dec1(d1)

        return self.out_conv(d1)   # num_classes, 224, 224
