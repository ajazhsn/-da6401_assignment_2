import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


def conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DecoderBlock(nn.Module):
    """
    TransposedConv upsample + skip concat + 2x conv-bn-relu.
    No bilinear interpolation as per assignment requirement.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            conv_bn_relu(in_ch + skip_ch, out_ch),
            conv_bn_relu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            skip = skip[:, :, :x.shape[2], :x.shape[3]]
        return self.conv(torch.cat([x, skip], dim=1))


class VGG11UNet(nn.Module):
    """
    U-Net segmentation on VGG11 encoder.
    Loss: CE + Dice (CE for stable gradients, Dice for class imbalance).
    """
    def __init__(self, num_classes: int = 3, in_channels: int = 3,
                 dropout_p: float = 0.5):
        super().__init__()
        self.encoder    = VGG11Encoder(in_channels)
        self.dec5       = DecoderBlock(512, 512, 512)
        self.dec4       = DecoderBlock(512, 512, 256)
        self.dec3       = DecoderBlock(256, 256, 128)
        self.dec2       = DecoderBlock(128, 128, 64)
        self.dec1       = DecoderBlock(64,  64,  32)
        self.dropout    = CustomDropout(p=dropout_p)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        self._init_decoder()

    def _init_decoder(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bn, feats = self.encoder(x, return_features=True)
        d5 = self.dec5(bn,  feats["b5"])
        d4 = self.dec4(d5,  feats["b4"])
        d3 = self.dec3(d4,  feats["b3"])
        d2 = self.dec2(d3,  feats["b2"])
        d1 = self.dec1(d2,  feats["b1"])
        return self.final_conv(self.dropout(d1))
