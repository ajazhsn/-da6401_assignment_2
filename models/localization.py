import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder, IMAGE_SIZE
from models.layers import CustomDropout

_BN_DIM = 7 * 7 * 512


class RegressionHead(nn.Module):
    """
    Bbox regression head. Output: [cx,cy,w,h] in pixel space.
    Sigmoid * IMAGE_SIZE constrains output to valid image coordinates,
    preventing collapsed IoU from out-of-bounds predictions.
    """
    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_BN_DIM, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(x)) * IMAGE_SIZE


class VGG11Localizer(nn.Module):
    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels)
        self.head    = RegressionHead(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x, return_features=False))
