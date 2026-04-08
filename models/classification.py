import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder, IMAGE_SIZE
from models.layers import CustomDropout

_BN_DIM = 7 * 7 * 512


class ClassificationHead(nn.Module):
    """
    FC head on VGG11 bottleneck.
    BatchNorm1d before Dropout: BN sees full distribution,
    Dropout then acts as ensemble regularizer on normalized features.
    """
    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_BN_DIM, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
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
        return self.head(x)


class VGG11Classifier(nn.Module):
    def __init__(self, num_classes: int = 37, in_channels: int = 3,
                 dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels)
        self.head    = ClassificationHead(num_classes, dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x, return_features=False))
