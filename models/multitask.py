"""
MultiTaskPerceptionModel: single forward pass for all 3 tasks.
Uses 3 separate encoders — one per task — so each head receives
the exact feature distribution it was trained on.
"""
import os
import torch
import torch.nn as nn
import gdown

from models.vgg11 import VGG11Encoder
from models.localization import RegressionHead
from models.segmentation import DecoderBlock
from models.layers import CustomDropout

_BOTTLENECK = 7 * 7 * 512
IMAGE_SIZE  = 224


class ClassificationHead(nn.Module):
    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_BOTTLENECK, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


def _load_sd(path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def _prefix(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items()
            if k.startswith(prefix)}


class MultiTaskPerceptionModel(nn.Module):
    def __init__(self,
                 classifier_path: str = "checkpoints/classifier.pth",
                 localizer_path:  str = "checkpoints/localizer.pth",
                 unet_path:       str = "checkpoints/unet.pth",
                 num_breeds:   int = 37,
                 seg_classes:  int = 3,
                 in_channels:  int = 3):
        super().__init__()

        # Download checkpoints
        os.makedirs(os.path.dirname(classifier_path), exist_ok=True)
        gdown.download(id="1awbviDYTW8yF_oL78YW5_E_nIP6BU4M0",
                       output=classifier_path, quiet=False)
        gdown.download(id="1Ub_6gwnRiHhOQAiL6kXsJMAT5W0Jf19L",
                       output=localizer_path,  quiet=False)
        gdown.download(id="1H1Nv9hUhrRnIulPAmRYNpCcTqbUgSBdp",
                       output=unet_path,       quiet=False)

        # 3 separate encoders — each trained with its own head
        self.enc_cls = VGG11Encoder(in_channels)
        self.enc_loc = VGG11Encoder(in_channels)
        self.enc_seg = VGG11Encoder(in_channels)

        # Heads
        self.cls_head = ClassificationHead(num_breeds)
        self.loc_head = RegressionHead()

        # Segmentation decoder
        self.dec5    = DecoderBlock(512, 512, 512)
        self.dec4    = DecoderBlock(512, 512, 256)
        self.dec3    = DecoderBlock(256, 256, 128)
        self.dec2    = DecoderBlock(128, 128, 64)
        self.dec1    = DecoderBlock(64,  64,  32)
        self.seg_drop = CustomDropout(p=0.5)
        self.seg_out  = nn.Conv2d(32, seg_classes, 1)

        device = torch.device("cpu")
        self._load(classifier_path, localizer_path, unet_path, device)

    def _load(self, clf, loc, seg, device):
        # Classifier
        if os.path.isfile(clf):
            sd = _load_sd(clf, device)
            self.enc_cls.load_state_dict(_prefix(sd, "encoder."), strict=False)
            self.cls_head.load_state_dict(_prefix(sd, "head."),    strict=False)
            print(f"Loaded classifier from {clf}")

        # Localizer
        if os.path.isfile(loc):
            sd = _load_sd(loc, device)
            self.enc_loc.load_state_dict(_prefix(sd, "encoder."), strict=False)
            self.loc_head.load_state_dict(_prefix(sd, "head."),    strict=False)
            print(f"Loaded localizer from {loc}")

        # UNet
        if os.path.isfile(seg):
            sd = _load_sd(seg, device)
            self.enc_seg.load_state_dict(_prefix(sd, "encoder."), strict=False)
            for name in ["dec5", "dec4", "dec3", "dec2", "dec1"]:
                getattr(self, name).load_state_dict(
                    _prefix(sd, f"{name}."), strict=False)
            self.seg_out.load_state_dict(_prefix(sd, "final_conv."), strict=False)
            print(f"Loaded unet from {seg}")

    def forward(self, x: torch.Tensor) -> dict:
        # Classification
        cls_out = self.cls_head(self.enc_cls(x, return_features=False))

        # Localization
        loc_out = self.loc_head(self.enc_loc(x, return_features=False))

        # Segmentation
        bn, feats = self.enc_seg(x, return_features=True)
        d5  = self.dec5(bn,  feats["b5"])
        d4  = self.dec4(d5,  feats["b4"])
        d3  = self.dec3(d4,  feats["b3"])
        d2  = self.dec2(d3,  feats["b2"])
        d1  = self.dec1(d2,  feats["b1"])
        seg_out = self.seg_out(self.seg_drop(d1))

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }
