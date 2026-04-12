"""
MultiTaskPerceptionModel — single forward pass, 3 separate encoders.
Each encoder is loaded from its own checkpoint so each head receives
the feature distribution it was trained on.
"""
import os
import torch
import torch.nn as nn
import gdown

from models.vgg11 import VGG11Encoder
from models.classification import ClassificationHead
from models.localization import RegressionHead
from models.segmentation import DecoderBlock
from models.layers import CustomDropout


def _load_ckpt(path: str, device: torch.device) -> dict:
    sd = torch.load(path, map_location=device)
    return sd.get("state_dict", sd) if isinstance(sd, dict) else sd


def _sub(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items()
            if k.startswith(prefix)}


class MultiTaskPerceptionModel(nn.Module):
    def __init__(self,
                 classifier_path: str = "checkpoints/classifier.pth",
                 localizer_path:  str = "checkpoints/localizer.pth",
                 unet_path:       str = "checkpoints/unet.pth",
                 num_breeds:  int = 37,
                 seg_classes: int = 3,
                 in_channels: int = 3):
        super().__init__()

        os.makedirs(os.path.dirname(classifier_path), exist_ok=True)
        gdown.download(id="1hZfpmwz3xBoWv3mBuYLhoR6cJBoE7nNZ",
                       output=classifier_path, quiet=False)
        gdown.download(id="1BfYlykJNyy-FwXPiR5qF89YA43aZWDgO",
                       output=localizer_path,  quiet=False)
        gdown.download(id="1GkmVFmPEVSyiJiqLWswnwtry7n3e8YY7",
                       output=unet_path,       quiet=False)

        # 3 independent encoders — critical for correct head performance
        self.enc_cls = VGG11Encoder(in_channels)
        self.enc_loc = VGG11Encoder(in_channels)
        self.enc_seg = VGG11Encoder(in_channels)

        self.cls_head = ClassificationHead(num_breeds)
        self.loc_head = RegressionHead()

        self.dec5     = DecoderBlock(512, 512, 512)
        self.dec4     = DecoderBlock(512, 512, 256)
        self.dec3     = DecoderBlock(256, 256, 128)
        self.dec2     = DecoderBlock(128, 128, 64)
        self.dec1     = DecoderBlock(64,  64,  32)
        self.seg_drop = CustomDropout(p=0.5)
        self.seg_out  = nn.Conv2d(32, seg_classes, 1)

        self._load_weights(classifier_path, localizer_path,
                           unet_path, torch.device("cpu"))

    def _load_weights(self, clf, loc, seg, device):
        if os.path.isfile(clf):
            sd = _load_ckpt(clf, device)
            self.enc_cls.load_state_dict(_sub(sd, "encoder."), strict=False)
            self.cls_head.load_state_dict(_sub(sd, "head."),   strict=False)

        if os.path.isfile(loc):
            sd = _load_ckpt(loc, device)
            self.enc_loc.load_state_dict(_sub(sd, "encoder."), strict=False)
            self.loc_head.load_state_dict(_sub(sd, "head."),   strict=False)

        if os.path.isfile(seg):
            sd = _load_ckpt(seg, device)
            self.enc_seg.load_state_dict(_sub(sd, "encoder."), strict=False)
            for name in ["dec5","dec4","dec3","dec2","dec1"]:
                getattr(self, name).load_state_dict(
                    _sub(sd, f"{name}."), strict=False)
            self.seg_out.load_state_dict(_sub(sd, "final_conv."), strict=False)

    def forward(self, x: torch.Tensor) -> dict:
        cls_out = self.cls_head(self.enc_cls(x, return_features=False))
        loc_out = self.loc_head(self.enc_loc(x, return_features=False))

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
