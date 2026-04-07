"""
MultiTaskPerceptionModel: downloads checkpoints from Google Drive,
loads classifier.pth, localizer.pth, unet.pth and performs all
three tasks in a single forward pass.
"""
import os
import torch
import torch.nn as nn
import gdown
from models.vgg11 import VGG11
from models.localizer import VGG11Localizer
from models.unet import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """
    Unified multi-task model.
    Single forward pass → (class_logits, bbox_pred, seg_mask_logits)
    Downloads weights from Google Drive and shares the VGG11 backbone.
    """

    def __init__(self,
                 classifier_ckpt: str = "checkpoints/classifier.pth",
                 localizer_ckpt:  str = "checkpoints/localizer.pth",
                 unet_ckpt:       str = "checkpoints/unet.pth",
                 num_classes: int = 37,
                 device: str = "cpu"):
        super().__init__()
        self.device = device

        # ── Download checkpoints from Google Drive ─────────────
        os.makedirs(os.path.dirname(classifier_ckpt), exist_ok=True)

        gdown.download(
            id="1awbviDYTW8yF_oL78YW5_E_nIP6BU4M0",
            output=classifier_ckpt, quiet=False)
        gdown.download(
            id="1Ub_6gwnRiHhOQAiL6kXsJMAT5W0Jf19L",
            output=localizer_ckpt, quiet=False)
        gdown.download(
            id="1H1Nv9hUhrRnIulPAmRYNpCcTqbUgSBdp",
            output=unet_ckpt, quiet=False)

        # ── Build classifier and load weights ──────────────────
        self.classifier = VGG11(num_classes=num_classes)
        self.classifier.load_state_dict(
            torch.load(classifier_ckpt, map_location=device))

        # ── Shared backbone blocks from classifier ─────────────
        self.shared_enc1 = self.classifier.block1
        self.shared_enc2 = self.classifier.block2
        self.shared_enc3 = self.classifier.block3
        self.shared_enc4 = self.classifier.block4
        self.shared_enc5 = self.classifier.block5

        # ── Classification head ────────────────────────────────
        self.cls_avgpool = self.classifier.avgpool
        self.cls_head = self.classifier.classifier

        # ── Localizer — extract regression head ────────────────
        _loc = VGG11Localizer(pretrained_vgg=self.classifier)
        _loc.load_state_dict(
            torch.load(localizer_ckpt, map_location=device))
        self.loc_avgpool = _loc.avgpool
        self.loc_head = _loc.regressor

        # ── UNet — extract decoder ─────────────────────────────
        _unet = VGG11UNet(pretrained_vgg=self.classifier, num_classes=3)
        _unet.load_state_dict(
            torch.load(unet_ckpt, map_location=device))
        self.unet_up5 = _unet.up5
        self.unet_dec5 = _unet.dec5
        self.unet_up4 = _unet.up4
        self.unet_dec4 = _unet.dec4
        self.unet_up3 = _unet.up3
        self.unet_dec3 = _unet.dec3
        self.unet_up2 = _unet.up2
        self.unet_dec2 = _unet.dec2
        self.unet_up1 = _unet.up1
        self.unet_dec1 = _unet.dec1
        self.unet_out = _unet.out_conv

        self.to(device)

    def forward(self, x: torch.Tensor):
        """
        x: (N, 3, 224, 224) normalized image tensor
        Returns:
            class_logits : (N, 37)
            bbox_pred    : (N, 4)  [cx, cy, w, h] in pixel space
            seg_logits   : (N, 3, 224, 224)
        """
        # ── Shared encoder ─────────────────────────────────────
        e1 = self.shared_enc1(x)   # 64,  112
        e2 = self.shared_enc2(e1)  # 128,  56
        e3 = self.shared_enc3(e2)  # 256,  28
        e4 = self.shared_enc4(e3)  # 512,  14
        e5 = self.shared_enc5(e4)  # 512,   7

        # ── Classification ─────────────────────────────────────
        cls_feat = torch.flatten(self.cls_avgpool(e5), 1)
        class_logits = self.cls_head(cls_feat)

        # ── Localization ───────────────────────────────────────
        loc_feat = torch.flatten(self.loc_avgpool(e5), 1)
        bbox_pred = self.loc_head(loc_feat)

        # ── Segmentation decoder ───────────────────────────────
        d5 = self.unet_dec5(torch.cat([self.unet_up5(e5), e4], dim=1))
        d4 = self.unet_dec4(torch.cat([self.unet_up4(d5), e3], dim=1))
        d3 = self.unet_dec3(torch.cat([self.unet_up3(d4), e2], dim=1))
        d2 = self.unet_dec2(torch.cat([self.unet_up2(d3), e1], dim=1))
        d1 = self.unet_dec1(self.unet_up1(d2))
        seg_logits = self.unet_out(d1)

        # NEW - returns dictionary
        return {
            'classification': class_logits,
            'localization':   bbox_pred,
            'segmentation':   seg_logits
        }


if __name__ == "__main__":
    model = MultiTaskPerceptionModel(
        classifier_ckpt="checkpoints/classifier.pth",
        localizer_ckpt="checkpoints/localizer.pth",
        unet_ckpt="checkpoints/unet.pth",
    )
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    cls, bbox, seg = model(x)
    print("cls :", cls.shape)   # (2, 37)
    print("bbox:", bbox.shape)  # (2, 4)
    print("seg :", seg.shape)   # (2, 3, 224, 224)
