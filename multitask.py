"""
MultiTaskPerceptionModel: loads classifier.pth, localizer.pth, unet.pth
and performs all three tasks in a single forward pass.
Paths are relative as required by the assignment spec.
"""
import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.localizer import VGG11Localizer
from models.unet import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """
    Unified multi-task model.
    Single forward pass → (class_logits, bbox_pred, seg_mask_logits)
    Loads weights from saved checkpoints and shares the VGG11 backbone.
    """

    def __init__(self,
                 classifier_ckpt: str = "classifier.pth",
                 localizer_ckpt:  str = "localizer.pth",
                 unet_ckpt:       str = "unet.pth",
                 num_classes: int = 37,
                 device: str = "cpu"):
        super().__init__()
        self.device = device

        # --- Build and load each component ---
        # Classifier
        self.classifier = VGG11(num_classes=num_classes)
        self.classifier.load_state_dict(
            torch.load(classifier_ckpt, map_location=device))

        # Shared backbone: use the classifier's blocks
        self.shared_enc1 = self.classifier.block1
        self.shared_enc2 = self.classifier.block2
        self.shared_enc3 = self.classifier.block3
        self.shared_enc4 = self.classifier.block4
        self.shared_enc5 = self.classifier.block5

        # Localizer — load full model, then extract regression head only
        _loc = VGG11Localizer(pretrained_vgg=self.classifier)
        _loc.load_state_dict(torch.load(localizer_ckpt, map_location=device))
        self.loc_head = _loc.regressor
        self.loc_avgpool = _loc.avgpool

        # UNet — load full model, extract decoder only
        _unet = VGG11UNet(pretrained_vgg=self.classifier, num_classes=3)
        _unet.load_state_dict(torch.load(unet_ckpt, map_location=device))
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

        # Classification head
        self.cls_avgpool = self.classifier.avgpool
        self.cls_head = self.classifier.classifier

        self.to(device)

    def forward(self, x: torch.Tensor):
        """
        x: (N, 3, 224, 224) normalized image tensor
        Returns:
            class_logits:  (N, 37)
            bbox_pred:     (N, 4)   [xc, yc, w, h] in pixel space
            seg_logits:    (N, 3, 224, 224)
        """
        # --- Shared encoder ---
        e1 = self.shared_enc1(x)
        e2 = self.shared_enc2(e1)
        e3 = self.shared_enc3(e2)
        e4 = self.shared_enc4(e3)
        e5 = self.shared_enc5(e4)

        # --- Classification head ---
        cls_feat = self.cls_avgpool(e5)
        cls_feat = torch.flatten(cls_feat, 1)
        class_logits = self.cls_head(cls_feat)

        # --- Localization head ---
        loc_feat = self.loc_avgpool(e5)
        loc_feat = torch.flatten(loc_feat, 1)
        bbox_pred = self.loc_head(loc_feat)

        # --- Segmentation decoder (U-Net) ---
        d5 = self.unet_up5(e5)
        d5 = self.unet_dec5(torch.cat([d5, e4], dim=1))
        d4 = self.unet_up4(d5)
        d4 = self.unet_dec4(torch.cat([d4, e3], dim=1))
        d3 = self.unet_up3(d4)
        d3 = self.unet_dec3(torch.cat([d3, e2], dim=1))
        d2 = self.unet_up2(d3)
        d2 = self.unet_dec2(torch.cat([d2, e1], dim=1))
        d1 = self.unet_up1(d2)
        d1 = self.unet_dec1(d1)
        seg_logits = self.unet_out(d1)

        return class_logits, bbox_pred, seg_logits


if __name__ == "__main__":
    # Quick sanity check
    model = MultiTaskPerceptionModel(
        classifier_ckpt="classifier.pth",
        localizer_ckpt="localizer.pth",
        unet_ckpt="unet.pth",
    )
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    cls, bbox, seg = model(x)
    print("cls:", cls.shape)    # (2, 37)
    print("bbox:", bbox.shape)  # (2, 4)
    print("seg:", seg.shape)    # (2, 3, 224, 224)
