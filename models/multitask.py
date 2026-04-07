import os
import torch
import torch.nn as nn
import gdown
from models.vgg11 import VGG11
from models.localizer import VGG11Localizer
from models.unet import VGG11UNet

# Alias
VGG11Encoder = VGG11


class MultiTaskPerceptionModel(nn.Module):
    def __init__(self,
                 classifier_ckpt: str = "checkpoints/classifier.pth",
                 localizer_ckpt:  str = "checkpoints/localizer.pth",
                 unet_ckpt:       str = "checkpoints/unet.pth",
                 num_classes: int = 37,
                 device: str = "cpu"):
        super().__init__()

        # ── Download checkpoints ───────────────────────────────
        os.makedirs(os.path.dirname(classifier_ckpt), exist_ok=True)
        gdown.download(id="1awbviDYTW8yF_oL78YW5_E_nIP6BU4M0",
                       output=classifier_ckpt, quiet=False)
        gdown.download(id="1Ub_6gwnRiHhOQAiL6kXsJMAT5W0Jf19L",
                       output=localizer_ckpt,  quiet=False)
        gdown.download(id="1H1Nv9hUhrRnIulPAmRYNpCcTqbUgSBdp",
                       output=unet_ckpt,       quiet=False)

        # ── Load classifier and use as shared backbone ─────────
        vgg = VGG11(num_classes=num_classes)
        vgg.load_state_dict(torch.load(classifier_ckpt, map_location=device))
        vgg.eval()

        # Shared encoder blocks
        self.enc1 = vgg.block1
        self.enc2 = vgg.block2
        self.enc3 = vgg.block3
        self.enc4 = vgg.block4
        self.enc5 = vgg.block5
        self.avgpool = vgg.avgpool

        # Classification head
        self.cls_head = vgg.classifier

        # ── Load localizer — extract regression head only ──────
        loc_model = VGG11Localizer(pretrained_vgg=vgg, freeze_blocks=0)
        loc_state = torch.load(localizer_ckpt, map_location=device)
        loc_model.load_state_dict(loc_state)
        self.loc_avgpool = loc_model.avgpool
        self.loc_head = loc_model.regressor

        # ── Load UNet — extract decoder only ──────────────────
        unet_model = VGG11UNet(pretrained_vgg=vgg, num_classes=3)
        unet_state = torch.load(unet_ckpt, map_location=device)
        unet_model.load_state_dict(unet_state)
        self.up5 = unet_model.up5
        self.dec5 = unet_model.dec5
        self.up4 = unet_model.up4
        self.dec4 = unet_model.dec4
        self.up3 = unet_model.up3
        self.dec3 = unet_model.dec3
        self.up2 = unet_model.up2
        self.dec2 = unet_model.dec2
        self.up1 = unet_model.up1
        self.dec1 = unet_model.dec1
        self.seg_out = unet_model.out_conv

        self.to(device)

    def forward(self, x: torch.Tensor) -> dict:
        # Shared encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # Classification
        cls_feat = torch.flatten(self.avgpool(e5), 1)
        class_logits = self.cls_head(cls_feat)

        # Localization
        loc_feat = torch.flatten(self.loc_avgpool(e5), 1)
        bbox_pred = self.loc_head(loc_feat)

        # Segmentation
        d5 = self.dec5(torch.cat([self.up5(e5), e4], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        d1 = self.dec1(self.up1(d2))
        seg_logits = self.seg_out(d1)

        return {
            'classification': class_logits,
            'localization':   bbox_pred,
            'segmentation':   seg_logits
        }
