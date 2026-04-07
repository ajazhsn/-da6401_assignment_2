"""
Inference script for DA6401 Assignment 2.
"""
import torch
import numpy as np
from PIL import Image
from models.multitask import MultiTaskPerceptionModel

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def preprocess(image_path: str, img_size: int = 224) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    x = np.array(img).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)


def run_inference(image_path: str, device: str = "cpu"):
    model = MultiTaskPerceptionModel(
        classifier_ckpt="checkpoints/classifier.pth",
        localizer_ckpt="checkpoints/localizer.pth",
        unet_ckpt="checkpoints/unet.pth",
        device=device
    )
    model.eval()
    img_tensor = preprocess(image_path).to(device)
    with torch.no_grad():
        cls_logits, bbox, seg_logits = model(img_tensor)
    pred_class = cls_logits.argmax(1).item()
    pred_bbox = bbox[0].cpu().numpy()
    pred_mask = seg_logits[0].argmax(0).cpu().numpy()
    print(f"Predicted class: {pred_class}")
    print(f"Predicted bbox:  {pred_bbox}")
    print(f"Mask shape:      {pred_mask.shape}")
    return pred_class, pred_bbox, pred_mask


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image",  required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    run_inference(args.image, args.device)
