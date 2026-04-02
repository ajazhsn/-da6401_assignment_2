import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from models.vgg11 import VGG11
from models.unet import VGG11UNet
from dataset import PetDataset


class DiceLoss(nn.Module):
    """
    Soft Dice Loss over all classes.
    Works on softmax probabilities vs one-hot targets.
    Smooth=1 avoids division by zero on empty classes.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, C, H, W)   targets: (N, H, W)
        C = logits.shape[1]
        probs      = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, C).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_oh).sum(dim=(2, 3))
        union        = probs.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))
        dice         = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor):
    """
    Returns (dice_score, pixel_accuracy) for a batch.
    Both are averaged over the batch.
    """
    C     = logits.shape[1]
    preds = logits.argmax(dim=1)   # (N, H, W)

    # Pixel accuracy
    pixel_acc = (preds == targets).float().mean().item()

    # Dice score per class then averaged
    dice = 0.0
    for c in range(C):
        pred_c  = (preds   == c).float()
        tgt_c   = (targets == c).float()
        inter   = (pred_c * tgt_c).sum().item()
        denom   = pred_c.sum().item() + tgt_c.sum().item()
        dice   += (2 * inter + 1.0) / (denom + 1.0)
    dice /= C

    return dice, pixel_acc


def train_unet(data_root: str, classifier_ckpt: str = "classifier.pth",
               epochs: int = 30, lr: float = 1e-4, batch_size: int = 16,
               freeze_encoder: bool = False, freeze_blocks: int = 0,
               device: str = "cuda", save_path: str = "unet.pth",
               run_name: str = "unet-full-finetune"):

    wandb.init(project="da6401-a2", name=run_name)

    train_ds = PetDataset(data_root, split="train", task="segment")
    val_ds   = PetDataset(data_root, split="val",   task="segment")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)

    # Load pretrained VGG11 backbone
    vgg = VGG11(num_classes=37)
    vgg.load_state_dict(torch.load(classifier_ckpt, map_location="cpu"))

    model = VGG11UNet(
        pretrained_vgg=vgg,
        num_classes=3,
        freeze_encoder=freeze_encoder,
        freeze_blocks=freeze_blocks
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    # Class weights: border pixels are rare → upweight them
    # trimap: 0=bg, 1=fg, 2=border
    class_weights = torch.tensor([1.0, 1.0, 2.0]).to(device)
    ce_loss   = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss(smooth=1.0)

    best_dice = 0.0

    for epoch in range(epochs):
        # ── TRAINING ──────────────────────────────────────────
        model.train()
        train_loss = 0.0; n = 0
        for batch in train_dl:
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = ce_loss(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        scheduler.step()
        train_loss /= n

        # ── VALIDATION ────────────────────────────────────────
        model.eval()
        val_loss = 0.0; val_dice = 0.0; val_pix_acc = 0.0; vn = 0
        with torch.no_grad():
            for batch in val_dl:
                imgs  = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                loss   = ce_loss(logits, masks) + dice_loss(logits, masks)
                dice, pix_acc = compute_metrics(logits, masks)
                val_loss    += loss.item()  * imgs.size(0)
                val_dice    += dice         * imgs.size(0)
                val_pix_acc += pix_acc      * imgs.size(0)
                vn += imgs.size(0)

        val_loss    /= vn
        val_dice    /= vn
        val_pix_acc /= vn

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Dice: {val_dice:.4f} | "
              f"Pix Acc: {val_pix_acc:.4f}")

        wandb.log({
            "train/seg_loss":   train_loss,
            "val/seg_loss":     val_loss,
            "val/dice":         val_dice,
            "val/pixel_acc":    val_pix_acc,
            "epoch":            epoch + 1
        })

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model → {save_path}  (dice={val_dice:.4f})")

    wandb.finish()
    print(f"Best Val Dice: {best_dice:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",       default="./oxford-iiit-pet")
    p.add_argument("--classifier_ckpt", default="classifier.pth")
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--freeze_encoder",  action="store_true")
    p.add_argument("--freeze_blocks",   type=int,   default=0)
    p.add_argument("--save_path",       default="unet.pth")
    p.add_argument("--run_name",        default="unet-full-finetune")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_unet(args.data_root, args.classifier_ckpt, args.epochs, args.lr,
               args.batch_size, args.freeze_encoder, args.freeze_blocks,
               device, args.save_path, args.run_name)
