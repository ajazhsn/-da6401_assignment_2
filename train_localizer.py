import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from models.vgg11 import VGG11
from models.localizer import VGG11Localizer
from losses.iou_loss import IoULoss
from dataset import PetDataset


def train_localizer(data_root: str, classifier_ckpt: str = "classifier.pth",
                    epochs: int = 25, lr: float = 1e-4, batch_size: int = 32,
                    device: str = "cuda", save_path: str = "localizer.pth"):
    wandb.init(project="da6401-a2", name="vgg11-localizer")

    train_ds = PetDataset(data_root, split="train", task="localize")
    val_ds   = PetDataset(data_root, split="val",   task="localize")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    # Load pretrained VGG11
    vgg = VGG11(num_classes=37)
    vgg.load_state_dict(torch.load(classifier_ckpt, map_location="cpu"))

    model = VGG11Localizer(pretrained_vgg=vgg, freeze_blocks=3).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=4, factor=0.5)

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")

    IMG_SIZE = 224.0  # normalize coords to [0,1] range for stable MSE

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # ── TRAINING ──────────────────────────────
        model.train()
        total_loss = 0; n = 0
        for batch in train_dl:
            imgs   = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)
            mask   = (bboxes.sum(dim=1) > 0)
            if mask.sum() == 0:
                continue
            imgs   = imgs[mask]
            bboxes = bboxes[mask]

            # Normalize to [0,1] for stable MSE
            bboxes_norm = bboxes / IMG_SIZE

            optimizer.zero_grad()
            preds = model(imgs) / IMG_SIZE   # normalize preds too
            loss  = mse_loss(preds, bboxes_norm) + iou_loss(preds, bboxes_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

        train_loss = total_loss / max(n, 1)

        # ── VALIDATION ────────────────────────────
        model.eval()
        val_loss = 0; vn = 0
        with torch.no_grad():
            for batch in val_dl:
                imgs   = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)
                mask   = (bboxes.sum(dim=1) > 0)
                if mask.sum() == 0:
                    continue
                imgs   = imgs[mask]
                bboxes = bboxes[mask]
                bboxes_norm = bboxes / IMG_SIZE
                preds  = model(imgs) / IMG_SIZE
                loss   = mse_loss(preds, bboxes_norm) + iou_loss(preds, bboxes_norm)
                val_loss += loss.item() * imgs.size(0)
                vn += imgs.size(0)

        val_loss = val_loss / max(vn, 1)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        wandb.log({"train/loc_loss": train_loss, "val/loc_loss": val_loss, "epoch": epoch+1})

        # ── SAVE BEST ─────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model → {save_path}")

    wandb.finish()
    print(f"Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",        default="./oxford-iiit-pet")
    p.add_argument("--classifier_ckpt",  default="classifier.pth")
    p.add_argument("--epochs",           type=int,   default=25)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--save_path",        default="localizer.pth")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_localizer(args.data_root, args.classifier_ckpt, args.epochs,
                    args.lr, args.batch_size, device, args.save_path)
