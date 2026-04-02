import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from models.vgg11 import VGG11
from models.localizer import VGG11Localizer
from losses.iou_loss import IoULoss
from dataset import PetDataset


def train_localizer(data_root: str, classifier_ckpt: str = "classifier.pth",
                    epochs: int = 30, lr: float = 1e-4, batch_size: int = 32,
                    device: str = "cuda", save_path: str = "localizer.pth"):
    wandb.init(project="da6401-a2", name="vgg11-localizer")

    train_ds = PetDataset(data_root, split="train", task="localize")
    val_ds = PetDataset(data_root, split="val",   task="localize")
    # Filter out samples without bbox
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=2)
    val_dl = DataLoader(val_ds,   batch_size=batch_size,
                        shuffle=False, num_workers=2)

    # Load pretrained VGG11
    vgg = VGG11(num_classes=37)
    vgg.load_state_dict(torch.load(classifier_ckpt, map_location="cpu"))

    model = VGG11Localizer(pretrained_vgg=vgg, freeze_blocks=3).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5)

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n = 0
        for batch in train_dl:
            imgs = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)
            mask = (bboxes.sum(dim=1) > 0)   # skip samples with no bbox
            if mask.sum() == 0:
                continue
            imgs = imgs[mask]
            bboxes = bboxes[mask]
            optimizer.zero_grad()
            preds = model(imgs)
            loss = mse_loss(preds, bboxes) + iou_loss(preds, bboxes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        scheduler.step()

        val_loss = 0
        vn = 0
        model.eval()
        with torch.no_grad():
            for batch in val_dl:
                imgs = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)
                mask = (bboxes.sum(dim=1) > 0)
                if mask.sum() == 0:
                    continue
                imgs = imgs[mask]
                bboxes = bboxes[mask]
                preds = model(imgs)
                loss = mse_loss(preds, bboxes) + iou_loss(preds, bboxes)
                val_loss += loss.item() * imgs.size(0)
                vn += imgs.size(0)

        tl = total_loss/max(n, 1)
        vl = val_loss/max(vn, 1)
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {tl:.4f} | Val Loss: {vl:.4f}")
        wandb.log({"train/loc_loss": tl, "val/loc_loss": vl, "epoch": epoch+1})

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), save_path)
    wandb.finish()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",        default="./oxford-iiit-pet")
    p.add_argument("--classifier_ckpt",  default="classifier.pth")
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--save_path",        default="localizer.pth")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_localizer(args.data_root, args.classifier_ckpt, args.epochs,
                    args.lr, args.batch_size, device, args.save_path)
