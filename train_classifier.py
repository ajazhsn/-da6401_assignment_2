import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import math
from sklearn.metrics import f1_score
from models.classification import VGG11Classifier
from dataset import PetDataset


def train_classifier(data_root, epochs=60, lr=3e-4, batch_size=64,
                     dropout_p=0.4, device="cuda",
                     save_path="checkpoints/classifier.pth"):
    wandb.init(project="da6401-a2", name="classifier-final")

    train_ds = PetDataset(data_root, split="train", task="classify")
    val_ds   = PetDataset(data_root, split="val",   task="classify")
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

    model     = VGG11Classifier(num_classes=37, dropout_p=dropout_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < 3:
            return (epoch + 1) / 3
        return 0.5 * (1 + math.cos(math.pi * (epoch-3) / max(epochs-3, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    eval_crit = nn.CrossEntropyLoss()
    best_f1   = 0.0

    for epoch in range(epochs):
        model.train()
        tl, tc, tt = 0, 0, 0
        tp, tlab = [], []
        for batch in train_dl:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * imgs.size(0)
            p   = logits.argmax(1)
            tc += (p == labels).sum().item(); tt += imgs.size(0)
            tp.extend(p.cpu().numpy()); tlab.extend(labels.cpu().numpy())
        scheduler.step()
        train_f1 = f1_score(tlab, tp, average="macro", zero_division=0)

        model.eval()
        vl, vc, vt = 0, 0, 0
        vp, vlab = [], []
        with torch.no_grad():
            for batch in val_dl:
                imgs   = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(imgs)
                loss   = eval_crit(logits, labels)
                vl += loss.item() * imgs.size(0)
                p   = logits.argmax(1)
                vc += (p == labels).sum().item(); vt += imgs.size(0)
                vp.extend(p.cpu().numpy()); vlab.extend(labels.cpu().numpy())
        val_f1 = f1_score(vlab, vp, average="macro", zero_division=0)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}/{epochs} | LR:{lr_now:.6f} | "
              f"Train F1:{train_f1:.4f} | Val F1:{val_f1:.4f}")
        wandb.log({"train/f1": train_f1, "train/loss": tl/tt,
                   "val/f1": val_f1, "val/loss": vl/vt,
                   "lr": lr_now, "epoch": epoch+1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved (val_f1={val_f1:.4f})")

    wandb.finish()
    print(f"Best Val F1: {best_f1:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default="./oxford-iiit-pet")
    p.add_argument("--epochs",     type=int,   default=60)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--dropout_p",  type=float, default=0.4)
    p.add_argument("--save_path",  default="checkpoints/classifier.pth")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_classifier(args.data_root, args.epochs, args.lr,
                     args.batch_size, args.dropout_p, device, args.save_path)
