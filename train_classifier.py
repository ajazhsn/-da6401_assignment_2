import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import math
from sklearn.metrics import f1_score
from models.vgg11 import VGG11
from dataset import PetDataset


def train_classifier(data_root: str, epochs: int = 60, lr: float = 3e-4,
                     batch_size: int = 64, dropout_p: float = 0.4,
                     device: str = "cuda", save_path: str = "classifier.pth"):

    wandb.init(project="da6401-a2", name="vgg11-classifier-final")

    train_ds = PetDataset(data_root, split="train", task="classify")
    val_ds   = PetDataset(data_root, split="val",   task="classify")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)

    model     = VGG11(num_classes=37, dropout_p=dropout_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < 3:
            return (epoch + 1) / 3
        progress = (epoch - 3) / (epochs - 3)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
    eval_crit  = nn.CrossEntropyLoss()   # no smoothing for validation

    best_val_f1 = 0.0

    for epoch in range(epochs):
        # ── TRAINING ──────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0, 0, 0
        train_preds_all, train_labels_all = [], []

        for batch in train_dl:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds       = logits.argmax(1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            train_preds_all.extend(preds.cpu().numpy())
            train_labels_all.extend(labels.cpu().numpy())

        scheduler.step()
        train_loss = total_loss / total
        train_acc  = correct / total
        train_f1   = f1_score(train_labels_all, train_preds_all,
                              average="macro", zero_division=0)

        # ── VALIDATION ────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_preds_all, val_labels_all = [], []

        with torch.no_grad():
            for batch in val_dl:
                imgs   = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(imgs)
                loss   = eval_crit(logits, labels)
                val_loss    += loss.item() * imgs.size(0)
                preds        = logits.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)
                val_preds_all.extend(preds.cpu().numpy())
                val_labels_all.extend(labels.cpu().numpy())

        val_loss /= val_total
        val_acc   = val_correct / val_total
        val_f1    = f1_score(val_labels_all, val_preds_all,
                             average="macro", zero_division=0)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        wandb.log({
            "train/loss":     train_loss,
            "train/acc":      train_acc,
            "train/macro_f1": train_f1,
            "val/loss":       val_loss,
            "val/acc":        val_acc,
            "val/macro_f1":   val_f1,
            "lr":             current_lr,
            "epoch":          epoch + 1
        })

        # Save best by F1 — this is what the autograder measures
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model (val_f1={val_f1:.4f}, val_acc={val_acc:.4f})")

    wandb.finish()
    print(f"Best Val Macro F1: {best_val_f1:.4f}")
    return model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default="./oxford-iiit-pet")
    p.add_argument("--epochs",     type=int,   default=60)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--dropout_p",  type=float, default=0.4)
    p.add_argument("--save_path",  default="classifier.pth")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_classifier(args.data_root, args.epochs, args.lr,
                     args.batch_size, args.dropout_p, device, args.save_path)
