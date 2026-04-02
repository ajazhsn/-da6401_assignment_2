import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import numpy as np
from PIL import Image
from models.vgg11 import VGG11
from dataset import PetDataset


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation — reduces overfitting significantly."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_classifier(data_root: str, epochs: int = 60, lr: float = 1e-3,
                     batch_size: int = 64, dropout_p: float = 0.4,
                     device: str = "cuda", save_path: str = "classifier.pth"):
    wandb.init(project="da6401-a2", name="vgg11-classifier-v2")

    train_ds = PetDataset(data_root, split="train", task="classify")
    val_ds   = PetDataset(data_root, split="val",   task="classify")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)

    model     = VGG11(num_classes=37, dropout_p=dropout_p).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4, nesterov=True)

    # Warmup for 5 epochs then cosine decay
    def lr_lambda(epoch):
        if epoch < 5:
            return (epoch + 1) / 5      # linear warmup
        progress = (epoch - 5) / (epochs - 5)
        return 0.5 * (1 + np.cos(np.pi * progress))  # cosine decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing reduces overfit

    best_val_acc = 0.0

    for epoch in range(epochs):
        # ── TRAINING ──────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in train_dl:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Apply mixup
            imgs_mix, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.2)

            optimizer.zero_grad()
            logits = model(imgs_mix)
            loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            # For accuracy, use original labels (not mixed)
            with torch.no_grad():
                orig_logits = model(imgs)
            correct += (orig_logits.argmax(1) == labels).sum().item()
            total   += imgs.size(0)

        scheduler.step()
        train_loss = total_loss / total
        train_acc  = correct / total

        # ── VALIDATION ────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_dl:
                imgs   = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(imgs)
                loss   = criterion(logits, labels)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)
        val_loss /= val_total
        val_acc   = val_correct / val_total

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.5f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        wandb.log({"train/loss": train_loss, "train/acc": train_acc,
                   "val/loss":   val_loss,   "val/acc":   val_acc,
                   "lr": current_lr,         "epoch":     epoch+1})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model → {save_path}  (val_acc={val_acc:.4f})")

    wandb.finish()
    print(f"Best Val Acc: {best_val_acc:.4f}")
    return model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default="./oxford-iiit-pet")
    p.add_argument("--epochs",     type=int,   default=60)
    p.add_argument("--lr",         type=float, default=0.01)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--dropout_p",  type=float, default=0.4)
    p.add_argument("--save_path",  default="classifier.pth")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_classifier(args.data_root, args.epochs, args.lr,
                     args.batch_size, args.dropout_p, device, args.save_path)
