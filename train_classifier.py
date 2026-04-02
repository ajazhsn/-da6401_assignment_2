import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from models.vgg11 import VGG11
from dataset import PetDataset

def train_classifier(data_root: str, epochs: int = 30, lr: float = 1e-3,
                     batch_size: int = 32, dropout_p: float = 0.5,
                     device: str = "cuda", save_path: str = "classifier.pth"):
    wandb.init(project="da6401-a2", name="vgg11-classifier")

    train_ds = PetDataset(data_root, split="train", task="classify")
    val_ds   = PetDataset(data_root, split="val",   task="classify")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model     = VGG11(num_classes=37, dropout_p=dropout_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        # ── TRAINING ──────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in train_dl:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

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

        # ── SCHEDULER (AFTER val_acc is computed) ─────────────
        scheduler.step(val_acc)

        # ── LOGGING ───────────────────────────────────────────
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        wandb.log({"train/loss": train_loss, "train/acc": train_acc,
                   "val/loss":   val_loss,   "val/acc":   val_acc,
                   "epoch": epoch+1})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model → {save_path}")

    wandb.finish()
    print(f"Best Val Acc: {best_val_acc:.4f}")
    return model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default="./oxford-iiit-pet")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--dropout_p",  type=float, default=0.5)
    p.add_argument("--save_path",  default="classifier.pth")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_classifier(args.data_root, args.epochs, args.lr,
                     args.batch_size, args.dropout_p, device, args.save_path)
