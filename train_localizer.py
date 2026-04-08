import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from models.localization import VGG11Localizer
from losses.iou_loss import IoULoss
from dataset import PetDataset


def train_localizer(data_root, classifier_ckpt="checkpoints/classifier.pth",
                    epochs=60, lr=1e-4, batch_size=32,
                    device="cuda", save_path="checkpoints/localizer.pth"):
    wandb.init(project="da6401-a2", name="localizer-final")

    train_ds = PetDataset(data_root, split="train", task="localize")
    val_ds   = PetDataset(data_root, split="val",   task="localize")
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, num_workers=2)

    # Load pretrained encoder weights into localizer
    from models.classification import VGG11Classifier
    cls_model = VGG11Classifier(num_classes=37)
    cls_sd    = torch.load(classifier_ckpt, map_location="cpu")
    # Load only encoder weights
    enc_sd = {k[len("encoder."):]: v for k, v in cls_sd.items()
              if k.startswith("encoder.")}

    model = VGG11Localizer().to(device)
    if enc_sd:
        model.encoder.load_state_dict(enc_sd, strict=False)
        print("Loaded pretrained encoder into localizer")

    # Freeze early blocks
    for name, param in model.encoder.named_parameters():
        if name.startswith(("block1", "block2", "pool1", "pool2")):
            param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5)

    mse      = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")
    IMG      = 224.0
    best_iou = 0.0

    for epoch in range(epochs):
        model.train()
        tl, n = 0, 0
        for batch in train_dl:
            imgs   = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)
            mask   = bboxes.sum(1) > 0
            if mask.sum() == 0: continue
            imgs, bboxes = imgs[mask], bboxes[mask]
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = mse(preds/IMG, bboxes/IMG) + iou_loss(preds/IMG, bboxes/IMG)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * imgs.size(0); n += imgs.size(0)

        model.eval()
        vl, viou, vn = 0, 0, 0
        with torch.no_grad():
            for batch in val_dl:
                imgs   = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)
                mask   = bboxes.sum(1) > 0
                if mask.sum() == 0: continue
                imgs, bboxes = imgs[mask], bboxes[mask]
                preds = model(imgs)
                loss  = mse(preds/IMG, bboxes/IMG) + iou_loss(preds/IMG, bboxes/IMG)
                vl   += loss.item() * imgs.size(0)
                viou += (1 - iou_loss(preds/IMG, bboxes/IMG).item()) * imgs.size(0)
                vn   += imgs.size(0)

        vl /= max(vn,1); viou /= max(vn,1)
        scheduler.step(vl)
        print(f"Epoch {epoch+1}/{epochs} | Loss:{tl/max(n,1):.4f} | "
              f"Val Loss:{vl:.4f} | Val IoU:{viou:.4f}")
        wandb.log({"train/loc_loss": tl/max(n,1), "val/loc_loss": vl,
                   "val/iou": viou, "epoch": epoch+1})

        if viou > best_iou:
            best_iou = viou
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved (val_iou={viou:.4f})")

    wandb.finish()
    print(f"Best Val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",       default="./oxford-iiit-pet")
    p.add_argument("--classifier_ckpt", default="checkpoints/classifier.pth")
    p.add_argument("--epochs",          type=int,   default=60)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--save_path",       default="checkpoints/localizer.pth")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_localizer(args.data_root, args.classifier_ckpt, args.epochs,
                    args.lr, args.batch_size, device, args.save_path)
