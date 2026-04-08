import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from models.segmentation import VGG11UNet
from dataset import PetDataset


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        C      = logits.shape[1]
        probs  = F.softmax(logits, dim=1)
        t_oh   = F.one_hot(targets, C).permute(0,3,1,2).float()
        inter  = (probs * t_oh).sum(dim=(2,3))
        union  = probs.sum(dim=(2,3)) + t_oh.sum(dim=(2,3))
        return 1.0 - ((2*inter + self.smooth) / (union + self.smooth)).mean()


def dice_score(logits, targets):
    C = logits.shape[1]; preds = logits.argmax(1); s = 0.0
    for c in range(C):
        pc = (preds==c).float(); tc = (targets==c).float()
        s += (2*(pc*tc).sum()+1) / (pc.sum()+tc.sum()+1)
    return s / C


def train_unet(data_root, classifier_ckpt="checkpoints/classifier.pth",
               epochs=30, lr=1e-4, batch_size=16, freeze_encoder=False,
               freeze_blocks=0, device="cuda",
               save_path="checkpoints/unet.pth",
               run_name="unet-full-finetune"):
    wandb.init(project="da6401-a2", name=run_name)

    train_ds = PetDataset(data_root, split="train", task="segment")
    val_ds   = PetDataset(data_root, split="val",   task="segment")
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

    model = VGG11UNet(num_classes=3).to(device)

    # Load pretrained encoder
    cls_sd  = torch.load(classifier_ckpt, map_location="cpu")
    enc_sd  = {k[len("encoder."):]: v for k, v in cls_sd.items()
               if k.startswith("encoder.")}
    if enc_sd:
        model.encoder.load_state_dict(enc_sd, strict=False)
        print("Loaded pretrained encoder into UNet")

    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    elif freeze_blocks > 0:
        blocks = [f"block{i}" for i in range(1, freeze_blocks+1)]
        for n, p in model.encoder.named_parameters():
            if any(n.startswith(b) for b in blocks):
                p.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    ce   = nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.0,2.0]).to(device))
    dice = DiceLoss()
    best = 0.0

    for epoch in range(epochs):
        model.train()
        tl, n = 0, 0
        for batch in train_dl:
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = ce(logits, masks) + dice(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * imgs.size(0); n += imgs.size(0)
        scheduler.step()

        model.eval()
        vl, vd, vn = 0, 0, 0
        with torch.no_grad():
            for batch in val_dl:
                imgs  = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                vl += (ce(logits,masks)+dice(logits,masks)).item()*imgs.size(0)
                vd += dice_score(logits, masks).item() * imgs.size(0)
                vn += imgs.size(0)
        vl/=vn; vd/=vn

        print(f"Epoch {epoch+1}/{epochs} | Loss:{tl/n:.4f} | "
              f"Val Loss:{vl:.4f} | Dice:{vd:.4f}")
        wandb.log({"train/seg_loss":tl/n, "val/seg_loss":vl,
                   "val/dice":vd, "epoch":epoch+1})

        if vd > best:
            best = vd
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved (dice={vd:.4f})")

    wandb.finish()
    print(f"Best Dice: {best:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",       default="./oxford-iiit-pet")
    p.add_argument("--classifier_ckpt", default="checkpoints/classifier.pth")
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--freeze_encoder",  action="store_true")
    p.add_argument("--freeze_blocks",   type=int,   default=0)
    p.add_argument("--save_path",       default="checkpoints/unet.pth")
    p.add_argument("--run_name",        default="unet-full-finetune")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_unet(args.data_root, args.classifier_ckpt, args.epochs, args.lr,
               args.batch_size, args.freeze_encoder, args.freeze_blocks,
               device, args.save_path, args.run_name)
