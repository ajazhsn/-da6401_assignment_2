import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from models.vgg11 import VGG11
from models.unet import VGG11UNet
from dataset import PetDataset


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (N, C, H, W), targets: (N, H, W)
        probs = F.softmax(logits, dim=1)
        C = logits.shape[1]
        targets_oh = F.one_hot(targets, C).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_oh).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


def dice_score(logits, targets):
    probs = torch.softmax(logits, dim=1).argmax(dim=1)
    C = logits.shape[1]
    score = 0.0
    for c in range(C):
        pred_c = (probs == c).float()
        tgt_c = (targets == c).float()
        inter = (pred_c * tgt_c).sum()
        score += (2 * inter + 1) / (pred_c.sum() + tgt_c.sum() + 1)
    return score / C


def train_unet(data_root: str, classifier_ckpt: str = "classifier.pth",
               epochs: int = 30, lr: float = 1e-4, batch_size: int = 16,
               freeze_encoder: bool = False, device: str = "cuda",
               save_path: str = "unet.pth", run_name: str = "unet-full-finetune"):
    wandb.init(project="da6401-a2", name=run_name)

    train_ds = PetDataset(data_root, split="train", task="segment")
    val_ds = PetDataset(data_root, split="val",   task="segment")
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=2)
    val_dl = DataLoader(val_ds,   batch_size=batch_size,
                        shuffle=False, num_workers=2)

    vgg = VGG11(num_classes=37)
    vgg.load_state_dict(torch.load(classifier_ckpt, map_location="cpu"))

    model = VGG11UNet(pretrained_vgg=vgg, num_classes=3,
                      freeze_encoder=freeze_encoder).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()

    best_dice = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n = 0
        for batch in train_dl:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = ce_loss(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        scheduler.step()

        model.eval()
        val_loss = 0
        val_dice = 0
        vn = 0
        with torch.no_grad():
            for batch in val_dl:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                loss = ce_loss(logits, masks) + dice_loss(logits, masks)
                val_loss += loss.item() * imgs.size(0)
                val_dice += dice_score(logits, masks).item() * imgs.size(0)
                vn += imgs.size(0)

        tl = train_loss/n
        vl = val_loss/vn
        vd = val_dice/vn
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {tl:.4f} | Val Loss: {vl:.4f} | Dice: {vd:.4f}")
        wandb.log({"train/seg_loss": tl, "val/seg_loss": vl,
                  "val/dice": vd, "epoch": epoch+1})

        if vd > best_dice:
            best_dice = vd
            torch.save(model.state_dict(), save_path)
    wandb.finish()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",       default="./oxford-iiit-pet")
    p.add_argument("--classifier_ckpt", default="classifier.pth")
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--freeze_encoder",  action="store_true")
    p.add_argument("--save_path",       default="unet.pth")
    p.add_argument("--run_name",        default="unet-full-finetune")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_unet(args.data_root, args.classifier_ckpt, args.epochs, args.lr,
               args.batch_size, args.freeze_encoder, device, args.save_path, args.run_name)
