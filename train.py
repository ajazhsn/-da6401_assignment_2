"""
Unified training script for DA6401 Assignment 2.
"""
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        choices=["classify", "localize", "segment"])
    parser.add_argument("--data_root",       default="./oxford-iiit-pet")
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--dropout_p",       type=float, default=0.4)
    parser.add_argument("--classifier_ckpt",
                        default="checkpoints/classifier.pth")
    parser.add_argument("--freeze_encoder",  action="store_true")
    parser.add_argument("--freeze_blocks",   type=int,   default=0)
    parser.add_argument("--save_path",       default="checkpoints/model.pth")
    parser.add_argument("--run_name",        default="training-run")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Task: {args.task}")

    if args.task == "classify":
        from train_classifier import train_classifier
        train_classifier(args.data_root, args.epochs, args.lr,
                         args.batch_size, args.dropout_p, device, args.save_path)

    elif args.task == "localize":
        from train_localizer import train_localizer
        train_localizer(args.data_root, args.classifier_ckpt, args.epochs,
                        args.lr, args.batch_size, device, args.save_path)

    elif args.task == "segment":
        from train_unet import train_unet
        train_unet(args.data_root, args.classifier_ckpt, args.epochs,
                   args.lr, args.batch_size, args.freeze_encoder,
                   args.freeze_blocks, device, args.save_path, args.run_name)


if __name__ == "__main__":
    main()
