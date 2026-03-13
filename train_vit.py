"""Train Vision Transformer on raw cell images for NET/VTE prediction."""

import os
import argparse

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms

from vit import Seanformer


def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transforms = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])
    val_transforms = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    train_dataset = datasets.ImageFolder(root=args.train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=args.val_dir, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )

    checkpoint_dir = os.path.join(args.save_dir, "checkpoints")

    trainer = pl.Trainer(
        default_root_dir=checkpoint_dir,
        logger=TensorBoardLogger(args.log_dir),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=args.num_gpus,
        max_epochs=args.num_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_epoch"),
            LearningRateMonitor("epoch"),
        ]
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    pretrained_filename = os.path.join(checkpoint_dir, "seanformer.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = Seanformer.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = Seanformer(args.lr, model_kwargs={
            'init_img_size': args.image_size,
            'init_patch_size': args.patch_size,
            'embed_dim': args.embed_dim,
            'hidden_dim': args.hidden_dim,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'num_classes': args.num_classes,
            'dropout': args.dropout,
        })
        trainer.fit(model, train_loader, val_loader)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision Transformer for NET/VTE prediction")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Path to training images (ImageFolder format: train_dir/class/*.jpg)")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Path to validation images (ImageFolder format)")
    parser.add_argument("--log_dir", type=str, default="./lightning_logs")
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=360,
                        help="Height and width of input images")
    parser.add_argument("--patch_size", type=int, default=12)
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    torch.multiprocessing.set_start_method("spawn")
    train_model(args)
