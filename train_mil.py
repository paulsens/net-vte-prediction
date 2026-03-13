"""Train MIL Self-Attention model on pre-extracted cell embeddings for NET/VTE prediction."""

import os
import argparse
import pickle
import random
import sys

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchmetrics.classification import BinaryF1Score, BinaryAUROC, AUROC, F1Score

import selfattnmodel


class TensorDataset(Dataset):
    def __init__(self, data_path, split, T_min, T_max):
        super().__init__()
        self.data_files = []
        self.labels = []
        self.T_min = T_min
        self.T_max = T_max

        label_dirs = ["before", "after"]
        label_map = {'before': 0, 'after': 1}

        data_path = os.path.join(data_path, split)

        for label_dir in label_dirs:
            label_path = os.path.join(data_path, label_dir)
            files = []
            print(f"{label_path} has {len(os.listdir(label_path))} files")
            for file in os.listdir(label_path):
                if not file.endswith(".p"):
                    continue
                files.append(os.path.join(label_path, file))

            self.data_files.extend(files)
            self.labels.extend([label_map[label_dir]] * len(files))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        with open(self.data_files[idx], "rb") as f:
            encodings_dict = pickle.load(f)

        if isinstance(encodings_dict, int) or len(encodings_dict) == 0:
            print(f"Bad data at {self.data_files[idx]}")
            sys.exit(1)

        temp_encoding = encodings_dict[0]['encoding']
        embed_dim = len(temp_encoding)
        label = self.labels[idx]

        num_instances = np.random.randint(self.T_min, self.T_max)
        padding_length = num_instances
        num_pads = self.T_max - padding_length

        if num_instances > len(encodings_dict):
            print(f"Warning: {self.data_files[idx]} has only {len(encodings_dict)} cells, need {num_instances}")

        instance_idxs = random.sample(range(len(encodings_dict)), k=num_instances)

        padding = torch.zeros(embed_dim)
        X = []
        for i in range(num_instances):
            X.append(encodings_dict[instance_idxs[i]]['encoding'])
        for _ in range(num_pads):
            X.append(padding)

        return torch.stack(X), label, padding_length, self.data_files[idx]


class TensorDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, T_min, T_max):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.T_min = T_min
        self.T_max = T_max

    def setup(self, stage=None):
        self.train_dataset = TensorDataset(self.data_dir, split="train",
                                           T_min=self.T_min, T_max=self.T_max)
        self.val_dataset = TensorDataset(self.data_dir, split="val",
                                         T_min=self.T_min, T_max=self.T_max)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, persistent_workers=True)


def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_module = TensorDataModule(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, T_min=args.t_min, T_max=args.t_max
    )
    data_module.setup()

    config = {"lr": args.lr}
    model = selfattnmodel.MILSelfAttention(
        init_mil_embed=args.embed_dim, mil_head=args.mil_head,
        num_classes=args.num_classes, attn_head_size=args.attn_head,
        agg_method=args.agg_method, T_min=args.t_min, T_max=args.t_max,
        config=config
    )

    logger = TensorBoardLogger(args.log_dir)
    trainer = pl.Trainer(
        max_epochs=args.num_epochs, logger=logger,
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=args.num_gpus,
        callbacks=[
            ModelCheckpoint(save_weights_only=False, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
        ]
    )
    trainer.fit(model, data_module)

    # Load best checkpoint
    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model = model.cpu()

    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluate on validation set
    val_loader = data_module.val_dataloader()
    val_preds, val_labels = [], []
    attention_dict = {0: [], 1: []}

    for step, (x, y, padding_length, filepath) in enumerate(val_loader):
        x, y = x.cpu(), y.cpu()
        prediction, attn = model(x, padding_length)
        prediction_init = prediction[0]
        prediction_class = torch.argmax(prediction_init).item()

        val_preds.append(prediction_class)
        y_val = y[0].item()
        attention_dict[y_val].append((padding_length, prediction_class, attn, filepath, x))
        val_labels.append(y_val)

    # Save attention dict
    attn_path = os.path.join(args.save_dir, "attention_dict.p")
    with open(attn_path, "wb") as f:
        pickle.dump(attention_dict, f)

    # Compute metrics
    val_preds_t = torch.Tensor(val_preds).to(device)
    val_labels_t = torch.Tensor(val_labels).long().to(device)

    f1metric = BinaryF1Score().to(device)
    auroc_metric = BinaryAUROC(thresholds=200).to(device)

    f1metric.update(val_preds_t, val_labels_t)
    auroc_metric.update(val_preds_t, val_labels_t)

    print(f"F1 Score: {f1metric.compute()}")
    print(f"AUROC: {auroc_metric.compute()}")

    return model


def find_empty_directories(directory):
    empty_dirs = []
    for root, dirs, files in os.walk(directory, topdown=False):
        if not os.listdir(root):
            empty_dirs.append(root)
    return empty_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MIL Self-Attention model for NET/VTE prediction")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to data directory containing train/ and val/ subdirectories")
    parser.add_argument("--log_dir", type=str, default="./lightning_logs",
                        help="TensorBoard log directory")
    parser.add_argument("--save_dir", type=str, default="./saved_models",
                        help="Directory to save trained model and attention weights")
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=2048,
                        help="Dimension of input cell embeddings")
    parser.add_argument("--mil_head", type=int, default=256,
                        help="Output size of the aggregation network")
    parser.add_argument("--attn_head", type=int, default=128,
                        help="Output size of the attention network")
    parser.add_argument("--agg_method", type=str, default="normal",
                        choices=["normal", "gm", "lse"])
    parser.add_argument("--t_min", type=int, default=114,
                        help="Minimum number of instances per bag")
    parser.add_argument("--t_max", type=int, default=115,
                        help="Maximum number of instances per bag (bags padded to this length)")
    args = parser.parse_args()

    empty_dirs = find_empty_directories(args.data_dir)
    assert not empty_dirs, f"Empty directories found: {empty_dirs}"

    torch.multiprocessing.set_start_method("spawn")
    train_model(args)
