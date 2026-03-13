from torch.nn import Sequential, Linear, Dropout
import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics import Accuracy
from einops.layers.torch import Rearrange
from torcheval.metrics import BinaryConfusionMatrix
import schedulers


class AttentionBlock(nn.Module):
    """Transformer encoder block: LayerNorm -> MultiheadAttention -> LayerNorm -> FFN."""

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        attn_output, _ = self.attn(inp_x, inp_x, inp_x)
        x = x + attn_output
        x = x + self.linear(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):
    """Standard ViT: patch embedding, learnable positional encoding, transformer stack, CLS token."""

    def __init__(self, lr, init_img_size, init_patch_size, embed_dim, hidden_dim,
                 num_heads, num_layers, num_classes, dropout=0.0):
        super().__init__()

        self.image_height = init_img_size
        self.image_width = init_img_size
        self.patch_height = init_patch_size
        self.patch_width = init_patch_size
        self.num_classes = num_classes
        self.num_patches = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)
        self.embed_dim = embed_dim
        self.channels = 3

        self.transformer = nn.Sequential(
            *[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout)
              for _ in range(num_layers)]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.patch_dim = self.channels * self.patch_height * self.patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                      p1=self.patch_height, p2=self.patch_width),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        B, channels, _, _ = x.shape
        x = self.to_patch_embedding(x)

        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x += self.pos_embedding

        x = self.transformer(x)

        cls = x[0][0].reshape(B, self.embed_dim)
        return self.mlp_head(cls)


class Seanformer(pl.LightningModule):
    """PyTorch Lightning wrapper for VisionTransformer with training/validation logic."""

    def __init__(self, lr, model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(lr, **model_kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.epoch_n = 0
        self.lr = lr

        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.val_confusion = BinaryConfusionMatrix()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler_config = {'lr': self.lr, 'milestones': [1000]}
        self.scheduler = schedulers.Scheduler(scheduler_config).cosine_flat(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": self.scheduler}

    def _calculate_loss(self, batch, mode=None):
        imgs, labels = batch
        preds = self.model(imgs)
        y_hat = preds.argmax(dim=-1)
        loss = self.criterion(preds, labels)
        self.log(f'{mode}_loss', loss, sync_dist=True)

        if mode == "train":
            batch_value = self.train_accuracy(y_hat, labels)
            self.log("train_acc_step", batch_value)
        elif mode == "val":
            self.val_accuracy.update(y_hat, labels)
            self.val_confusion.update(y_hat, labels)

        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", self.val_accuracy.compute(), sync_dist=True)
        self.val_accuracy.reset()
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", current_lr, on_epoch=True, sync_dist=True)
        print(f"VAL CONFUSION MATRIX FOR EPOCH {self.epoch_n}: {self.val_confusion.compute()}")
        self.val_confusion.reset()
        self.epoch_n += 1
