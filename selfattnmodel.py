from torch.nn import Sequential, Linear, ELU, Tanh, Sigmoid, Softmax, Dropout, Flatten
import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
import schedulers


class AggregateConcatenate(nn.Module):
    """Aggregates cell embeddings using multiple statistics (mean, max, min, std),
    then concatenates the bag-level aggregations with per-cell transformed embeddings."""

    def __init__(self, init_embed_size, hidden_size, agg_out_size, agg_method='normal', dropout=0.2):
        super().__init__()

        agg_dict = {'normal': 4, 'gm': 3, 'lse': 2}
        self.num_heads = agg_dict[agg_method]
        self.agg_method = agg_method

        self.aggregators = Sequential(
            Flatten(start_dim=0, end_dim=1),
            Linear(init_embed_size, hidden_size),
            ELU(),
            Dropout(dropout),
            Linear(hidden_size, agg_out_size),
            Tanh()
        )

        self.adj_input_network = Sequential(
            Flatten(start_dim=0, end_dim=1),
            Linear(init_embed_size, hidden_size),
            ELU(),
            Dropout(dropout),
            Linear(hidden_size, agg_out_size),
            Tanh()
        )

    def normal_aggregation(self, queries, padding_lengths):
        batches = queries.shape[0]
        batch_representations = []

        for b in range(batches):
            non_padded_queries = queries[b, :padding_lengths[b], :]
            bag_representation = [
                torch.mean(non_padded_queries, dim=0),
                torch.max(non_padded_queries, dim=0).values,
                torch.min(non_padded_queries, dim=0).values,
                torch.std(non_padded_queries, dim=0),
            ]
            batch_representations.append(torch.stack(bag_representation))

        return torch.stack(batch_representations)  # B x n x agg_out_size

    def gm_aggregation(self, batch_aggregators, padding_lengths):
        batch_size = batch_aggregators.shape[0]
        batch_representations = []

        for bag in range(batch_size):
            bag_non_padded_aggs = batch_aggregators[bag, :padding_lengths[bag], :]
            bag_representation = [
                self.generalized_mean(bag_non_padded_aggs, power=5.0),
                self.generalized_mean(bag_non_padded_aggs, power=2.5),
                self.generalized_mean(bag_non_padded_aggs, power=1.0),
            ]
            batch_representations.append(torch.stack(bag_representation))

        return torch.stack(batch_representations)

    def lse_aggregation(self, queries, padding_lengths):
        batch_size = queries.shape[0]
        batch_representations = []

        for bag in range(batch_size):
            non_padded_queries = queries[bag, :padding_lengths[bag], :]
            bag_representation = [
                self.log_sum_exponentiation(non_padded_queries, power=2.5),
                self.log_sum_exponentiation(non_padded_queries, power=1.0),
                self.log_sum_exponentiation(non_padded_queries, power=0.5),
            ]
            batch_representations.append(torch.stack(bag_representation))

        return torch.stack(batch_representations)

    def generalized_mean(self, inp_tensor, power=1.0):
        n = inp_tensor.shape[0]
        return (1/n * torch.sum(inp_tensor**power, dim=0))**(1/power)

    def log_sum_exponentiation(self, inp_tensor, power=1.0):
        n = inp_tensor.shape[0]
        sum_exp = (1/n) * torch.sum(torch.exp(power*inp_tensor), dim=0)
        return (1/power) * torch.log(sum_exp)

    def forward(self, x, padding_lengths):
        # x: B x T_max x E
        B = x.shape[0]
        T_max = x.shape[1]

        aggregations = self.aggregators(x).view(B, T_max, -1)  # B x T_max x agg_out_size

        if self.agg_method == 'normal':
            bag_aggregations = self.normal_aggregation(aggregations, padding_lengths)
        elif self.agg_method == 'gm':
            bag_aggregations = self.gm_aggregation(aggregations, padding_lengths)
        elif self.agg_method == 'lse':
            bag_aggregations = self.lse_aggregation(aggregations, padding_lengths)

        adjacent_inputs = self.adj_input_network(x).view(B, T_max, -1)

        # B x (n+T_max) x agg_out_size
        return torch.cat((bag_aggregations, adjacent_inputs), dim=1)


class MultiheadAttention(nn.Module):
    """Self-attention layer with separate key/query/value projections."""

    def __init__(self, inp_embedding_size, attn_head_size, num_attn_heads, n, dropout=0.3):
        super().__init__()
        self.num_agg_heads = n

        self.key = Sequential(
            Linear(inp_embedding_size, 128), ELU(), Dropout(dropout),
            Linear(128, attn_head_size), Tanh()
        )
        self.query = Sequential(
            Linear(inp_embedding_size, 128), ELU(), Dropout(dropout),
            Linear(128, attn_head_size), Sigmoid()
        )
        self.value = Sequential(
            Linear(inp_embedding_size, 128), ELU(), Dropout(dropout),
            Linear(128, attn_head_size),
        )

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=attn_head_size, num_heads=num_attn_heads,
            dropout=dropout, batch_first=True
        )

    def forward(self, input_tensor):
        # input_tensor: B x L x E, where L = n + T_max
        key = self.key(input_tensor)
        query = self.query(input_tensor)
        value = self.value(input_tensor)

        full_output, attn = self.multihead_attention(query, key, value)

        # Extract attention from aggregation heads to cell positions
        head_attn = attn[:, :self.num_agg_heads, self.num_agg_heads:]  # B x n x T_max
        head_output = full_output[:, :self.num_agg_heads, :]  # B x n x outsize

        return head_output, full_output, attn, head_attn


class MILSelfAttention(pl.LightningModule):
    """MIL model with aggregate-concatenate and two stacked self-attention layers."""

    def __init__(self, init_mil_embed, mil_head, num_classes, attn_head_size,
                 agg_method='normal', step_up_size=None, T_min=None, T_max=None, config=None):
        super().__init__()
        task = "binary" if num_classes == 2 else "multiclass"
        self.train_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.val_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()

        agg_dict = {'normal': 4, 'gm': 3, 'lse': 3}
        self.outsize = attn_head_size
        self.num_agg_heads = agg_dict[agg_method]

        self.aggregation = AggregateConcatenate(
            init_mil_embed, 256, mil_head, agg_method, dropout=0.2
        )
        self.attention1 = MultiheadAttention(
            inp_embedding_size=mil_head, attn_head_size=attn_head_size,
            num_attn_heads=1, n=self.num_agg_heads, dropout=0.2
        )
        self.attention2 = MultiheadAttention(
            inp_embedding_size=attn_head_size, attn_head_size=attn_head_size,
            num_attn_heads=1, n=self.num_agg_heads, dropout=0.2
        )

        self.classifier = Sequential(
            Linear(self.num_agg_heads * attn_head_size, 128),
            ELU(), Dropout(0.3),
            Linear(128, 64),
            ELU(), Dropout(0.3),
            Linear(64, num_classes),
            Softmax(dim=1)
        )

    def forward(self, x, padding_lengths):
        # x: B x T_max x E
        concatenated = self.aggregation(x, padding_lengths)
        _, full_out1, _, _ = self.attention1(concatenated)
        head_out2, _, _, head_attn2 = self.attention2(full_out1)

        head_out2 = torch.flatten(head_out2, start_dim=1)  # B x (n*outsize)
        pred_probs = self.classifier(head_out2)  # B x n_classes

        return pred_probs, head_attn2  # B x n_classes, B x n x T_max

    def training_step(self, batch, batch_idx):
        x, y, p, filename = batch
        y_hat, attn_scores = self(x, p)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        class_prediction = torch.tensor([torch.argmax(y_hat)]).to('cpu')
        y = y.to('cpu')
        self.train_accuracy(class_prediction, y)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, p, filename = batch
        y_hat, attn_scores = self(x, p)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        class_prediction = torch.tensor([torch.argmax(y_hat)]).to('cpu')
        y = y.to('cpu')
        self.val_accuracy(class_prediction, y)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        initial_lr = self.config["lr"]
        optimizer = torch.optim.AdamW(self.parameters(), lr=initial_lr)
        scheduler_config = {'lr': initial_lr, 'milestones': [500]}
        scheduler = schedulers.Scheduler(scheduler_config).cosine_flat(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
