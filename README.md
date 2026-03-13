# NET-VTE Prediction

Predicting venous thromboembolism (VTE) events by detecting neutrophil extracellular traps (NETs) in CellaVision peripheral blood smear data.

Two approaches are implemented:
1. **MIL Self-Attention** (`train_mil.py`): Multiple Instance Learning on pre-extracted 2048-dim cell embeddings (bags of ~114 WBCs per patient)
2. **Vision Transformer** (`train_vit.py`): ViT trained directly on cropped cell images (360x360 patches)

Both models perform binary classification: **before** vs **after** a VTE-associated clinical event.

> **Note:** This project did not yield clinically significant results but is published as a reference for the methodology.

## Data Format

### For MIL (`train_mil.py`)

Pre-extracted embeddings stored as pickle files, one per patient:

```
/path/to/data/
    train/
        before/
            patient_id.p    # pickle: dict keyed by cell index
            ...             #   each value: {'encoding': tensor(2048,), 'morphology': str}
        after/
            patient_id.p
            ...
    val/
        before/
            ...
        after/
            ...
```

### For ViT (`train_vit.py`)

Cropped cell images in PyTorch `ImageFolder` format:

```
/path/to/images/
    train/
        before/
            image1.jpg
            ...
        after/
            image1.jpg
            ...
    val/
        before/
            ...
        after/
            ...
```

## Usage

### MIL Self-Attention

```bash
python train_mil.py \
    --data_dir /path/to/data \
    --num_epochs 300 \
    --lr 1e-5 \
    --log_dir /path/to/logs \
    --save_dir /path/to/models
```

### Vision Transformer

```bash
python train_vit.py \
    --train_dir /path/to/images/train \
    --val_dir /path/to/images/val \
    --num_epochs 50 \
    --lr 1e-2 \
    --log_dir /path/to/logs \
    --save_dir /path/to/models
```

## Architecture

### MIL Self-Attention (`selfattnmodel.py`)
- **AggregateConcatenate**: Transforms cell embeddings and computes bag-level aggregations (mean, max, min, std), then concatenates them with the individual cell representations
- **MultiheadAttention**: Two stacked self-attention layers over the concatenated sequence
- **Classifier**: MLP head on the aggregation-head outputs

### Vision Transformer (`vit.py`)
- Standard ViT: patch embedding, learnable positional encoding, transformer encoder stack, CLS token classification
- 360x360 images with 12x12 patches = 900 patches per image

## Requirements

- PyTorch
- PyTorch Lightning
- torchmetrics
- torcheval
- einops
- torchsummary
