# WideBind Mini

Compact local training sandbox for [WideBind](https://github.com/BlackCatSpb/widebind) — a hybrid D-space LM with VSA long-range memory and bottleneck bind.

**Purpose:** validate architectural ideas (gradient accumulation, soft EOS reset, FP32 stability, NaN-free VSA scan) on a local GPU before scaling to the full D=4096 model on Colab.

## Configuration

| Param | Default | Description |
|---|---|---|
| `D` | 896 | Model dimension |
| `n_layers` | 12 | Transformer layers |
| `bind_K` | 32 | Bind dictionary size |
| `mlp_groups` | 8 | Expert groups (G) |
| `code_dim` | 32 | Bind code dimension (d) |
| `vocab` | 50000 | Vocabulary size |
| `seq_len` | 512 | Context length |
| `batch_size` | 2 | Micro-batch size |
| `params` | **~10.85M** | Total parameters |

Fits **2 GB VRAM** (MX550 tested) in pure FP32 with `batch_size=2`.

## Quick Start

```bash
# Train with gradient accumulation (effective batch = 1024×8 = 8192)
python train.py --data-dir ./wb --accum 8

# Train with default settings
python train.py --data-dir ./wb

# Custom architecture
python train.py --data-dir ./wb --D 1024 --n-layers 16 --mlp-groups 12
```

## Data

Expects `token_stream_*.bin` files (uint16 numpy memmap) in `--data-dir`. Three streams included:
- ADVENTUR (~2.1B tokens)
- DRAMA (~2.1B tokens)
- FANTASY (~2.1B tokens)

Total: **~6.3B tokens**.

## Key Features

- **Pure FP32 training** — no AMP, no autocast, no GradScaler
- **Gradient accumulation** — `--accum N` for effective batch = 1024 × N
- **Chunked VSA scan** (CHUNK=32) — avoids `exp(-log_cum)` overflow that caused NaN
- **Soft EOS reset** — `_soft_reset(state, factor=0.3)` decays state at sequence boundaries
- **Bidirectional MirrorLR** — learning rate adapts to gate/scale dynamics
- **Checkpoint resume** — auto-loads latest `step_*.pt` on restart
- **Interrupt save** — Ctrl+C saves `interrupt_step_N.pt`

## Project Structure

```
├── train.py              # Training script (FP32, no AMP)
├── core/
│   ├── config.py         # Compact WideBandConfig (D=896, L=12, G=8)
│   ├── model.py          # WideBindStack, VSA scan, mirror, bind
│   ├── lambda_utils.py   # λ-d hierarchy helpers
│   └── zeckendorf_readout.py
├── compression/
│   ├── __init__.py
│   └── fcf_cpr.py        # Compression utilities
├── checkpoints/          # Saved model checkpoints
├── wb/                   # Token streams (gitignored)
└── README.md
```

## Training Metrics

Example log line:
```
step=   220 loss=10.5267 |1-a|=0.1142 g_var=0.0001 ls_var=0.0077 lr=8.02e-05 tok/s=45 mem=5.67GB
```

| Metric | Meaning |
|---|---|
| `loss` | Cross-entropy loss |
| `\|1-a\|` | Mean |1 − α_diag| across layers (specialization) |
| `g_var` | Gate variance across experts |
| `ls_var` | Variance of log_scale across layers |
| `lr` | Current learning rate |
| `tok/s` | Tokens per second |
| `mem` | GPU memory usage (GB) |

## Status

Stable training confirmed on NVIDIA GeForce MX550 (2.1 GB VRAM). No NaN, stable gradients, correct gradient accumulation.

Next step: port accumulation + soft reset to full D=4096 model in `notebooks/colab.ipynb`.
