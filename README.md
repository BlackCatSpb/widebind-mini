# WideBind Mini

Compact local training sandbox for [WideBind](https://github.com/BlackCatSpb/widebind). Validates architectural ideas on MX550 before scaling to full D=4096.

## Configuration

| Param | Default | Description |
|---|---|---|
| `D` | 896 | Model dimension |
| `n_layers` | 12 | Number of layers |
| `bind_K` | 32 | Bind bottleneck size |
| `mlp_groups` | 8 | Expert groups (G) |
| `params` | **~11.2M** | Total parameters |
| `bind_twist_mode` | shift | BottleneckBind mode (off/shift/cascade) |

Fits **2 GB VRAM** (MX550) in pure FP32, `batch_size=2`, `accum=8` (effective batch 8192).

## Quick Start

```bash
python train.py --data-dir ./wb --accum 8
```

## Key Features

- **Pure FP32** — no AMP, no autocast, no GradScaler
- **BottleneckBind** — Fibonacci-twisted (shift S=4 by default)
- **Chunked VSA scan** (CHUNK=32) — NaN-free
- **Gradient accumulation** (`--accum N`)
- **Soft EOS reset** — state *= 0.3 at EOS boundaries
- **Bidirectional MirrorLR** — adaptive LR
- **Resume** — auto-loads latest `step_*.pt`
- **CTRL+C** — saves `step_N.pt` for seamless resume
- **Eval checkpoint** — saves `eval_N.pt` on every eval

## Data

`token_stream_*.bin` (uint16 memmap) in `--data-dir`:
- ADVENTUR (~2.1B tokens)
- DRAMA (~2.1B tokens)
- FANTASY (~2.1B tokens)

Total: **~6.3B tokens**.

## Training Metrics

```
step=   110 loss=10.4639 |1-a|=0.1127 g_var=0.0039 ls_var=0.0077 lr=2.17e-04 tok/s=56 mem=5.68GB
```

| Metric | Meaning |
|---|---|
| `loss` | Cross-entropy loss |
| `\|1-a\|` | Mean |1 − α_diag| across layers |
| `g_var` | Gate variance across experts |
| `ls_var` | Variance of log_scale |
| `lr` | Current learning rate |
| `tok/s` | Tokens per second |
| `mem` | GPU memory (GB) |

## Status

Stable training on MX550 (2.1 GB VRAM). No NaN, stable gradients. Shift mode active.
