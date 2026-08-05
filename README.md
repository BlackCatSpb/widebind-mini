# WideBind Mini

**Тестовый полигон архитектуры WideBind** — не-трансформерный sequence model с самоорганизующимися системами.

D=448, L=6, G=8, ~4.5M параметров. Обучается на GPU с 2GB VRAM.

---

## Философия

WideBind строится на принципе **интеллект через самоорганизацию** — системы модели сами адаптируются к данным без ручного тюнинга.

Координата всех гиперпараметров — **λ_d ≈ 1.839** (трибоначчи):
- Все пороги, LR multipliers, init values выводятся из λ_d
- Масштабируется без изменения констант

---

## Архитектура

```
token IDs → SignedAmpCodec (sparse 6/32 + Gaussian readout) → [WideBindBlock × L] → logits
```

### WideBindBlock

```
h → RMSNorm → Conv1d(k=48) → TrajectorySpiralBind → VSA Memory → GroupedCognitiveMirror → DCT Spectral → GroupedMLP
```

### Ключевые компоненты

| Компонент | Описание | Адаптивность |
|---|---|---|
| **SignedAmpCodec** | Sparse binary codes + factorized Gaussian readout вместо softmax | σ ∝ uncertainty, gain ∝ 1/‖z‖ |
| **TrajectorySpiralBind** | Complex cross-mixing между траекториями слоёв (hp, VSA, mirror) | Phase rotation, soft EMA decay |
| **VSA Memory** | Multi-scale vector superposition (τ=8/32/128/512) | Per-channel decay, surprisal-gated write |
| **GroupedMirror** | 8 экспертов с predictive gating (temp/pred/smooth/sym/help) | Adaptive τ, gate per-token |
| **Collective Layer** | Zero-parameter concept bank с maturity gating | CV(resvar) maturity detection |
| **λ_d Hierarchy** | Все гиперпараметры из одного числа | Scale-invariant |

---

## TrajectorySpiralBind (ключевое нововведение)

Скрещивание не внутри слоя, а **между траекториями**:

```
u = hp_t · w_u
v = traj[d] · w_v          # traj = [hp_{t-1}, VSA_state, mirror_correction]
v_rot = v · exp(i·θ(τ))    # phase rotation
prod = u · v_rot           # complex multiplication
out = Σ_d [Re(prod); Im(prod)] @ W_out
```

**Преимущества:**
- Градиенты текут между слоями напрямую (не только через residual)
- 3 измерения с разными частотами — каждый ловит свой масштаб
- Soft EMA decay вместо hard detach

---

## Адаптивные механизмы

| Механизм | Триггер | Действие |
|---|---|---|
| Codec σ | EMA loss variance | σ растёт при неопределённости |
| Codec gain | ‖pre-activation‖ | gain ∝ 1/‖z‖ (anti-saturation) |
| Mirror scale | ‖delta‖ per expert | scale ∝ 1/(1+‖δ‖) |
| Trajectory state | EMA decay | 0.9·old + 0.1·new |
| LR schedule | λ_d-derived thresholds | Adaptive improvement detection |
| Collective maturity | CV(resvar) < 0.15 | Автозапуск concept bank |

---

## Запуск

```bash
# Базовый запуск (CUDA)
python train.py --data-dir ./wb --head codec --accum 4

# С private memory
python train.py --data-dir ./wb --head codec --accum 4 --private-mem

# CPU тест
python train.py --data-dir ./wb --head codec --accum 1 --device cpu
```

### Основные аргументы

| Аргумент | Default | Описание |
|---|---|---|
| `--D` | 448 | Размерность |
| `--n-layers` | 6 | Число слоёв |
| `--head` | codec | LM head (codec/partitioned/sigmoid_coded) |
| `--bind-twist-mode` | trajectory_spiral | off/shift/cascade/spiral/trajectory_spiral |
| `--private-mem` | — | Private memory + contradiction detection |
| `--accum` | 1 | Gradient accumulation |
| `--max-steps` | 20000 | Всего шагов |

---

## Статус обучения (step ~700)

```
step=605 loss=11.00 |1-a|=0.19 alpha_novelty=-0.015 mature=1.0
```

- Loss падает (11.39 → 11.00)
- Alpha адаптация работает (0.28 → 0.81)
- Collective layer активирован
- Все адаптивные механизмы функционируют

---

## Структура

```
├── train.py              # Training loop (FP32)
├── core/
│   ├── config.py         # WideBandConfig
│   ├── amp_codec.py      # SignedAmpCodec (adaptive σ/gain)
│   ├── bind.py           # TrajectorySpiralBind, SpiralBind, BottleneckBind
│   ├── block.py          # WideBindBlock (trajectory state management)
│   ├── mirror.py         # GroupedCognitiveMirror (adaptive scale)
│   ├── stack.py          # WideBindStack (adaptive LR schedule)
│   ├── concept_layer.py  # Collective concept bank
│   └── lambda_utils.py   # λ_d hierarchy
├── wb/                   # Token streams (gitignored)
└── checkpoints_codec/    # Checkpoints (gitignored)
```

---

## Отличия от Transformer

| Аспект | Transformer | WideBind |
|---|---|---|
| Token representation | Dense embedding | Sparse binary codes |
| Sequence mixing | Softmax attention | Bilinear twist + VSA |
| Memory | None | Multi-scale VSA |
| Expert gating | None/MoE router | Sigmoid + self-consistency |
| Hyperparameters | Many independent | λ_d-derived |
