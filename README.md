# WideBind Mini

**11.2M параметров, 12 слоёв, D=896.** Локальный тренировочный полигон для архитектуры WideBind. Те же компоненты, что и в полной модели (D=4096, 161M), но в масштабе, влезающем в 2 GB VRAM.

## Архитектура

```
token IDs → PartitionedEmbedding (8×112, sparse 6/32) → [WideBindBlock × 12] → Final RMS Norm → PartitionedHead → logits
```

Каждый блок:

```
h → RMSNorm → Conv1d(groups=D, k=48) → BottleneckBind(D→K=32→D) → VSA Memory (chunked scan) → GroupedCognitiveMirror (8 экспертов) → DCT Spectral → GroupedMLP (8 групп, ×4)
```

**BottleneckBind** — скрещивание размерностей через K=32 с Фибоначчи-твистом. Три режима (`--bind-twist-mode`):
- `off` — `(hp·w_u) ⊙ (hp·w_v) @ W_out`, классическая регрессия
- `shift` (по умолчанию) — сумма S билинейных произведений с golden-ratio сдвигом по K-пространству
- `cascade` — Фибоначчи-вложенные моночлены

`tie_bind=True`: W_out = W_proj^T (автоэнкодер). w_u/v инициализируются с std=1.0 (std³ критичен).

**VSA Memory** — векторная суперпозиция с chunked prefix scan (CHUNK=32), surprisal-gated i_gate, dual readout + first moment. fp32 guard для численной стабильности. Per-channel τ до ~163K (b_d=12.0).

**GroupedCognitiveMirror** — 8 экспертов, каждый в своём d=112, с 4 EMA-нормированными сигналами (temp/pred/smooth/sym), learnable softmax-весами, K-space gate. α — скаляр per expert.

**GroupedMLP** — 8 групп × (112→448→112, SiLU). 79.6% параметров.

## Параметры (tied, D=896, L=12)

| Компонент | Параметров | % |
|---|---|---|
| Embed + LM Head | 8,192 | 0.07 |
| BottleneckBind (K=32) | 28,736 | 0.26 |
| GroupedCognitiveMirror | 62,432 | 0.56 |
| Conv1d (k=48) | 43,008 | 0.38 |
| DCT Spectral | 10,752 | 0.10 |
| VSA gates | 215,040 | 1.92 |
| GroupedMLP (expand=4) | 8,912,896 | 79.56 |
| **Total** | **~11.2M** | **100** |

## Тренировка

### Запуск

```bash
python train.py --data-dir ./wb --accum 8
```

Ключевые аргументы:

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--D` | 896 | Размерность модели |
| `--n-layers` | 12 | Число слоёв |
| `--accum` | 1 | Градиентная аккумуляция |
| `--batch-size` | 2 | Микро-батч |
| `--bind-twist-mode` | shift | Режим BottleneckBind |
| `--max-steps` | 300K | Всего шагов |
| `--eval-interval` | 500 | Каждый N-й шаг — eval |
| `--save-interval` | 2000 | Сохранение чекпоинта |

### Данные

Токен-потоки (uint16 memmap) в `--data-dir`:
- ADVENTUR (~2.1B токенов)
- DRAMA (~2.1B токенов)  
- FANTASY (~2.1B токенов)

Всего: **~6.3B токенов**. Циклический перебор потоков, мягкий сброс состояния на EOS.

### Оптимизация

- **AdamW** (0.9, 0.95), LR=3e-4, weight_decay=0.01
- **Gradient accumulation**: effective batch = 1024 × N
- **Bidirectional MirrorLR** — LR растёт с var(log_scale), |1-α|, gate_var. Без forced cosine decay
- **Pure FP32** — никакого AMP, autocast или GradScaler

### Чекпоинты

- `best.pt` — лучший по val_loss
- `step_{N}.pt` — каждый save_interval + по Ctrl+C (авто-resume)
- `eval_{N}.pt` — каждый eval
- Resume: автоматически подхватывает последний `step_*.pt`

### Loss

| Компонент | Вес |
|---|---|
| CE (PAD/EOS замаскирован) | 1.0 |
| pred_loss (K-space) | adaptive (0.05–1.0) |
| gate_l1 | 0.001 |
| reinforce | 0.01 |
| balance | 0.01 |
| diversity | 0.001 |
| nuclear / orth | 1e-5 / 1e-4 |

## Метрики тренировки

```
step=   220 loss=9.4155 |1-a|=0.1127 g_var=0.0039 ls_var=0.0077 lr=2.58e-04 tok/s=56 mem=5.68GB
```

| Метрика | Что значит |
|---|---|
| `loss` | Cross-entropy loss |
| `\|1-a\|` | Среднее \|1 − α_diag\| по слоям (специализация mirror) |
| `g_var` | Дисперсия гейтов по экспертам |
| `ls_var` | Дисперсия log_scale mirror |
| `lr` | Текущий learning rate |
| `tok/s` | Токенов в секунду |
| `mem` | Использование VRAM (GB) |

## Структура проекта

```
├── train.py              # Тренировочный скрипт (FP32)
├── core/
│   ├── config.py         # WideBandConfig (D=896, L=12, G=8)
│   ├── model.py          # WideBindStack, VSA scan, BottleneckBind, mirror, MLP
│   ├── lambda_utils.py   # λ-d иерархия
│   └── zeckendorf_readout.py
├── compression/          # FCF-CPR (сжатие чекпоинтов)
├── checkpoints/          # Чекпоинты (gitignored)
├── wb/                   # Токен-потоки (gitignored)
└── README.md
```

## Статус

Стабильная тренировка на MX550 (2.1 GB VRAM). Чистый FP32, без NaN, градиентная аккумуляция работает, soft EOS reset корректен. Active режим: shift (S=4 по умолчанию, но при старте с конфигом по умолчанию S=1 — уточнить в `--bind-twist-S`).
