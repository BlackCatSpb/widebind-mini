# WideBind Mini — Экспериментальный полигон

**Полигон для обкатки новых методов перед переносом в основную модель (WideBind).**

## Назначение

- Тестирование новых архитектурных решений на маленькой модели
- Ablation studies
- Быстрые эксперименты (D=896, L=12, ~4.5M params)

## Архитектура

Аналогична WideBind, но меньшего масштаба:

```
WideBind Mini: D=896, L=12, G=8, bind_K=32
WideBind:      D=2560, L=24, G=32, bind_K=32
```

## Структура проекта

```
WideBind Mini/
├── core/                    # Основной код
│   ├── config.py            # WideBandConfig
│   ├── block.py             # WideBindBlock
│   ├── stack.py             # WideBindStack
│   ├── bind.py              # TrajectorySpiralBind, SpiralBind, BottleneckBind
│   ├── mirror.py            # GroupedCognitiveMirror
│   ├── embedding.py         # PartitionedEmbedding, PartitionedHead
│   ├── sigmoid_head.py      # SigmoidCodedHead
│   ├── cognitive_head.py    # CognitiveCodedHead
│   ├── concept_layer.py     # CollectiveConceptLayer
│   ├── mlp.py               # GroupedMLP
│   ├── vsa_utils.py         # VSA utilities
│   ├── lambda_utils.py      # Lambda_d hierarchy
│   ├── amp_codec.py         # SignedAmpCodec (порт из основного)
│   └── zeckendorf_readout.py # ZeckendorfReadout
│
├── scripts/
│   ├── train.py             # Цикл обучения
│   ├── generate.py          # Генерация
│   ├── probe_concepts.py    # Анализ концептов
│   ├── test_fractal_codec.py
│   ├── test_spiral_bind.py
│   ├── test_trajectory_bind.py
│   └── archive/
│
├── notebooks/
│   └── colab_mini.ipynb     # Colab ноутбук
│
├── tests/
│   └── test_specs.py        # Тесты спецификаций
│
├── sandbox_mirror/          # Песочница для mirror экспериментов
│
├── docs/                    # Документация
│   ├── ARCHITECTURE_DETAILED.md
│   ├── MIRROR_COLD_START.md
│   ├── PRIVATE_MEM.md
│   └── archive/
│
├── archive/                 # Архивные файлы
└── README.md                # Этот файл
```

## Текущая модель

| Параметр | Значение |
|----------|----------|
| D | 896 |
| Layers | 12 |
| Experts (G) | 8 |
| bind_K | 32 |
| vocab | 50000 |
| seq_len | 512 |
| Parameters | ~4.5M |

## Обучение

```bash
python scripts/train.py --data-dir ./data
```

## Связь с WideBind

Новые методы сначала тестируются здесь, затем переносятся в `WideBind/core/`.

**Статус переноса:**
- ✅ TrajectorySpiralBind → WideBind
- ✅ SigmoidCodedHead → WideBind
- ✅ Adaptive maturity collective → WideBind
- ✅ Adaptive mirror temperature → WideBind
- ✅ Variable Precision Memory → WideBind
- ✅ Explicit Reasoning → WideBind
