# WideBind Mini

**11.20M параметров (default), 12 слоёв, D=896.** Локальный тренировочный полигон для архитектуры WideBind.
Единственная известная архитектура с трёхслойным дифференцируемым аппаратом саморефлексии:
знание → мета-знание (private memory) → арбитр (contradiction gate).

Внутренняя память экспертов, cross-expert attention, граф концепций и детекция противоречий
в K-space — все механизмы встроены в градиентный спуск, без символических надстроек.

---

## 1. Философия

Большие языковые модели сегодня — это однослойные системы:
- Transformer: attention между токенами. Нет внутренней модели собственного знания.
- MoE: gate выбирает экспертов, но эксперты не общаются. Нет коллективной памяти.

WideBind строится на трёх слоях:

| Слой | Название | Сущность | Пластичность |
|------|----------|----------|-------------|
| L0 | Knowledge | Веса, `W_proj`, `alpha_diag`, MLP | **Опасна** — правка весов может разрушить модель |
| L1 | Meta-Knowledge | `_private_mem[g]` — EMA уверенных K-space состояний экспертов | **Безопасна** — это не правка фактов, а правка отношения к фактам |
| L2 | Arbiter | Gate с 5 сигналами, contradiction detection | Самонастраивается |

Принцип: **модель знает (L0), знает что она знает (L1), и решает кому верить (L2)**.
Всё это — одна дифференцируемая функция forward pass.

---

## 2. Архитектура модели

```
token IDs → PartitionedEmbedding (8×112, sparse 6/32) → [WideBindBlock × 12] → Final RMS Norm → PartitionedHead → logits
```

Каждый блок:

```
h → RMSNorm → Conv1d(groups=D, k=48) → BottleneckBind(D→K=32→D) → VSA Memory (chunked scan) → GroupedCognitiveMirror (8 экспертов) → DCT Spectral → GroupedMLP (8 групп, ×4)
```

### 2.1 BottleneckBind — межканальное скрещивание

Скрещивание размерностей через K=32 с Фибоначчи-твистом. Три режима (`--bind-twist-mode`):

- **off**: классическая билинейная регрессия `(hp·w_u) ⊙ (hp·w_v) @ W_out`
- **shift** (по умолчанию): сумма S билинейных произведений с golden-ratio сдвигом по K-пространству
- **cascade**: Фибоначчи-вложенные моночлены

`tie_bind=True`: W_out = W_proj^T (автоэнкодерная структура). w_u/v с std=1.0 (критичен для градиента).

### 2.2 VSA Memory — мультимасштабная векторная суперпозиция

Векторная суперпозиция с chunked prefix scan (CHUNK=32), 4 фиксированных масштаба
τ = [8, 32, 128, 512] с softmax-комбинацией (`scale_w` ∈ ℝ^{4×D}).

Каждый масштаб τ_s имеет свой prefix scan, результат — взвешенная сумма через
softmax по `scale_w`. Это даёт model 4 временных горизонта: короткий (τ=8),
средний (τ=32), длинный (τ=128), очень длинный (τ=512).

surprisal-gated i_gate, dual readout + first moment. fp32 guard для численной стабильности.

Пять гиперпараметров адаптируются через AdaptiveController из сигналов mirror:
write gate, read gate, decay, memory-to-value scale, noise.

### 2.3 Staircase K-space (иерархия mirror k)

Размер K-space mirror растёт с глубиной слоя, а не фиксирован:

| Слои | k | d/k | Характер |
|------|---|-----|----------|
| L0–L3 (ранние) | 8 | 14 | Широкое подпространство — грубые паттерны |
| L4–L7 (средние) | 16 | 7 | Среднее разрешение — смешанные признаки |
| L8–L11 (глубокие) | 32 | 3.5 | Узкое подпространство — тонкая специализация |

Идея: ранние слои обрабатывают общие паттерны (мало каналов, широкая проекция),
глубокие layer строят точные коррекции (много каналов, узкая проекция).

### 2.4 λ_d LR Hierarchy (дифференцированный learning rate)

Разные компоненты учатся с разными скоростями через λ_d = 1.839 (трибоначчи):

| Уровень | p | Множитель LR | Компоненты |
|---------|---|-------------|-----------|
| Embedding / VSA | −2 | 0.30× | embedding, readout, VSA memory |
| MLP Core | −1 | 0.54× | MLP, bind W_proj |
| Base | 0 | 1.00× | conv, norm, W_out, head |
| Mirror / Gates | +1 | 1.84× | mirror projections, α, gates, w_i, b_i |

### 2.5 GroupedMirror — ансамбль экспертов

8 экспертов, каждый в своём d=112 подпространстве D=896. Каждый эксперт:

1. Проецирует `h` в K-space (k = 8/16/32 staircase): `hp = h_reshape @ W_proj[g]`
2. Вычисляет 4 базовых сигнала коррекции + help_k (private memory)
3. EMA-нормирует сигналы (соизмеримость перед softmax)
4. Смешивает через learnable softmax-веса
5. Проецирует delta из K-space в D-space через W_out
6. Вычисляет per-expert gate: sigmoid(|pred_err| + |delta| + grad_mod + dvar_mod + contradiction)
7. Модулирует MLP и VSA memory через usefulness

**Сигналы:**

| Сигнал | Формула | Роль |
|--------|---------|------|
| temp_k | `hp - mc_k` | Отклонение от центроида памяти (долговременная стабильность) |
| pred_error | `hp - alpha * hp_prev` | Ошибка предсказания K-space траектории |
| smooth_k | `hp - conv(hp)` | Локальная когерентность (плавность) |
| sym_k | `(hp·w_u) * (hp_prev·w_v)` | Билинейное временное взаимодействие |
| **help_k** | `attn @ private_mem * sigmoid(w_help) * trust` | **Коллективная память уверенных экспертов** |

### 2.6 Private Memory Bank (Meta-Knowledge Layer)

**`--private-mem`** — ключевое нововведение. Каждый эксперт накапливает EMA своих уверенных
K-space состояний в `_private_mem[g]` (G×k). Другие эксперты читают из этой коллективной
памяти через cross-expert attention.

#### Механизм записи

Запись происходит только когда эксперт одновременно:
1. **Уверен**: `conf = sigmoid(-|pred_error|)` — gate закрыт = предсказание хорошее
2. **Не противоречит коллективу**: `contra = sigmoid(||hp - help_k||/||hp|| - 1)` — низкое рассогласование
3. **Не под социальным давлением**: `social_pressure = 1 - 0.5*sigmoid(relu(contra_expert) + isolation)`

Итоговая запись: `conf_plastic = conf * (1 - contra) * social_pressure`.
Soft-competition (`T=0.5`) предотвращает winner-take-all: `conf_soft^0.5 / sum(conf_soft^0.5)` (T<1 сглаживает, T>1 заостряет).

Adaptive decay: `decay = 0.990..0.999` — быстрый старт когда память пуста, медленный когда стабильна.

#### Механизм чтения

Неуверенный эксперт (`uncert = sigmoid(|pred_error|)`) запрашивает коллективный опыт:

```python
q = hp * uncert                     # запрос от неуверенных экспертов
keys = private_mem.detach().clone()  # замороженный снимок памяти
attn = softmax(q @ keys.T / √k)      # (G, G) — кто у кого спрашивает
help_k_base = attn @ keys            # взвешенная сумма уверенных состояний
trust = 1 - contra                   # доверие к коллективу
help_k = help_k_base * sigmoid(w_help) * trust
```

help_k — 5-й сигнал, со своим learnable весом в softmax-нормировке сигналов.

#### Детекция противоречий (EVA-inspired, в K-space)

Перевод символьных методов EVA-Ai в векторную форму:

| EVA (символьная) | WideBind (K-space) |
|-------------------|-------------------|
| Numeric: `|v1-v2|/|v1|` | `disagreement = ||hp - help_k|| / ||hp||` |
| Semantic: `1-cos(e1,e2)` | `concept_sim[g1,g2] = cos(pm[g1], pm[g2])` |
| Lexical: `1-Jaccard(w1,w2)` | `behavior_div = 1 - cos(hp_avg[g1], hp_avg[g2])` |
| Hierarchy: is_a chain | `contra_graph = concept_sim * behavior_div` |
| Cycle: A→B→C→A | broken trust chain: `t[i,j]*t[j,k]*(1-t[k,i])` |
| Ambiguity: multiple meanings | `contra_expert[g]` = mean over j of contra_graph |

#### Expert Knowledge Graph (G×G)

На каждом шаге обучения обновляется граф концепций:

- **`_concept_sim_ema`**: cosine similarity private_mem экспертов — кто хранит похожие концепты
- **`_behavior_div_ema`**: расхождение hp-траекторий — кто по-разному обрабатывает данные
- **`_trust_matrix`**: cross-expert attention decay — кто кому помогает
- **`_cached_contra_graph`**: `concept_sim * behavior_div` — матрица противоречий
- **`_cached_contra_expert`**: per-expert степень противоречия с коллективом
- **`_cached_dominance`**: per-expert авторитет (столбцы trust_matrix)
- **`_cached_isolation`**: per-expert изоляция

### 2.7 Arbiter: K-Space Gate (Layer 2)

Gate решает: открыть эксперта (доверить коррекцию) или закрыть (пропустить).
Пять сигналов:

```
gate_logits = |pred_error| @ w_gate          # Layer 0: "я не знаю этот паттерн"
            + |delta| @ w_delta_gate           # Mirror: "я применяю коррекцию"
            + grad_mod                          # Backprop: "меня учит loss"
            + dvar_mod                          # Internal: "я стабилен/нестабилен"
            + disagreement * w_contra          # Arbiter: "я противоречу коллективу"
            + contra_expert                     # Arbiter: "эксперт систематически противоречив"
```

`w_contra[g]` — learnable per-expert bias. Init +0.01: disagreement открывает gate
(«когда я противоречу коллективу — доверяй внешнему сигналу»).

### 2.8 GroupedMLP — Feed-Forward

8 групп × (112 → 448 → 112, SiLU). 79.6% всех параметров.
Параметры модулируются per-expert usefulness от mirror.

---

## 3. Параметры (D=896, L=12, G=8, bind_twist_mode=shift)

| Компонент | Параметров | % |
|-----------|-----------|----|
| Embed + LM Head | 51,920 | 0.42 |
| BottleneckBind (K=32, S=4 multi-ocular) | 1,723,776 | 13.96 |
| GroupedCognitiveMirror (staircase k=8/16/32) | 239,612 | 1.94 |
| Conv1d (k=48, groups=D) | 521,472 | 4.22 |
| DCT Spectral (λ_k × D) | 10,752 | 0.09 |
| VSA Gates + Multi-scale (w_i, w_q, scale_w, etc.) | 172,044 | 1.39 |
| GroupedMLP (expand=4) | 9,644,640 | 78.11 |
| **Total** | **~12.35M** | **100** |

При `--bind-twist-mode off`: BottleneckBind → 689,280 (S=1, tied), total ~11.20M.

---

## 4. Тренировка

### 4.1 Запуск

```bash
# Базовая тренировка (без private memory)
python train.py --data-dir ./wb --accum 8

# С private memory, contradiction gate и concept graph
python train.py --data-dir ./wb --accum 8 --private-mem

# С diversity loss для var(log_scale) — переопределить default (0.0001)
python train.py --data-dir ./wb --accum 8 --private-mem --div-weight 0.005
```

### 4.2 Ключевые аргументы

| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--D` | 896 | Размерность модели |
| `--n-layers` | 12 | Число слоёв |
| `--mlp-groups` | 8 | Число групп (экспертов) |
| `--accum` | 1 | Градиентная аккумуляция (effective batch × N) |
| `--batch-size` | 2 | Микро-батч |
| `--bind-twist-mode` | shift | Режим BottleneckBind (off/shift/cascade) |
| `--private-mem` | — | Включить мета-познание (private memory, contradiction, concept graph) |
| `--div-weight` | 0.0001 | Вес diversity loss (var(log_scale) bonus) |
| `--ranking-weight` | 0.1 | Pairwise порядок ls_mean по gate_usage |
| `--signal-entropy-weight` | 0.001 | Энтропийная регуляризация весов сигналов |
| `--log-scale-l2-weight` | 0.01 | L2 на exp(log_scale) > 10 |
| `--mirror-k-staircase` | True | Иерархия K-space по глубине (8/16/32) |
| `--lambda-lr-hierarchy` | True | λ_d LR иерархия (0.29×...3.38×) |
| `--compile` | — | torch.compile (только CUDA, на MX550 не работает) |
| `--max-steps` | 300K | Всего шагов |
| `--eval-interval` | 500 | Каждый N-й шаг — eval |
| `--save-interval` | 2000 | Сохранение чекпоинта |
| `--resume` | — | Путь к чекпоинту для продолжения |

### 4.3 Данные

Токен-потоки (uint16 memmap) в `--data-dir`:
- ADVENTUR (~2.1B токенов)
- DRAMA (~2.1B токенов)
- FANTASY (~2.1B токенов)

Всего: **~6.3B токенов**. Циклический перебор потоков, мягкий сброс состояния на EOS (state *= 0.3).

### 4.4 Оптимизация

- **AdamW** (β₁=0.9, β₂=0.95), LR=3e-4, weight_decay=0.01
- **Gradient accumulation**: effective batch = batch_size × seq_len × accum_steps
- **Bidirectional MirrorLR**: LR растёт с var(log_scale), |1-α|, gate_var. Без forced cosine decay.
  Если специализация падает — LR растёт, если стабилизируется — LR падает. Асимметричное ускорение/торможение.
- **Gradient clipping**: 0.5
- **Pure FP32**: никакого AMP, autocast или GradScaler
- **Градиентная аккумуляция**: loss усредняется по микро-шагам, grad нормируется N_steps

### 4.5 Loss

| Компонент | Вес | Описание |
|-----------|-----|-----------|
| CE (PAD/EOS masked) | 1.0 | Стандартный cross-entropy |
| pred_loss (K-space) | adaptive 0.05–1.0 | Ауксильярный loss на pred_error в K-space |
| gate_l1 | 0.001 | Штраф за открытые gate (разреженность) |
| reinforce | 0.001 | Подкрепление: gate должен совпадать с usefulness |
| balance | 0.001 | Баланс использования экспертов (load balancing) |
| diversity (var) | 0.0001 | Дисперсия log_scale (специализация) — `.var()` а не `.sum()` |
| ranking | 0.01 | Pairwise: ls_mean_i > ls_mean_j если gate_usage_i > gate_usage_j (margin=0.01) |
| signal_entropy | 0.001 | −H(ω) — равномерное использование 5 сигналов коррекции |
| log_scale_l2 | 0.01 | L2 на exp(log_scale) > 10 (защита от gradient explosion) |
| nuclear / orth | 1e-5 / 1e-4 | Регуляризация W_proj |

### 4.6 Чекпоинты

- `best.pt` — лучший по val_loss
- `step_{N}.pt` — каждый save_interval + по Ctrl+C (авто-resume)
- `eval_{N}.pt` — каждый eval
- Resume: автоматически подхватывает последний `step_*.pt`
- Сохраняется: model weights + optimizer + step counter + private memory (persistent buffers)

### 4.7 Статус тренировки

```
step=  1000 loss=5.2345 |1-a|=0.0987 g_var=0.0056 ls_var=0.0089 lr=2.45e-04 tok/s=56 mem=5.72GB
```

| Метрика | Что значит |
|---------|-----------|
| `loss` | Cross-entropy loss |
| `|1-a|` | Среднее |1 - alpha_diag| по слоям (специализация mirror) |
| `g_var` | Дисперсия гейтов по экспертам |
| `ls_var` | Дисперсия log_scale mirror |
| `lr` | Текущий learning rate |
| `tok/s` | Токенов в секунду |
| `mem` | Использование VRAM (GB) |

---

## 5. Аудит ловушек (исправлено в рамках разработки private memory)

### 5.1 NaN в concept_graph при step 0
`F.normalize(zero_vector)` → 0/0 = NaN → падение concept_graph → NaN во всех производных.
**Fix**: `pm / pm_norm.clamp(min=1e-10)` вместо `F.normalize()`.

### 5.2 Cold start write starvation
`w_help` init 0 → `sigmoid(0) = 0.5`, help_k вдвое слабее. trust = 1 - contra ≈ 0.5 (ещё вдвое).
Net: help_k ≈ 12% от help_k_base. Gradient до w_help идёт через 5 операций → vanishing.
**Fix**: w_help init `log(3) ≈ 1.1` → sigmoid ~0.75. w_contra init +0.01.

### 5.3 Monoculture (winner-take-all write)
`conf_bc = conf_plastic * G / sum(conf_plastic)` — жёсткая нормализация: один уверенный
эксперт захватывает весь write budget.
**Fix**: soft-competition `conf = conf^T / sum(conf^T)` с T=0.5 (T>1 заостряет, T<1 сглаживает).

### 5.6 Cold start private memory
`_private_mem` инициализировался нулями — `attn @ zeros = 0`, help_k не работал первые шаги.
**Fix**: `torch.randn(G, k) * 0.01` вместо `torch.zeros`.

### 5.7 Signal imbalance
help_k мог доминировать над остальными 4 сигналами (temp/pred/smooth/sym).
**Fix**: энтропийная регуляризация `-H(omega)` с весом 0.001 — поощрение равномерного использования всех сигналов.

### 5.8 Shift mode rank limitation
При `tie_bind=True` в режиме `shift` все S слагаемых проецировались через один W_out = W_proj^T,
ограничивая ранг суммы до K вместо S×K.
**Fix**: при shift + tie_bind автоматически `ocular="multi"` (отдельный W_out на каждый сдвиг).

### 5.9 Staircase k=4 bottleneck на L0-L3
k=4 при 5 сигналах коррекции — один сигнал проецируется в 4-мерное K-space.
**Fix**: staircase изменён с [4, 8, 16] на [8, 16, 32], минимальное k=8.
d/k на ранних слоях — 14 (было 28), достаточно для 5 сигналов.

### 5.10 Per-channel VSA τ overflow
Per-channel b_d могло уходить в экстремумы (τ → 163K), вызывая numerical overflow
в prefix scan. **Fix**: multi-scale VSA с 4 фиксированными τ = [8, 32, 128, 512],
комбинируемыми через per-channel softmax (`scale_w` ∈ ℝ^{4×D}).

### 5.11 LR death — замирание обучения
Единый LR для всех компонентов приводил к замиранию: gates не успевали за mirror,
VSA память дрейфовала. **Fix**: λ_d LR hierarchy — 6 уровней (0.087×…3.38×).

### 5.4 Упорный скептик vs последовательный оппозиционер
Отрицательный contra_expert (эксперт с противоположными, но последовательными концептами)
не должен триггерить social pressure.
**Fix**: `relu(contra_expert)` перед sigmoid.

### 5.5 Adaptive decay
При пустой `_private_mem` (первые шаги) накопление идёт с decay=0.990.
При стабильной `_private_mem` (norm > 3) decay → 0.999.
Переход плавный через `sigmoid(3.0 - pm_norm)`.

---

## 6. Структура проекта

```
├── train.py                     # Тренировочный скрипт (FP32), CLI + training loop
├── README.md                    # Этот файл
├── docs/
│   ├── ARCHITECTURE.md          # Полное описание архитектуры
│   ├── PRIVATE_MEM.md           # Meta-Knowledge Layer: private memory, contradiction, concept graph
│   └── (см. WideBind/docs/REVIEW_2026-07-22.md)  # Архитектурный обзор
├── core/
│   ├── config.py                # WideBandConfig (D=896, L=12, G=8, private_mem...)
│   ├── model.py                 # WideBindStack, VSA scan, BottleneckBind, GroupedCognitiveMirror, MLP
│   ├── lambda_utils.py          # lambda-d иерархия
│   └── zeckendorf_readout.py    # Zeckendorf LM Head (альтернатива PartitionedHead)
├── compression/                 # FCF-CPR сжатие чекпоинтов
├── checkpoints/                 # Чекпоинты (gitignored)
├── wb/                          # Токен-потоки ADVENTUR/DRAMA/FANTASY (gitignored)
└── logs/                        # Тренировочные логи (gitignored)
```

---

## 7. Отличия от полной WideBind (D=4096, L=32)

| Параметр | Mini | Полная |
|----------|------|--------|
| D | 896 | 4096 |
| Слоёв | 12 | 32 |
| Экспертов | 8 | 32 |
| d per expert | 112 | 128 |
| K (bind) | 32 | 64 |
| K (mirror) | 8/16/32 (staircase) | 32 |
| Параметров | 12.23M | 161M |
| Expand MLP | 4 | 4 |
| VSA | Multi-scale (4τ) | Per-channel τ |
| LR hierarchy | λ_d (0.087×…3.38×) | — (planned) |
| Ranking loss | ✓ | — (planned) |
| Signal entropy | ✓ | — (planned) |
| Log_scale L2 | ✓ | — (planned) |
| VRAM (train) | ~180 MB (L=512) | ~24 GB |
| tok/s (MX550) | ~56 | — |

---

## 8. Статус

Стабильная тренировка (pure FP32).
VRAM: ~180 MB (B=2, L=512, accum=8) — значительно ниже оценки 2.1 GB из-за оптимизации активаций.
tok/s: ~56 (MX550, L=512).
Активный режим: `shift` + `--private-mem` + staircase k + multi-scale VSA.

### Текущие изменения (July 2026)

| Изменение | Описание |
|-----------|----------|
| **bind_twist_mode default** | `"off"` → `"shift"` — Fibonacci-twist включён по умолчанию |
| **Staircase k fix** | [4, 8, 16] → [8, 16, 32] — устранён bottleneck на ранних слоях |
| **Embed sparsity fix** | `_mix_scale=0.1` → `2.0` — sigmoid активирует только активные коды (0.99 vs 0.50) |
| **MirrorLR geometry** | `min()` → геометрическое среднее `(a·b·c)^(1/3)` — мягче, не режет LR вдвое при всплеске var |
| **b_i clamp** | `max(b_i, -4.0)` — i_gate ≥ 0.018 на глубоких слоях, память не замирает |
| **λ_d LR hierarchy** | VSA: λ^(-4)→λ^(-2), Gate: λ^2→λ^1 — разница 6.2× вместо 39× |
| **VSA eps** | 1e-10 → 1e-6 — численная стабильность при экстремальных decay |
| **CLI `--div-weight`** | default 0.0 → 0.0001 — синхронизация с config |
| **Soft routing overlay** | `gate_bonus = (spec×0.5 + cons×0.5) × w_contra × 0.1` из `_behavior_div_ema` + `_concept_sim_ema`. Без новых параметров. |
| **Gate EMA fix** | `.data.mul_()` вместо `.mul_()` — in-place мутация ломала autograd |
| **Aux weights reduced** | `div_weight=0.0001`, `balance=0.001`, `reinforce=0.001`. `div_loss` переведён на `.var()`. |
| **Loss monitoring** | `_cached_losses` dict — каждый aux loss логируется отдельно |
| **debug_mind()** | gate_selectivity, behavior_div, concept_sim, spec_index, cons_index |

### Loss компоненты (актуальные веса)

| Компонент | Вес | Описание |
|-----------|-----|----------|
| CE (PAD/EOS masked) | 1.0 | Стандартный cross-entropy |
| pred_loss (K-space) | adaptive 0.05–1.0 | MSE предсказания K-space (устанавливается AdaptiveController) |
| gate_l1 | 0.001 | Разреженность gate |
| reinforce | 0.001 | Gate должен совпадать с usefulness |
| balance | 0.001 | Load balancing (энтропия использования → logG) |
| div (var(log_scale)) | 0.0001 | Специализация mirror (отрицательный loss, push variance) |
| ranking | 0.1 | Сортировка log_scale по gate_usage |
| signal_entropy | 0.001 | −H(ω) — равномерность сигналов |
| log_scale_l2 | 0.01 | L2 на exp(log_scale) > 10 |

### Следующие шаги

1. Полноценный запуск на T4 (Colab) с `--private-mem` и staircase k
2. Мониторинг contra_expert, trust_matrix, scale_w через debug_mind()
3. Мониторинг scale_w (softmax multi-scale VSA) — равномерность 4 масштабов
4. Перенос проверенных архитектурных улучшений в полную WideBind (D=4096)
