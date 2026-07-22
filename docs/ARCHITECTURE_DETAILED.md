# WideBind Mini: Архитектурный обзор

**12.23M параметров, 12 слоёв, D=896, 8 экспертов.**

Модель-полигон для трёхслойного дифференцируемого аппарата саморефлексии.
Знание (L0) → Мета-знание (L1) → Арбитр (L2).

---

## 1. Философия и мотивация

### 1.1 Проблема современных LLM

Все современные языковые модели (Transformer, MoE, RWKV, Mamba) имеют
принципиальное ограничение: **у них нет внутренней модели собственного знания**.

- Transformer: attention между токенами. Модель не знает, что она знает.
- MoE: gate выбирает экспертов, но эксперты не общаются. Нет коллективной памяти.
- EVA (символическая): есть concept graph и рефлексия, но на тексте, через LLM вызовы.
  Не дифференцируемо. Не масштабируется в обучении.

### 1.2 Решение: три слоя саморефлексии

| Слой | Название | Сущность | Пластичность |
|------|----------|----------|-------------|
| L0 | Knowledge | Веса, W_proj, alpha_diag, MLP | **Опасна** — правка весов может разрушить модель |
| L1 | Meta-Knowledge | _private_mem[g] — EMA уверенных K-space состояний | **Безопасна** — это не правка фактов, а правка отношения к фактам |
| L2 | Arbiter | Gate с 5 сигналами, contradiction detection | Самонастраивается через w_contra |

**Принцип:** модель знает (L0), знает что она знает (L1), и решает кому верить (L2).
Все три слоя — одна дифференцируемая функция forward pass. Градиент течёт через все
три слоя одновременно, но с разной скоростью и разным уровнем риска.

### 1.3 Что это даёт

1. **Безопасная пластичность.** L1 (private memory) можно менять без риска разрушить
   знания L0. Это аналог «я подумал и передумал» — не переписывание фактов,
   а переоценка их достоверности.

2. **Коллективный разум.** Эксперты обмениваются уверенными состояниями через
   cross-expert attention. Неуверенный эксперт спрашивает у коллектива.

3. **Детекция противоречий.** Если эксперт думает как все (concept_sim высок),
   но действует иначе (behavior_div высок) — это противоречие. Gate открывается,
   модель обращает внимание на расхождение.

4. **Социальная динамика.** Доминирующие эксперты получают больший вес в trust_matrix.
   Изолированным экспертам social_pressure блокирует запись в коллективную память.

---

## 2. Полная архитектура модели

### 2.1 High-level pipeline

```
token IDs → PartitionedEmbedding (8×112, sparse 6/32)
          → [WideBindBlock × 12]
          → Final RMS Norm
          → PartitionedHead → logits
```

### 2.2 Один WideBindBlock

Каждый блок:

```
h_in
  → RMSNorm
  → Conv1d (groups=D, k=48, causal padding)
  → BottleneckBind (D↔K=32, S=4, shift/tied→multi)
  → VSA Memory (chunked prefix scan, τ до 163K)
  → GroupedCognitiveMirror (8 экспертов, K-space)
  → DCT Spectral (частотная фильтрация)
  → GroupedMLP (8 групп × expand=4)
  → skip connection
→ h_out
```

### 2.3 PartitionedEmbedding и PartitionedHead

Входной и выходной слои работают с разделённым словарём.

Словарь V = 11000 токенов разбивается на G = 8 частей по V//G ≈ 1375 токенов.
Каждая часть эмбедится в своё d = 112 подпространство, затем объединяется в D = 896.

```python
# Embedding: (B, L) → (B, L, D)
h_g = embed.weight[V_g]  # (V//G, d)
h = concat([h_1, ..., h_8], dim=-1)  # (B, L, D)

# Head: (B, L, D) → (B, L, V)
h_g = h.reshape(B, L, G, d)  # split back
logits_g = h_g @ head.weight[V_g].T  # per-group
logits = scatter_add(logits_g)  # merge
```

Преимущества:
- Каждый эксперт имеет собственное подпространство эмбеддингов.
- Эмбеддинги естественно сегментированы по группам экспертов.
- Sparse вычисления: только 6 из 32 токенов активны при infer.

---

## 3. BottleneckBind — межканальное скрещивание

### 3.1 Архитектура

BottleneckBind — это слой, который проецирует D-мерное представление
через узкое горлышко K=32, применяет билинейное скрещивание и
проецирует обратно в D.

```
h (B, L, D) → W_proj (D→K) → hp (B, L, K)
  → twist(hp) → (B, L, K)
  → W_out (K→D) → (B, L, D)
```

### 3.2 Режимы скрещивания

#### off: классическая билинейная регрессия

```
bind = (hp · w_u) ⊙ (hp · w_v) @ W_out
```

Простое поэлементное произведение двух линейных проекций hp.
Ранг скрещивания ≤ K.

#### shift (режим по умолчанию): сумма S билинейных произведений с golden-ratio сдвигом

```
bind = Σ_s [(hp · w_u(s)) ⊙ roll(hp · w_v(s), g_s)] @ W_out(s)
```

где g_s = floor(s · K / φ) mod K — сдвиг по золотому сечению.

Каждый сдвиг создаёт новый паттерн межканального взаимодействия.
Сумма S слагаемых даёт нелинейный ранг до S·K (только при разных W_out).

**Multi ocular:** При tie_bind=True (W_out = W_proj^T) ранг каждого слагаемого
ограничен K, и сумма тоже. Поэтому при shift + tie_bind автоматически
включается `ocular="multi"`: отдельный W_out(s) для каждого сдвига.
Это добавляет параметры (S·K·D вместо K·D), но даёт полный ранг S·K.

#### cascade: Фибоначчи-вложенные моночлены

```
u₁ = hp · w_u(1)
u₂ = hp · w_v(2)
u_n = (u_{n-1} · w_u(n)) ⊙ roll(u_{n-2} · w_v(n), g_n)
bind = u_S @ W_out
```

Нелинейная глубина растёт по Фибоначчи: deg(u_n) = F_n.
При S=4 степени: [1, 1, 2, 3]. При S=6: [1, 1, 2, 3, 5, 8].

Для численной стабильности u_n перенормируется к ||u₁|| после каждого шага.

### 3.3 Инициализация

- w_u, w_v: std = 1.0 (критичен для градиента — std³ эффект).
- W_proj, W_out: std = 0.02.
- tie_bind = True: W_out = W_proj^T (автоэнкодерная структура).
  При shift + tie_bind → multi ocular (отдельные W_out(s), tie_bind отключается).

### 3.4 Параметры

BottleneckBind (K=32, S=4, D=896):
- W_proj: D · K = 28,672
- w_u, w_v: S · K · 2 = 256
- W_out (multi): S · K · D = 114,688
- Total per layer: 143,616
- Total for 12 layers: 1,723,392 (14.1% от модели)

---

## 4. VSA Memory — векторная суперпозиция

### 4.1 Концепция

VSA (Vector-Symbolic Architecture) Memory хранит суперпозицию векторов
с экспоненциальным затуханием. Это аналог рабочей памяти — модель может
«помнить» предыдущие состояния каналов с разными временными константами.

### 4.2 Механизм

```
mem[t] = α · mem[t-1] + β · x[t]   (write)
read_out = γ · mem[t] + δ · x[t]    (dual readout)
```

где α — decay (per-channel), β — write gate, γ, δ — learnable.

Chunked prefix scan (CHUNK=32) позволяет вычислять суперпозицию за
O(L · log(CHUNK)) вместо O(L²).

### 4.3 Per-channel временные константы

Каждый из D=896 каналов имеет собственную τ:

- b_d — learnable bias (init ~3.0)
- τ_d = exp(2.0 + 3.0 · sigmoid(b_d)) — от e² ≈ 7.4 до e⁵ ≈ 148
  при b_d=12.0: τ до ~163K токенов

AdaptiveController регулирует b_d из сигналов mirror:
- Если ошибка предсказания эксперта растёт — τ уменьшается (быстрее забываем)
- Если ошибка падает — τ растёт (дольше помним)

### 4.4 Параметры

VSA gates (per layer):
- w_mem2v, w_in, w_out, b_d, b_i: D + D + D + D + D = 5 × 896 = 4,480
- w_chunk, b_chunk: 2 × D = 1,792
- Total per layer: 17,920
- Total for 12 layers: 215,040 (1.76% от модели)

---

## 5. GroupedCognitiveMirror — ансамбль экспертов

### 5.1 Общая структура

8 экспертов, каждый работает в своём d = 112 подпространстве D = 896.
Каждый эксперт:

1. Проецирует h_g (d=112) → hp_g (k=32) через W_proj (d×k)
2. Вычисляет 5 сигналов коррекции в K-space
3. EMA-нормирует сигналы (соизмеримость)
4. Смешивает через learnable softmax-веса (5 весов, sum=1)
5. Собирает delta = Σ w_i · signal_i
6. Проецирует delta обратно: delta @ W_out (k→d)
7. Вычисляет per-expert gate = sigmoid(Σ сигналов)
8. delta модулируется gate: mirror = delta · gate

```
h_g (B, L, d)
  → W_proj (d→k) → hp (B, L, k)
  → [5 сигналов: temp, pred_error, smooth, sym, help_k]
  → EMA norm → softmax mix → delta (B, L, k)
  → W_out (k→d) → linear (B, L, d)
  → gate (sigmoid, 5 компонент)
  → mirror = tanh(linear) · gate
  → skip: mirror + alpha · linear
  → scale: mirror · exp(log_scale)
```

### 5.2 Пять сигналов K-space

| Сигнал | Формула | Семантика |
|--------|---------|-----------|
| temp_k (temporal) | hp - mc_k | Отклонение от центроида памяти. Что изменилось с момента последнего уверенного состояния |
| pred_error (prediction) | hp - α · hp_prev | Ошибка предсказания K-space траектории. Насколько удивителен текущий паттерн |
| smooth_k (smoothness) | hp - conv1d(hp) | Локальная когерентность: гладкий ли переход? |
| sym_k (symmetry) | (hp·w_u) ⊙ (hp_prev·w_v) | Билинейное временное взаимодействие: cross-term между текущим и прошлым состоянием |
| help_k (collective) | attn @ private_mem · sigmoid(w_help) · trust | Коллективная память уверенных экспертов (см. раздел 6) |

### 5.3 EMA-нормировка сигналов

Сигналы имеют разные масштабы. Для соизмеримости перед softmax:

```python
rms = s.norm(dim=(-2,-1), keepdim=True).mean(dim=(0,1), keepdim=True)
ema[i] = 0.999 * ema[i] + 0.001 * rms.squeeze()
s_norm = s / (ema[i] + 1e-8)
```

Текущий RMS каждого сигнала экспоненциально сглаживается (τ ≈ 1000 шагов).

### 5.4 Learnable signal weights

```python
w = softmax(signal_log_weights)  # 5 weights, sum=1
delta = Σ w[i] · signals_normed[i]
```

Энтропийная регуляризация: +0.001 · H(w) в loss (поощрение равномерного
использования всех 5 сигналов, предотвращает доминирование одного).

### 5.5 K-Space gate

Gate решает: открыть эксперта (доверить коррекцию) или закрыть.

Пять компонент gate_logits:

1. **|pred_error| @ w_gate** — «я не знаю этот паттерн» (Layer 0)
2. **|delta| @ w_delta_gate** — «я применяю коррекцию» (Mirror)
3. **grad_mod** — «меня учит loss» (backprop signal)
4. **dvar_mod** — «я стабилен / нестабилен» (internal state)
5. **disagreement · w_contra** — «я противоречу коллективу» (Arbiter, L2)
6. **contra_expert** — «этот эксперт систематически противоречив» (collective)

```python
gate = sigmoid(gate_logits)  # (B, L, G)
```

### 5.6 Self-organizing usefulness

Каждый эксперт предсказывает свою полезность для текущего токена:

```python
usefulness_logits = predictor(delta)  # (B, L, G)
threshold = median(usefulness_logits, dim=-1)  # per-token threshold
usefulness = sigmoid((usefulness_logits - threshold) / temperature)
```

Эксперты выше медианы → usefulness > 0.5 (активны для этого токена).
Эксперты ниже медианы → usefulness < 0.5 (подавлены).

Temperature самоорганизуется через homeostatic control:
```python
target_entropy = 0.75 · G · log(2)  # ~75% of max entropy
if actual_entropy > target: temp += (острее конкуренция)
if actual_entropy < target: temp -= (мягче)
```

### 5.7 Выход mirror

```python
linear = delta @ W_out  # (B, L, G, d)
mirror_raw = tanh(linear) + skip_alpha · linear  # skip connection
mirror = mirror_raw · exp(log_scale)  # per-dimension gain
mirror = mirror · gate.unsqueeze(-1)  # gated
mirror = mirror.reshape(B, L, D)  # merge experts
```

### 5.8 Параметры

GroupedCognitiveMirror (per layer):
- W_proj: G · d · k = 8 × 112 × 32 = 28,672
- mc_k: G · k = 256
- w_gate, b_gate, w_delta_gate: 3 × G · k = 768
- w_sym_u, w_sym_v: 2 × G · k = 512
- alpha_diag: G · k = 256
- log_scale: G · d = 896
- signal_log_weights: 5
- usefulness_predictor: k·k + k + k·1 + 1 ≈ 1157
- W_out: G · k · d = 28,672
- mod_scales, biases: ~32
- Total per layer: ~61K
- Total for 12 layers: ~732K (5.98% от модели)

---

## 6. Private Memory Bank — мета-познание (L1)

### 6.1 Концепция

Private Memory — это коллективная память уверенных K-space состояний экспертов.
Каждый эксперт накапливает EMA своих состояний, когда он уверен,
не противоречит коллективу и не под социальным давлением.

```python
_private_mem[g]  # (G, k) — persistent buffer
```

### 6.2 Механизм записи

Запись происходит только при выполнении трёх условий:

#### Уверенность (confidence)

```python
conf = sigmoid(-|pred_error|)  # (B, L, G, 1)
```

Когда предсказание экспертное точное (pred_error → 0), conf → sigmoid(0) = 0.5.
Когда ошибка велика, conf → sigmoid(-∞) = 0.0.
Не gate, а inverse gate: gate закрыт = предсказание хорошее.

#### Непротиворечие коллективу

```python
contra = sigmoid(||hp - help_k|| / ||hp|| - 1)
```

disagreement = ||hp - help_k|| / ||hp|| — относительное расхождение.
Если disagreement < 1 (эксперт согласен с коллективом), contra < 0.5.
Если disagreement > 1 (эксперт противоречит), contra > 0.5.

#### Социальное давление

```python
social_pressure = 1 - 0.5 · sigmoid(relu(contra_expert) + isolation)
```

contra_expert — среднее противоречие эксперта по всем парам (см. граф концепций).
isolation — насколько эксперт изолирован в trust_matrix.

relu(contra_expert): только положительные противоречия (непоследовательные
эксперты) влияют на давление. Последовательный оппозиционер с contra_expert < 0
не страдает от social_pressure.

#### Итоговая пластичность

```python
conf_plastic = conf · (1 - contra) · social_pressure
```

Произведение трёх факторов в [0, 1] → conf_plastic ∈ [0, 1].

#### Soft-competition

```python
conf_soft = conf_plastic ** 0.5  # T=0.5: сглаживает, <1 = истинная soft-competition
conf_bc = conf_soft · G / sum(conf_soft)  # нормировка на G
```

T=0.5 (а не 2.0, как было в первой версии): T > 1 заостряет распределение
(один эксперт захватывает весь write budget), T < 1 сглаживает.

#### Запись в память

```python
weighted_hp = mean(conf_bc · hp.detach(), dim=(0,1))  # (G, k)
pm_decay = 0.999 - 0.009 · sigmoid(3.0 - ||pm||)  # [0.990, 0.999]
_private_mem = pm_decay · _private_mem + (1 - pm_decay) · weighted_hp
_private_mem.clamp_(-10.0, 10.0)
```

Adaptive decay: быстрый старт (decay=0.990, быстрее заполняется)
когда память пуста, медленный (decay=0.999) когда стабильна (||pm|| > 3).

Инициализация: torch.randn(G, k) · 0.01 (не нули — иначе attn@zeros = 0
на старте, help_k не работает).

### 6.3 Механизм чтения (cross-expert attention)

Когда эксперт неуверен, он запрашивает коллективный опыт:

```python
uncert = sigmoid(|pred_error|)          # неуверенность: 1 - conf
q = hp · uncert                          # запрос от неуверенных экспертов
keys = private_mem.detach().clone()      # замороженный снимок памяти
attn = softmax(q @ keys.T / √k, dim=-1)  # (B, L, G, G) — кто у кого спрашивает
help_k_base = attn @ keys                # (B, L, G, k) — взвешенная сумма
```

Ключевые детали:
- keys заморожен (`.detach().clone()`) — предотвращает градиентный цикл:
  эксперт не может изменить память, из которой читает.
- attn имеет размер (G, G) — матрица «кто у кого спрашивает».
  Диагональ = эксперт спрашивает сам себя (свою же память).

#### Доверие к коллективу

```python
trust = 1 - contra  # от 0 (полное противоречие) до 1 (полное согласие)
help_k = help_k_base · sigmoid(w_help) · trust.unsqueeze(-1)
```

sigmoid(w_help): w_help init = log(3) ≈ 1.1 → sigmoid ~0.75.
trust = 1 - contra: высокое противоречие → низкое доверие к коллективу.

### 6.4 help_k как 5-й сигнал

help_k добавляется к 4 базовым сигналам (temp, pred_error, smooth, sym):

```python
signals = [temp_k, pred_error, smooth_k, sym_k, help_k]  # 5 сигналов
w = softmax(signal_log_weights)
delta = Σ w[i] · signals_normed[i]
```

Энтропийная регуляризация сигналов: -0.001 · H(w) в loss.
Поощряет равномерное использование всех 5 сигналов.

---

## 7. Expert Knowledge Graph (L1.5)

### 7.1 Концепция

Граф G×G, отражающий отношения между экспертами. Обновляется на каждом шаге
(в eval тоже, но без записи в память).

### 7.2 Компоненты графа

#### Concept Similarity (concept_sim)

```python
pm_n = _private_mem / ||_private_mem||.clamp(min=1e-10)  # normalized
concept_sim = pm_n @ pm_n.T  # (G, G) — cosine similarity
self._concept_sim_ema.mul_(0.999).add_(concept_sim, alpha=0.001)
```

Что хранят эксперты? Похожие концепты → высокий concept_sim.
Разные концепты → низкий.

Защита от NaN: clamp(min=1e-10) при нормировке нулевого вектора.

#### Behavior Divergence (behavior_div)

```python
hp_avg = mean(hp, dim=(0,1))  # (G, k) — усреднённое состояние эксперта за шаг
hp_n = F.normalize(hp_avg)
behavior_sim = hp_n @ hp_n.T
behavior_div = 1 - behavior_sim
self._behavior_div_ema.mul_(0.999).add_(behavior_div, alpha=0.001)
```

Как эксперты обрабатывают данные? Похожие траектории → низкий behavior_div.
Разные траектории → высокий.

#### Contradiction Graph (contra_graph)

```python
contra_graph = concept_sim · behavior_div
contra_expert[g] = mean_j(contra_graph[g, j])
```

Высокий contra_graph = эксперты думают похоже (concept_sim высок),
но действуют по-разному (behavior_div высок) = **противоречие**.

contra_expert[g] — среднее противоречие эксперта g со всеми остальными.

#### Trust Matrix

```python
trust_weights = mean(attn, dim=(0,1))  # (G, G) — средняя attention
self._trust_matrix.mul_(0.999).add_(trust_weights, alpha=0.001)
```

Кто у кого чаще спрашивает? trust_matrix[g1, g2] — насколько g1 доверяет g2.

#### Dominance и Isolation

```python
dominance[g] = sum_j trust_matrix[j, g]  # насколько g авторитетен
isolation[g] = 1 - sum_j trust_matrix[g, j] / G  # насколько g изолирован
```

### 7.3 EVA-inspired mapping

Символьные методы концептуального детектора EVA-Ai переведены в K-space:

| EVA (символическая) | WideBind (K-space) |
|-------------------|-------------------|
| Numeric: |v1-v2|/|v1| | disagreement = ||hp - help_k|| / ||hp|| |
| Semantic: 1-cos(e1,e2) | concept_sim = cos(pm[g1], pm[g2]) |
| Lexical: 1-Jaccard(w1,w2) | behavior_div = 1 - cos(hp_avg[g1], hp_avg[g2]) |
| Hierarchy: is_a chain | contra_graph = concept_sim · behavior_div |
| Cycle: A→B→C→A | broken trust chain: t[i,j] · t[j,k] · (1-t[k,i]) |
| Ambiguity: multiple meanings | contra_expert[g] = mean_j(contra_graph[g,j]) |

---

## 8. Arbiter: K-Space Gate (L2)

### 8.1 Роль

Gate — это механизм, который решает, доверить ли эксперту коррекцию
представления (открыть gate) или пропустить (закрыть).

Без gate: все эксперты всегда влияют на выход → усреднение → нет специализации.
Gate: эксперты конкурируют за право влиять на конкретные токены.

### 8.2 Пять компонент gate

```
gate_logits =
  |pred_error| @ w_gate          Layer 0: "я не знаю этот паттерн"
  + |delta| @ w_delta_gate        Mirror: "я применяю коррекцию"
  + grad_mod                      Backprop: "меня учит loss"
  + dvar_mod                      Internal: "я стабилен/нестабилен"
  + disagreement · w_contra       Arbiter: "я противоречу коллективу"
  + contra_expert                 Arbiter: "эксперт систематически противоречив"
```

### 8.3 grad_mod — gradient signal

```python
grad_mod = exp(log_grad_mod_scale) · tanh(prev_grad_norm + grad_mod_bias)
```

prev_grad_norm устанавливается hook'ом после backward.
Если loss «толкает» подпространство эксперта, gate открывается.

### 8.4 dvar_mod — variance signal

```python
dvar = mean(var(delta, dim=(0,1)), dim=-1)  # (G,) — variance of K-space correction
dvar_mod = exp(log_dvar_mod_scale) · tanh(dvar + dvar_mod_bias)
```

Если коррекция эксперта нестабильна (высокая variance), gate закрывается.
Если стабильна — открывается.

### 8.5 contradiction signal (L2, arbiter)

```python
gate_logits += disagreement · w_contra.unsqueeze(0).unsqueeze(0)
gate_logits += contra_expert.unsqueeze(0).unsqueeze(0)
```

w_contra init +0.01: disagreement открывает gate
(«когда я противоречу коллективу — доверяй внешнему сигналу»).

contra_expert: если эксперт систематически противоречив, его gate открыт
(«этот эксперт видит мир иначе — пусть говорит»).

---

## 9. GroupedMLP — Feed-Forward

### 9.1 Архитектура

8 групп × (112 → 448 → 112, SiLU). 79.6% всех параметров.

```python
h_g (B, L, G, d) → up_proj (d → d*expand=4) → SiLU → down_proj (d*expand → d)
```

Каждая группа — независимый MLP со своими весами.
Параметры модулируются per-expert usefulness от mirror:

```python
modulation = usefulness · sigmoid(mod_scale_mlp)  # (B, L, G)
out = mlp(h_g) · modulation.unsqueeze(-1)
```

### 9.2 Параметры

Per group: d · 4d + 4d · d = 2 · d² · expand = 2 · 112² · 4 = 100,352
Per layer: 8 × 100,352 = 802,816
Total for 12 layers: 9,633,792 (78.73% от модели с multi-ocular)

---

## 10. Параметры (полные, D=896, L=12, S=4)

| Компонент | Параметров | % |
|-----------|-----------|----|
| Embed + LM Head | 8,192 | 0.07 |
| BottleneckBind (S=4, multi) | 1,723,392 | 14.09 |
| GroupedCognitiveMirror | 732,096 | 5.98 |
| Conv1d (k=48) | 43,008 | 0.35 |
| DCT Spectral | 10,752 | 0.09 |
| VSA gates | 215,040 | 1.76 |
| GroupedMLP (expand=4) | 9,633,792 | 78.73 |
| **Total** | **~12,230,272** | **100** |

При `--private-mem`: дополнительно ~1K параметров (w_help + w_contra + buffers) — пренебрежимо мало.

---

## 11. Тренировка

### 11.1 Гиперпараметры

| Параметр | Значение |
|----------|---------|
| Оптимизатор | AdamW (β₁=0.9, β₂=0.95) |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Gradient clipping | 0.5 |
| Gradient accumulation | 8 steps (effective batch 8192) |
| LR scheduler | Bidirectional MirrorLR |
| Precision | Pure FP32 (no AMP, no autocast) |
| Vocab | 11000 токенов |

### 11.2 Bidirectional MirrorLR

LR растёт с var(log_scale), |1-α|, gate_var.
Без forced cosine decay.

Если специализация падает — LR растёт (модель быстрее ищет).
Если стабилизируется — LR падает (модель нашла решение).

Асимметричное ускорение/торможение: LR множитель = sigmoid(состояние).

### 11.3 Loss

| Компонент | Вес | Описание |
|-----------|-----|----------|
| CE (PAD/EOS masked) | 1.0 | Стандартный cross-entropy |
| pred_loss (K-space) | adaptive 0.05–1.0 | MSE предсказания K-space |
| gate_l1 | 0.001 | Разреженность gate |
| reinforce | 0.01 | Gate должен совпадать с usefulness |
| balance | 0.01 | Load balancing (энтропия использования) |
| diversity | 0.001 | var(log_scale) — специализация |
| signal_entropy | 0.001 | H(omega) — равномерность сигналов |
| nuclear | 1e-5 | Ядерная норма W_proj |
| orth | 1e-4 | Ортогональность W_proj |

### 11.4 Данные

Три потока по ~2.1B токенов (uint16 memmap):
- ADVENTUR
- DRAMA
- FANTASY

Циклический перебор потоков, мягкий сброс состояния на EOS (state ×= 0.3).

### 11.5 Метрики

```
step=  1000 loss=5.2345 |1-a|=0.0987 g_var=0.0056 ls_var=0.0089 lr=2.45e-04 tok/s=56 mem=5.72GB
```

| Метрика | Диапазон | Что значит |
|---------|----------|-----------|
| loss | 10.8 → ~3.5 | Cross-entropy (цель: ~3.5 после 300K steps) |
| |1-a| | 0.0–1.0 | Специализация mirror (1-α_diag). 0=все одинаковы, 1=полная специализация |
| g_var | 0.0–0.5 | Дисперсия gate. 0=все открыты, >0.05=специализация gate |
| ls_var | 0.0–∞ | Дисперсия log_scale. Растёт со специализацией экспертов |
| lr | 1e-5–5e-4 | MirrorLR: самоорганизуется |
| tok/s | ~42 | Производительность (MX550, 2.1 GB) |

---

## 12. Аудит ловушек (trap audit)

### 12.1 NaN в concept_graph при step 0
`F.normalize(zero_vector)` → 0/0 = NaN → падение concept_graph.
**Fix**: `pm / pm_norm.clamp(min=1e-10)` вместо `F.normalize()`.

### 12.2 Cold start write starvation
w_help init 0 → sigmoid(0) = 0.5, help_k вдвое слабее.
trust = 1 - contra ≈ 0.5 (ещё вдвое). Net: help_k ≈ 12% от help_k_base.
Gradient до w_help идёт через 5 операций → vanishing.
**Fix**: w_help init log(3) ≈ 1.1 → sigmoid ~0.75. w_contra init +0.01.

### 12.3 Monoculture (winner-take-all write)
conf_bc = conf_plastic · G / sum(conf_plastic) — жёсткая нормализация:
один уверенный эксперт захватывает весь write budget.
**Fix**: soft-competition conf^T / sum(conf^T) с T=0.5 (T>1 заостряет, T<1 сглаживает).

### 12.4 Упорный скептик vs последовательный оппозиционер
Отрицательный contra_expert (эксперт с противоположными, но последовательными
концептами) не должен триггерить social pressure.
**Fix**: relu(contra_expert) перед sigmoid.

### 12.5 Adaptive decay
При пустой _private_mem (первые шаги) накопление идёт с decay=0.990.
При стабильной _private_mem (norm > 3) decay → 0.999.
**Fix**: sigmoid(3.0 - pm_norm) — плавный переход.

### 12.6 Cold start private memory
_private_mem инициализировался нулями — attn @ zeros = 0,
help_k не работал первые шаги.
**Fix**: torch.randn(G, k) * 0.01 вместо torch.zeros.

### 12.7 Signal imbalance
help_k мог доминировать над остальными 4 сигналами (temp/pred/smooth/sym).
**Fix**: энтропийная регуляризация -H(omega) с весом 0.001.

### 12.8 Shift mode rank limitation
При tie_bind=True в режиме shift все S слагаемых проецировались через
один W_out = W_proj^T, ограничивая ранг суммы до K вместо S×K.
**Fix**: при shift + tie_bind → multi ocular (отдельный W_out на каждый сдвиг).

### 12.9 bind_twist_S mismatch
config.bind_twist_S по умолчанию был 1, из-за чего shift всегда работал с S=1
независимо от переданного режима.
**Fix**: default config 4, модель принудительно ставит S=1 для off mode.

---

## 13. Статус и ближайшие шаги

### Что работает
- Тренировка на MX550 (2.1 GB VRAM, ~42 tok/s)
- Private memory: запись (EMA + soft-competition + social pressure)
- Private memory: чтение (cross-expert attention + trust)
- Contradiction gate: disagreement · w_contra + contra_expert
- Expert Knowledge Graph: concept_sim, behavior_div, trust_matrix
- BottleneckBind: off / shift / cascade с multi ocular
- VSA Memory: chunked prefix scan, dual readout, per-channel τ
- GroupedCognitiveMirror: 5 сигналов + gate
- Gradient flow: grad_mod, dvar_mod, внимание через всю цепочку
- Soft-competition: T=0.5 (истинная)
- Signal entropy: H(omega) ≈ 1.609 (равномерное распределение на старте)

### Что тестируется
- Обучение с `--private-mem --div-weight 0.005` — первая тренировка с L1+L2
- Динамика concept_graph на реальных данных (первые 10K шагов)
- trust_matrix: кто у кого спрашивает

### Планы
1. Масштабирование: перенос на полную WideBind (D=4096, L=32)
2. Retrospective regret: если w_help.grad > 0 (вредит loss),
   принудительная реструктуризация private memory
3. Кэширование: checkpointing private memory для долгосрочной
   согласованности концептов через рестарты
4. Interpretability: direct readout concept_sim dendrogram
   (какие эксперты объединяются в кластеры)

---

*WideBind Mini — C. BlackCatSpb, July 2026*
*Вдохновлено EVA-Ai contradiction detection, VSA memory, Neuro-Symbolic AI*
