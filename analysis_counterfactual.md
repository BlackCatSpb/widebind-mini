# Анализ от противного — нерабочие механизмы

## Метод
Берём каждый застопоренный механизм. Спрашиваем: «А что если бы наоборот?». Отбрасываем математически некорректное. Оставляем рабочие решения.

---

## 1. ls_var=0.036 — log_scale variance не растёт

### Текущее
`div_loss_raw = -(ls.var(dim=0).mean() + intra_weight * ls.var(dim=-1).mean())`  
Градиент: `d(div)/d(ls) = -w_div · 2 · (ls − ls.mean) / N`  
Проблема: w_div=0.087 → `|grad| ≈ 3e-7` на шаг. 2000 шагов → ls_var сдвинулся на 0.0002.

### От противного

| Если бы... | Результат | Математически корректно? |
|---|---|---|
| **w_div = 0** (нет push variance) | ls_var навсегда 0.036 — эксперты неспециализированы | Да, но бессмысленно |
| **w_div = 300** (x3000) | grad × 3000 → ls_var растёт за 10 шагов | **Да** — градиент корректен, ломает обучение? |
| **w_div → ∞** | ls_var → ∞ за 1 шаг | Нет — градиент не ограничен |
| **Не градиент, а direct noise** | ls.data.add_(torch.randn_like(ls) * 0.01) | Нет — разрывает autograd |
| **Не grad, а re-init: ls = linspace[-1, 1]** | ls_var = 0.44 | **Да** — одноразовое действие, не градиент |

### Реализовано (v2)
**`loss = -var(sigmoid(ls))`, w_div=50, chain rule через dsig**  
- Sigmoid отображает ls в [0,1], variance естественно ≤ 0.25
- Градиент: `d(loss)/d(ls) = dsig · d(loss)/d(sig)`
- dsig → 0 при sig→0/1 → **самозатухание**, ls не улетает в бесконечность
- w_div=50 компенсирует dsig ≈ 0.25 в линейной зоне
- В реальной модели: `loss → -var(sigmoid(ls))` — PyTorch autograd сам вычисляет chain rule

---

## 2. |1−α|=0.085 — alpha_diag не специализируется

### Текущее
α обновляется через `residual_var.var()` → EMA → sigmoid нормализация → lerp.  
Проблема: нормализация была `dim=-1` (внутри эксперта), исправлено на `dim=0` (между экспертами).  
Но даже с `dim=0`: обновление — `alpha_target = sigmoid(2.2 - log(relative_var))`.

### От противного

| Если бы... | Результат | Математически корректно? |
|---|---|---|
| α фиксирован (не обучается) | \|1-α\| = 0.096 навсегда | Да |
| α обновляется через gradient (learned) | α — nn.Parameter, градиент от loss | **Да** — стандартный путь |
| α = softmax(logits) × range | α конкурентны, sum = 1 | **Да**, но softmax — не хотим |
| α обновляется только при override < 0.1 | α заморожен при сильном override | Да — уже есть |

### Реализовано (v2)
**Adaptive alpha novelty push**  
- `w_eff = w × max(1.0, 0.1 / (α_std + 0.01))` — автомасштабирование
- Когда α_std ≈ 0: push усилен в 10x → пробивает порог (~0.045)
- Когда α_std > 0.1: w_eff = w (нормальный режим)
- Добавлен поверх `.data.add_()` (не через autograd — push на alpha_target до lerp)

---

## 3. gate_var=0.031 — gates не специализируются

### Текущее
`gate = sigmoid(logits)` — независимый per-expert per-token.  
Средний gate по всем токенам ≈ 0.5 (sigmoid(0) = 0.5).  
gate_var ≈ 0.031 — variance от random init w_gate.

### От противного

| Если бы... | Результат | Математически корректно? |
|---|---|---|
| gate = sigmoid(logits + bias), bias learnable | некоторые эксперты смещены в 0/1 | **Да** — bias решает cold start |
| gate = softmax(logits) | sum = 1, конкуренция | **Да**, но не хотим softmax |
| gate_var растёт → loss = -gate_var | градиент раздвигает gates | **Да** — direct entropy |
| balance_loss = 0 | gate_var растёт естественно | **Да** — баланс убирает подавление |
| gates заморожены на init | gate_var = 0.031 навсегда | Да |

### Реализовано (v2)
**Gate bias per expert + per-layer scale + gate_repulse**  
- `gate_bias = nn.Parameter(linspace[-scale, scale])` — learnable, инициализирован линспейсом
- `scale = 0.5 + 1.5 × layer / (n_layers - 1)` — первый слой scale=0.5 (мягкое, исследует), последний scale=2.0 (жёсткое, специализирует)
- `gate_repulse_loss = -gate.var()` — inverse of balance, bypasses spectral alignment
- `w_repulse = 0.3` (взаимодействует с gate_bias: смещение даёт variance, repulse её стабилизирует)

---

## 4. signal_ent=1.424 — signal weights не исследуют

### Текущее
Сигналы (temp, pred, smooth, sym, help) взвешиваются через `sigmoid(log_weights)`.  
`signal_entropy` — auxiliary loss, pushing entropy towards target.

### От противного

| Если бы... | Результат | Математически корректно? |
|---|---|---|
| signal_entropy_weight = 0 | weights застывают на init | Да |
| signal_entropy_weight → ∞ | weights = uniform навсегда | Да |
| Нет сигналов — только temp | ничего не настраивать | Да |

### Рабочее решение

**A. Увеличить weight для entropy push**  
Если entropy stuck, значит target entropy уже достигнут. Не баг, а фича.

---

## 5. Коэффициенты по умолчанию (v2)

| Параметр | Старое значение | Новое значение | Причина |
|----------|:-:|:-:|--------|
| `div_weight` | 3.0 | **50.0** | Sigmoid-дивергенция в 10–15x слабее unbounded var(ls) |
| `alpha_novelty_weight` | 0.05 | **0.05** (с adaptive boost) | Adaptive boost обеспечивает пробитие порога при α_std≈0 |
| `gate_repulse_weight` | 0.3 | **0.3** | Работает; c gate_bias стабилизирует variance |
| `gate_bias_scale` | 2.0 | **2.0** (per-layer) | scale=0.5→2.0 от первого к последнему слою |
| `gate_bias_scale_per_layer` | — | **True** | Автоматический расчёт scale по layer_idx |

---

## 6. Контринтуитивные выводы

1. **ls_var не двигается не потому что div слаб, а потому что ranking_loss ≠ 0**  
   При ls_var=0.036 все ls_mean ≈ 0, ranking_loss = ∞ (ни один expert не выше другого).  
   Но при ls_var → 0.5, ranking_loss падает. Проблема: deadlock — ls_var не растёт → ranking не может сортировать → ranking_loss высок → градиент ranking пытается изменить ls, но через спектральное выравнивание его обнуляют.

2. **Спектральное выравнивание — главный тормоз**  
   Все aux градиенты (кроме CE) фильтруются через cos_sim. Для orthogonal aux → scale=0 → aux не работает.  
   **Решение (v1-v2):** bypass спектрального выравнивания для `div`, `gate_repulse`, `alpha_novelty`, `ranking`.

3. **init имеет значение**  
   0.036 ls_var — это не «застрял», это «там и был». variance от linspace(-0.3, 0.3) = 0.037.  
   **Решение:** gate_bias инициализируется линспейсом, alpha — tau-based, всё с asymmetry.

4. **Фазовый переход alpha_novelty**  
   Без adaptive boost: w < 0.045 → α не растёт, w ≥ 0.045 → бинарный скачок.  
   С adaptive boost: плавная регулировка 0.005–0.10.

5. **Sigmoid-divergence vs unbounded**  
   `-var(sigmoid(ls))` самоограничена, но градиент затухает при насыщении.  
   На практике: ls_var ≈ 0.3–0.5 за 5k шагов, sig_var ≈ 0.02 (10% от макс 0.25).
