# Private Memory & Plasticity — архитектура метакогнитивного слоя

## Сущность

Эксперты строят внутри модели свой собственный слой — хранилище внутренних представлений
относительно данных, которые модель усваивает при обучении. Внутреннее Я модели.

## Трёхслойная архитектура

### Layer 0: Knowledge (знание)
`W_proj`, `W_out`, `alpha_diag`, MLP — то, что модель выучила из данных.
Не трогается напрямую. Пластичность здесь опасна (свести модель с ума).

### Layer 1: Meta-Knowledge (мета-знание)
`_private_mem[g]` — (G, k) EMA уверенных K-space состояний эксперта.
Пластичность безопасна: это не правка фактов, а правка отношения к фактам.

Механизмы записи:
- `conf = sigmoid(-|pred_error|)` — уверенность эксперта
- `contra = sigmoid(||hp - help_k||/||hp|| - 1)` — противоречие с коллективом
- `social_pressure = 1 - 0.5*sigmoid(relu(contra_expert) + isolation)` — давление коллектива
- `conf_plastic = conf * (1 - contra) * social_pressure` — итоговая запись
- Soft-competition: `conf_bc = conf^T * G / sum(conf^T)` (T=2, не winner-take-all)
- Adaptive decay: `decay = 0.990..0.999` (быстрый старт, медленная стабилизация)

### Layer 2: Arbiter (арбитр между Layer 0 и Layer 1)
Gate с 5 сигналами:

| Сигнал | Layer | Роль |
|--------|-------|------|
| `|pred_error|` | K | «я не знаю» |
| `|delta|` | M | «я корректирую» |
| `grad_mod` | ext | «меня учит loss» |
| `dvar_mod` | int | «я стабилен/нестабилен» |
| `disagreement * w_contra + contra_expert` | **A** | «я противоречу коллективу» |

## Concept / Contradiction Detection (из EVA, адаптировано в K-space)

| EVA (символьная) | Наша (K-space) | Применение |
|-------------------|----------------|------------|
| Numeric: `|v1-v2|/|v1|` | `disagreement = ||hp - help_k|| / ||hp||` | gate, write |
| Semantic: `1-cos(e1,e2)` | `concept_sim = cos(pm[g1], pm[g2])` | hierarchy |
| Lexical: `1-Jaccard(w1,w2)` | `behavior_div = 1 - cos(hp[g1], hp[g2])` | contradiction |
| Hierarchy: is_a chain | `contra_graph = concept_sim * behavior_div` | contra_expert |
| Cycle: A->B->C->A | broken trust chain `t[i,j]*t[j,k]*(1-t[k,i])` | isolation |
| Ambiguity: multiple meanings | `contra_expert[g]` = mean_j(contra_graph) | social_pressure |

## Expert Knowledge Graph (G × G)

- `_concept_sim_ema` — кто разделяет концепты (cosine similarity private_mem)
- `_behavior_div_ema` — кто ведёт себя по-разному (1 - cos hp behavior)
- `_trust_matrix` — кто кому помогает (cross-expert attention decay)
- `_cached_contra_graph` — матрица противоречий G×G
- `_cached_contra_expert` — per-expert степень противоречия
- `_cached_dominance` — per-expert авторитет (столбцы trust_matrix)
- `_cached_isolation` — per-expert изоляция (недоверие остальных)

## Параметры (на слой, G=8, k=32)

- `_private_mem`: (G, k) — persistent, EMA уверенных состояний
- `w_help`: (G, 1) — scale help_k, init `log(3)`≈1.1 (sigmoid~0.75)
- `w_contra`: (G,) — contradiction bias, init +0.01 (disagreement открывает gate)
- `_concept_sim_ema`: (G, G) — not persistent (пересчитывается)
- `_behavior_div_ema`: (G, G) — not persistent
- `_trust_matrix`: (G, G) — not persistent

Флаг: `--private-mem` в train.py

## Аудит ловушек (исправлено)

1. **NaN в concept_graph при step 0** — F.normalize(zero_vector) → NaN.
   Fix: `pm / pm_norm.clamp(min=1e-10)` вместо `F.normalize()`.
2. **Cold start write** — w_help init 0 → sigmoid(0)=0.5, halving help_k.
   Fix: w_help init log(3) → sigmoid(1.1)=0.75.
3. **Monoculture (winner-take-all write)** — один эксперт захватывает write budget.
   Fix: soft-competition `conf^T / sum(conf^T)` с T=2.
4. **Упорный скептик vs последовательный оппозиционер** — negative contra_expert
   не должен триггерить social_pressure. Fix: `relu(contra_expert)` перед sigmoid.
5. **Adaptive decay** — быстрый старт при пустой памяти, медленный при стабильной.
