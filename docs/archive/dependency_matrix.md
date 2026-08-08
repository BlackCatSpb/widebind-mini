# Dependency Matrix — методы, параметры, подавления

## 1. Матрица влияния

```
                    ВЛИЯЕТ НА
МЕТОД →      ls_var  |1-α|  gate_var  ranking  aux_mirr  CE   signal
──────────────────────────────────────────────────────────────────────
div loss       +       ·      ·         ·        ·        ·    ·    
ranking loss   ≷      ·      ·         ✓        ·        ·    ·    
balance loss   ·       ·      −         ·        ·        ·    ·    
gate L1        ·       ·      −         ·        ·        ·    ·    
reinforce      ·       ·      +         ·        ·        ·    ·    
gate_repulse   ·       ·      +         ·        ·        ·    ·    
alpha_novelty  ·       +      ·         ·        ·        ·    ·    
gate_bias      ·       ·      +         ·        ·        ·    ·    
spectral alg   ·       ·      ·         −        −        ·    −    
bypass div     +       ·      ·         ·        ·        ·    ·    
bypass rank    ≷      ·      ·         ✓        ·        ·    ·    
bypass repulse ·       ·      +         ·        ·        ·    ·    
bypass nov.    ·       +      ·         ·        ·        ·    ·    
alpha update   ·       +      ·         ·        ·        ·    ·    
homeo boost    +       ·      +         ·        ·        ·    ·    
phase scaling  ·       ·      ·         ·        ·        ·    ·    
aux mirror     ·       ·      ·         ·        ✓        ·    ·    
pred loss      ·       ·      ·         ·        ·        +    ·    
LR scheduler   ·       ·      ·         ·        ·        +    ·    
```

**Легенда**: `+` push, `−` suppress, `≷` конфликт, `✓` работает, `·` нет прямого влияния

---

## 2. Цепочки подавления

### 2a. ls_var: тройное подавление

```
spectral_alignment → div_grad → ls_var ↑
                           ↓
                     cos_sim(CE, div) ≈ 0
                           ↓
                      scale = 0
                           ↓
                    div_grad заблокирован     ✗
```

**Исправлено** (entry 1 → entry 2): div bypasses spectral alignment.  
**Остаётся**: ranking_loss всё ещё через spectral alignment.

### 2b. gate_var: HHI-ловушка

```
balance_weight=0.026 → HHI(mean_gates) → loss → spectral_alignment
                                                           ↓
                                          cos_sim(CE, balance) ≈ 0
                                                           ↓
                                                     scale = 0
                                                           ↓
                                        balance не работает → gate uniform  ✗
```

Но даже если balance работал бы — он **штрафует gate_var**, а не помогает.  
**Парадокс**: balance_weight > 0 → gate_var ↓. balance_weight = 0 → gate_var может расти.

### 2c. ranking deadlock

```
gate_var ≈ 0.03 → gate_usage ≈ uniform → ranking не может отсортировать
                                                            ↓
                                              ranking_loss высокий
                                                            ↓
                                              spectral → scale ≈ 0
                                                            ↓
                                        ranking не работает → gate uniform  ✗
```

**Deadlock**: ranking требует gate_var, gate_var не растёт, ranking не работает.

### 2d. α update loop

```
pred_error → residual_var → dim=-1 norm (было) → все α одинаковы
                                                       ↓
                                             исправлено: dim=0 norm
                                                       ↓
                                            α специализируются? медленно
                                                       ↓
                                          lerp_rate=0.01 × residual_var
```

---

## 3. Конфликты (push vs suppress)

| Пара | Конфликт | Кто побеждает | Почему |
|------|----------|---------------|--------|
| div vs ranking | div: var↑, ranking: order | ranking (50× вес) | ranking_weight=0.001 vs 0.087? Нет, ranking_raw=48, div_raw=−0.042. 48×0.001=0.048 vs 0.087×0.042=0.0037. ranking в 13× сильнее |
| balance vs reinforce | balance: uniform, reinforce: match | balance | balance подавляет gate_var |
| spectral vs all aux | spectral: align, aux: orthogonal | spectral | scale=0 для orthogonal |
| lr_scheduler vs phase | lr↓ при var↓, phase: mirror scale | depends | |

---

## 4. Что упустили (gap analysis)

### Gap A: gate_var не растёт, потому что нет push-механизма
Все aux воздействия на gate — через cosine alignment (→0) или через подавление (balance).  
**Нет метода, который ПРЯМО увеличивает gate_var.** Ни div, ни ranking, ни reinforce.

Решение: `gate_var_loss = -gate_var * w_gate_var` — прямой градиент на logits.

### Gap B: ranking_loss и div_loss — одна цель, два подхода
Оба хотят экспертной специализации, но:  
- div толкает variance (количество = любые различия)  
- ranking требует порядок (качество = правильные различия)  

Они **не конфликтуют**, если ls_var > 0. Но при ls_var ≈ 0:  
- div: толкает все в разные стороны → работает при любом направлении  
- ranking: требует «правильного» направления → не работает без gate_var  

Пропущенный chain: div↑ → ls_var↑ → gate может использовать variance → gate_var↑ → ranking может работать.

### Gap C: alpha обновляется мимо autograd
`alpha_diag.data.lerp_(target, 0.01)` — in-place, не через граф.  
Градиент через alpha не течёт — alpha меняется только через эту heuristic.  
Альтернатива: `alpha_diag = nn.Parameter`, `alpha_loss` в aux_dict, градиент через autograd.

### Gap D: homeostatic boost → gate, не log_scale
Boost влияет на gate_logits → expert_gate → mirror_output → loss → log_scale grad.  
Двухшаговая косвенность: `boost → gate↑ → mirror_out↑ → grad(log_scale)↑`.  
Теряется ~50% силы сигнала (gate может быть 0.5 → mirror_out × 0.5 → grad × 0.5).

---

## 5. Сводка: 4 рабочих рычага

| Рычаг | Куда давит | Сейчас | Надо | Эффект |
|-------|-----------|--------|------|--------|
| **w_div** | ls_var ↑ | 0.087 | 1.0–3.0 | grad × 10–30 |
| **linspace init** | ls_var | (-0.3, 0.3) → 0.036 | (-1.0, 1.0) → 0.44 | ×10 init variance |
| **spectral align** | все aux ↓ | ON | OFF для div/ranking | div/ranking работают |
| **balance_weight** | gate_var ↓ | 0.026 | 0 | gate_var растёт |
| **lerp_rate α** | α адаптация | 0.01 | 0.1 | α быстрее в 10× |

**Первичный приоритет**: убрать spectral alignment для div, ranking, balance.  
**Вторичный**: w_div ×10, lerp_rate ×10, balance_weight → 0.
