# Mirror Cold-Start Problem

## Core Issue

`GroupedCognitiveMirror.log_scale` (shape `[G, d]`, per-expert per-dimension log-scale for
mirror output) does not differentiate during training. After 2100 steps with `div_weight=0.005`,
`ls_var` remained at its init value 0.0196 (no change at float32 precision). All G=8 experts
behave identically (behavior_div=0.0009, concept_sim=random, gates uniform ~0.5).

The MLP + embedding layers absorb the learning task; the mirror contributes nothing.

## Root Cause Analysis

### 1. Gradient to log_scale is structurally diluted

The gradient through any loss term to each element `ls[i,j]` of shape `(G, d)` is divided by
at least `d` (and often also by `G` and `L` depending on reduction).

**dilution sources**:

| operation | division factor | reason |
|-----------|----------------|--------|
| `ls.mean(dim=-1)` → per-expert scalar | `1/d` | averaging over d-dimensions |
| `F.mse_loss(..., reduction='mean')` | `1/(G*d)` by default | standard MSE reduction |
| `var(dim=0).mean()` | `1/d` | var over experts, then mean over d |
| `for layer ... loss /= n_layers` | `1/L` | averaging over layers |
| `all_ls.var()` (scalar) | `1/(L*G*d-1)` | single scalar variance |

Current combos and their actual per-element gradient for `div_loss`:

| Formula | Gradient per ls element | at init (x=0.1) | at ls_var=0.43 |
|---------|------------------------|-----------------|----------------|
| `-div_w * all_ls.var(dim=0).mean()` (old) | `-div_w * 2*(x-m)/[d*(L*G-1)]` | `-0.005*2*0.1/(28*95)` = **3.8e-7** | — |
| `-div_w * all_ls.var()` (current) | `-div_w * 2*(x-m)/(L*G*d-1)` | `-0.5*2*0.1/(2687)` = **3.7e-5** | `-0.5*2*1.0/2687` = **3.7e-4** |
| `-div_w * sum(sum((x-m)^2))` | `-div_w * 2*(x-m)` = **no division** | `-0.5*2*0.1` = **0.1** | `-0.5*2*1.0` = **1.0** |

The problem: MSE, var, mean all divide by N. The naive solution (remove all denominators)
creates numerical issues (loss magnitude scales with N, different N for mini vs main).

### 2. EMA signal norm suppresses mirror at init

`_signal_norm_ema` initialized to 3.0. All 5 signals (temp, pred, smooth, sym, help)
are divided by this EMA norm. At step 0:

- signal RMS ≈ 1.0 (random init)
- EMA norm = 3.0 (init)
- signal_normalized = signal / 3.0 ≈ 0.33× suppressed
- signal_weights = uniform ~0.2 (softmax of zeros)
- merged delta = 0.2 × 0.33 = 0.067

Mirror output ≈ delta @ W_out × exp(log_scale) × gate ≈ 0.067 × small × 1.0 × 0.5
= negligible. CE gradient through this to log_scale is near zero.

EMA decays from 3.0 to 1.0 over ~1000 steps (decay=0.001). By step 1000, signals are
correctly normalized. But the first 1000 steps train the network to IGNORE the mirror.
By the time normalization recovers, the model has learned "mirror = noise → gate ≈ 0".

**Fix applied**: EMA init → 1.0 (no suppression at step 0).

### 3. Div_loss gradient is self-defeating at low var

`-div_w * var(log_scale)` gradient: `∇ = -div_w * 2*(x - mean) / (N-1)`.

When all log_scale values are nearly equal (cold start), `(x - mean) ≈ 0`,
the gradient is ≈ 0 regardless of div_w. The loss only "activates" after
differentiation already exists — a chicken-and-egg problem.

**Linspace fix**: `ls_base = linspace(-1.0, 1.0, G).expand(G, d)` provides
immediate structured variance at step 0: ls_var ≈ 0.43 (vs 0.02 from pure noise init).

But linspace(-1.0, 1.0) → exp ∈ [0.37, 2.72]. With G=8, d=28:
expert 0 has exp(1.0) × 1.0 ≈ 2.72. Mirror output scales by 2.72 for half the experts.

12 layers × 2.72 × ~0.3 (mirror magnitude) = ~10 added to residual per step.
This overwhelms the embedding/MLP output, causing loss to climb (11 → 66+).

### 4. Gate-utility signal (benefit) is recursive

`benefit_loss = MSE(ls_mean, benefit * span * 2)` where:
- benefit = zero-meaned gate_usage
- gate_usage = per-expert average sigmoid(gate_logits)
- gate_logits depends on pred_error, delta, grad_mod, dvar_mod
- All of these depend on W_proj, W_out quality

At init: all gates ≈ 0.5 → benefit ≈ 0 → target ≈ 0 → MSE pulls ls_mean to 0,
COUNTERACTING the linspace init. This is why loss jumped to 216 initially.

**Fix applied**: `effective_weight = benefit_weight * |benefit_ema| * 2`.
When benefit_ema ≈ 0, effective_weight ≈ 0. Benefit loss phases in automatically
as gate differentiates.

### 5. Per-dim specialization vs per-expert scalar control

log_scale has shape (G, d) — each expert has d=28 independent scales.
But the benefit signal is per-expert scalar (gate_usage averaged over batch).
The gradient `∂benefit_loss / ∂ls[i,j]` passes through `ls.mean(dim=-1)` which adds `1/d`.

This means d dimensions within an expert are pushed COHERENTLY (same gradient
direction), making per-dim specialization rely solely on the very weak CE + div_loss
gradients. d dimensions cannot differentiate from benefit_loss alone.

## Summary of Applied Fixes

| Fix | File:Line | Status | Effect |
|-----|-----------|--------|--------|
| EMA norm init 3.0→1.0 | `model.py:326` | ✅ | No cold suppression |
| log_scale linspace(-1,1) | `model.py:329` | ✅ | Immediate ls_var=0.43 |
| div_weight 0.0→0.5 | `config.py:96` | ✅ | 100× stronger var push |
| `var(dim=0).mean()`→`var()` | `model.py:1508` | ✅ | 28× less dilution |
| var_loss damping (N factor) | — | ❌ | still ÷2687 |
| benefit_loss | `model.py` | ✅ | gate→log_scale anchor |
| benefit auto-scale by EMA magnitude | `model.py` | ✅ | phases in smoothly |
| benefit_weight=5.0 | `config.py` | ✅ | strong enough gradient |

## Remaining Issues

1. **linspace(-1.0, 1.0)** too aggressive — exp(1.0) = 2.72 destabilizes. Need
   gentler init that still allows gate→benefit→log_scale bootstrap.

2. **var loss ÷ N** — all variance-based losses divide by N, making per-element
   gradient tiny. No clean fix within MSE framework.

3. **div_loss vs benefit_loss competition** — div_loss pushes all elements apart,
   benefit_loss pushes per-expert means to gate target. They can conflict when
   the gate-implied ordering disagrees with variance maximization.

4. **Per-dim dead zone** — benefit_loss can only push per-expert mean, not
   individual d-dimensions. d-dim specialization has no direct gradient source.

## Questions for External Audit

1. Should log_scale be per-expert scalar (G,) instead of per-dim (G, d)?
   Current rationale: each d-dimension corresponds to a different K-subspace
   within the expert, so sub-expert specialization requires per-dim scales.
   But no gradient source provides per-dim differentiation.

2. Is a direct in-place update (not through loss) acceptable for log_scale?
   Similar to BatchNorm running stats: `log_scale += η * benefit` per step.
   This bypasses optimizer momentum but provides immediate, strong, correct signal.

3. Alternative: replace MSE/var losses with contrastive objectives.
   Instead of "pull ls_mean toward target", use "ls_mean should be ORDERED by gate".
   Pairwise ranking loss: `max(0, gate_j - gate_i) * (ls_mean_i - ls_mean_j)`.
   This naturally scales with N (order, not magnitude) and doesn't divide.

4. Should we use learnable per-expert bias for log_scale init within linspace?
   Each expert gets `ls_init_g = uniform(-1, 1)` instead of linspace.
   More random, less structured, but still provides variance.
