"""λ_d hierarchy: all constants derived from generalized golden ratio.

λ_d = positive root of x^d = x^{d-1} + ... + 1

For d=2: φ = (1+√5)/2 ≈ 1.618
For d≥2: converges to 2.0 as d→∞

Every empirical constant in the system becomes a derived function of λ_d,
eliminating manual tuning. The choice of d sets the "aggressiveness" of
the hierarchy — higher d → larger λ → more aggressive learning rates.
"""

import math
import numpy as np
import torch

_LAMBDA_CACHE: dict[int, float] = {}


def lambda_d(d: int) -> float:
    """Positive root of x^d = x^{d-1} + ... + 1.

    Fixed-point iteration: x_{k+1} = 2 - x_k^{-d}, converges in <30 steps.
    """
    if d < 2:
        d = 2
    if d in _LAMBDA_CACHE:
        return _LAMBDA_CACHE[d]

    x = 2.0
    for _ in range(100):
        x_new = 2.0 - 1.0 / (x ** d)
        if abs(x_new - x) < 1e-14:
            _LAMBDA_CACHE[d] = x_new
            return x_new
        x = x_new
    _LAMBDA_CACHE[d] = x
    return x


def fib(n: int) -> int:
    """Classical Fibonacci: F_0=0, F_1=1, F_2=1, F_3=2, ..."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def generalized_fib(n: int, d: int = 2) -> int:
    """d-step Fibonacci: F^(d)_n = Σ_{k=1..d} F^(d)_{n-k}.

    F^(d)_0 = 1, F^(d)_{-k}=0 for 1≤k≤d-1.
    For d=2: F^(2)_n = F_{n+1}.
    """
    if n < 0:
        return 0
    if n == 0:
        return 1
    seq = [0] * (d - 1) + [1]
    for _ in range(n):
        seq.append(sum(seq[-d:]))
    return seq[-1]


# ─── Λ_d class: all derived constants ──────────────────────────────────

class LambdaConfig:
    """All model hyperparameters derived from λ_d.

    Usage:
        lc = LambdaConfig(d=3)  # λ₃ ≈ 1.839
        lr = lc.base_lr          # λ⁻² ≈ 0.296
        tau = lc.memory_tau_hi   # F₂₁ ≈ 10946

    Conversion between λ_d values and common hyperparameter ranges:
        λ⁻²    → exploration threshold, min LR ratio (~0.1-0.4)
        λ⁻⁴    → differentiation threshold (~0.04-0.15)
        λ⁻¹    → base decay, EMA alpha baseline (~0.5-0.7)
        1-λ⁻²  → max write rate (~0.6-0.9)
        F_n    → buffer sizes, eval intervals, time constants
    """

    def __init__(self, d: int = 3):
        self._d = d
        self._lam = lambda_d(d)

    # ─── Core powers ───────────────────────────────────────────────

    @property
    def lam(self) -> float:
        return self._lam

    @property
    def lam_inv(self) -> float:
        return 1.0 / self._lam

    @property
    def lam_inv_sq(self) -> float:
        return 1.0 / (self._lam * self._lam)

    @property
    def lam_inv_cu(self) -> float:
        return 1.0 / (self._lam ** 3)

    @property
    def lam_inv_4(self) -> float:
        return 1.0 / (self._lam ** 4)

    @property
    def lam_sq(self) -> float:
        return self._lam * self._lam

    # ─── AdaptiveController thresholds ─────────────────────────────

    @property
    def exploration_threshold(self) -> float:
        """|mirror| threshold: λ⁻² ≈ 0.30 at d=3."""
        return max(0.1, self.lam_inv_sq)

    @property
    def differentiation_threshold(self) -> float:
        """var(log_scale) threshold: λ⁻⁴ ≈ 0.087 at d=3."""
        return max(0.02, self.lam_inv_4)

    # ─── Memory / gate ranges ──────────────────────────────────────

    @property
    def g_b_i_low(self) -> float:
        """b_i when expl=0: -(λ + 1/λ)."""
        return -(self._lam + self.lam_inv)

    @property
    def g_b_i_high(self) -> float:
        """b_i when expl=1: -λ⁻¹."""
        return -self.lam_inv

    @property
    def g_b_d_low(self) -> float:
        """b_d when expl=0 (longest memory): λ + 1/λ."""
        return self._lam + self.lam_inv

    @property
    def g_b_d_high(self) -> float:
        """b_d when expl=1 (shortest memory): λ - 1/λ."""
        return self._lam - self.lam_inv

    @property
    def mem2v_scale_min(self) -> float:
        """Minimum memory: λ⁻¹ (high diff → trust mirror → cut memory)."""
        return self.lam_inv

    @property
    def mem2v_scale_max(self) -> float:
        """Maximum memory: 1.0."""
        return 1.0

    @property
    def ema_alpha_min(self) -> float:
        """Fastest EMA: 1 - λ⁻⁶ ≈ 0.974 at d=3 (short ~40-step TC)."""
        return max(0.5, 1.0 - 1.0 / (self._lam ** 6))

    @property
    def ema_alpha_max(self) -> float:
        """Slowest EMA: 1 - λ⁻⁸ ≈ 0.992 at d=3 (long ~130-step TC)."""
        return max(0.5, 1.0 - 1.0 / (self._lam ** 8))

    @property
    def noise_scale_min(self) -> float:
        """Minimum noise: λ⁻⁸ ≈ 0.0076 at d=3."""
        return max(0.0005, 1.0 / (self._lam ** 8))

    @property
    def noise_scale_max(self) -> float:
        """Maximum noise: λ⁻⁶ ≈ 0.026 at d=3."""
        return max(0.002, 1.0 / (self._lam ** 6))

    @property
    def delta_var_ema_min(self) -> float:
        """Fast delta_var EMA: 1 - λ⁻⁴ ≈ 0.913 at d=3 (~12-step TC)."""
        return max(0.6, 1.0 - self.lam_inv_4)

    @property
    def delta_var_ema_max(self) -> float:
        """Slow delta_var EMA: 1 - λ⁻⁸ ≈ 0.992 at d=3 (~130-step TC)."""
        return max(0.6, 1.0 - 1.0 / (self._lam ** 8))

    # ─── Learning / scheduler ──────────────────────────────────────

    @property
    def warmup_steps(self) -> int:
        """Warmup: F_10 * lambda = 55 * lam."""
        return max(50, int(fib(10) * self._lam))

    @property
    def max_decay_steps(self) -> int:
        """Max decay: F_18 = 2584."""
        return fib(18)

    @property
    def target_var(self) -> float:
        """Target log_scale variance: λ⁻³ ≈ 0.16 at d=3."""
        return self.lam_inv_cu

    @property
    def mag_threshold(self) -> float:
        """Mirror magnitude threshold: λ⁻² ≈ 0.30 at d=3."""
        return self.lam_inv_sq

    @property
    def lr_min_ratio(self) -> float:
        """Minimum LR: λ⁻⁶ ≈ 0.026 at d=3."""
        return max(0.01, 1.0 / (self._lam ** 6))

    @property
    def var_min_for_lr_decay(self) -> float:
        """LR decay trigger: λ⁻⁸ ≈ 0.0076 at d=3."""
        return max(1e-5, 1.0 / (self._lam ** 8))

    # ─── Optimizer ─────────────────────────────────────────────────

    @property
    def gate_lr_mult(self) -> float:
        """Gate LR multiplier: λ² / (λ - 1) ≈ 3.38/0.84 = 4.02 at d=3.

        Derived from: faster learning for params that control
        temporal dynamics requires 1/(1-λ⁻¹) boosting.
        """
        return max(1.5, self.lam_sq / (self._lam - 1.0))

    @property
    def pred_weight_max(self) -> float:
        """Max alpha loss weight: λ⁻² ≈ 0.30 at d=3."""
        return self.lam_inv_sq

    @property
    def pred_weight_min(self) -> float:
        """Min alpha loss weight: λ⁻⁶ ≈ 0.026 at d=3."""
        return max(0.01, 1.0 / (self._lam ** 6))

    # ─── Mirror modulation ranges ──────────────────────────────────

    @property
    def tanh_bias_mod_max(self) -> float:
        """Max tanh bias amplification: 1 + λ⁻² ≈ 1.30 at d=3."""
        return 1.0 + self.lam_inv_sq

    @property
    def spectral_mod_lo(self) -> float:
        """Min spectral modulation: 1 - λ⁻⁴ ≈ 0.913 at d=3."""
        return max(0.7, 1.0 - self.lam_inv_4)

    @property
    def spectral_mod_hi(self) -> float:
        """Max spectral modulation: 1 + λ⁻⁴ ≈ 1.087 at d=3."""
        return 1.0 + self.lam_inv_4

    @property
    def pred_scale_mod_lo(self) -> float:
        """Min per-expert pred scale: λ⁻⁴ ≈ 0.087 at d=3."""
        return self.lam_inv_4

    @property
    def pred_scale_mod_hi(self) -> float:
        """Max per-expert pred scale: λ² ≈ 3.38 at d=3."""
        return self.lam_sq

    # ─── Buffer / interval sizes (Fibonacci numbers) ───────────────

    @property
    def eval_interval(self) -> int:
        """Eval every F_13 = 233 steps."""
        return max(100, fib(13))

    @property
    def save_interval(self) -> int:
        """Save every F_16 = 987 steps."""
        return max(500, fib(16))

    @property
    def log_interval(self) -> int:
        """Log every F_10 = 55 steps."""
        return max(50, fib(10))

    @property
    def patience(self) -> int:
        """Patience = F_20 = 6765."""
        return fib(20)

    # ─── Init values ───────────────────────────────────────────────

    @property
    def log_scale_init_std(self) -> float:
        """log_scale init std: λ⁻⁴ ≈ 0.087 at d=3."""
        return max(0.01, self.lam_inv_4)

    @property
    def conv_init_std(self) -> float:
        """Conv init std: λ⁻⁶ ≈ 0.026 at d=3."""
        return max(0.001, 1.0 / (self._lam ** 6))

    @property
    def w_d_init_std(self) -> float:
        """w_d init std: λ⁻⁴ ≈ 0.087 at d=3."""
        return max(0.01, self.lam_inv_4)

    # ─── Spectral radius diagnostic ───────────────────────────────

def spectral_radius(model, h, n_steps=20, n_iters=1):
    """Power iteration estimate of ρ(J) where J = ∂F/∂h at h.
    
    ρ(J) ≈ lim_{n→∞} ‖J^n v‖ / ‖J^{n-1} v‖ via power iteration.
    Uses torch.autograd.functional.jvp for O(D) per step (no D×D matrix).
    
    Edge of chaos: ρ ≈ 1.0 (Langton, 1990).
    Training (teacher forcing): ρ ≈ 1.0 expected.
    Inference (autoregressive): ρ > 1 → exponential error growth.
    
    Args:
        model: WideBindStack or any nn.Module
        h: (B, L, D) input tensor with requires_grad=True
        n_steps: power iteration steps
        n_iters: repeat with new random v and average (for reliability)
    
    Returns:
        float: estimated spectral radius
    """
    device = h.device
    dtype = h.dtype
    rhos = []
    with torch.no_grad():
        for _ in range(n_iters):
            v = torch.randn_like(h)
            for _ in range(n_steps):
                _, jvp = torch.autograd.functional.jvp(
                    lambda z: model(z, state=None)[0], h, v,
                    create_graph=False)
                v = jvp / (jvp.norm() + 1e-10)
            _, jvp = torch.autograd.functional.jvp(
                lambda z: model(z, state=None)[0], h, v,
                create_graph=False)
            # Rayleigh quotient: ρ ≈ (Jv)·v / ‖v‖²
            rho = (jvp * v).sum() / (v * v).sum().clamp(min=1e-10)
            rhos.append(rho.abs().item())
    return sum(rhos) / len(rhos)


# ─── Comparison helpers ────────────────────────────────────────

    def summary(self) -> dict:
        """Return all derived values as a flat dict."""
        return {k: getattr(self, k) for k in dir(self)
                if isinstance(getattr(self, k, None), (int, float))
                and not k.startswith('_')}

    @staticmethod
    def print_comparison(d: int = 3):
        """Print λ_d-derived values vs current WideBind defaults."""
        lc = LambdaConfig(d)
        from .config import WideBindConfig
        cfg = WideBindConfig()
        print(f'λ_{d} = {lc.lam:.6f}')
        print(f'  {"Parameter":<35} {"Old":>10} {"New":>10}')
        print(f'  {"─"*35} {"─"*10} {"─"*10}')
        pairs = [
            ('exploration_threshold', cfg.exploration_threshold,
             lc.exploration_threshold),
            ('differentiation_threshold', cfg.differentiation_threshold,
             lc.differentiation_threshold),
            ('w_mem2v_scale_min', cfg.w_mem2v_scale_min,
             lc.mem2v_scale_min),
            ('w_mem2v_scale_max', cfg.w_mem2v_scale_max,
             lc.mem2v_scale_max),
            ('ema_alpha_min', cfg.ema_alpha_min, lc.ema_alpha_min),
            ('ema_alpha_max', cfg.ema_alpha_max, lc.ema_alpha_max),
            ('noise_scale_min', cfg.noise_scale_min, lc.noise_scale_min),
            ('noise_scale_max', cfg.noise_scale_max, lc.noise_scale_max),
            ('delta_var_ema_min', cfg.delta_var_ema_min,
             lc.delta_var_ema_min),
            ('delta_var_ema_max', cfg.delta_var_ema_max,
             lc.delta_var_ema_max),
            ('warmup_steps', cfg.warmup_steps, lc.warmup_steps),
            ('target_var', cfg.target_var, lc.target_var),
            ('mag_threshold', cfg.mag_threshold, lc.mag_threshold),
            ('lr_min_ratio', cfg.lr_min_ratio, lc.lr_min_ratio),
            ('gate_lr_mult', cfg.gate_lr_mult, lc.gate_lr_mult),
            ('log_scale_init_std', cfg.log_scale_init_std,
             lc.log_scale_init_std),
            ('conv_init_std', cfg.conv_init_std, lc.conv_init_std),
            ('w_d_init_std', cfg.w_d_init_std, lc.w_d_init_std),
        ]
        for name, old, new in pairs:
            print(f'  {name:<35} {old:>10.4f} {new:>10.4f}')
