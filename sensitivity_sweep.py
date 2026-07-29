"""
Sensitivity sweep     parameter dependency analysis for WideBand Mini.

Usage:
    python sensitivity_sweep.py              # default: show full matrix
    python sensitivity_sweep.py --sweep div_weight 0 1 0.1   # sweep div_weight [0,1]
    python sensitivity_sweep.py --table                        # latex-style table
"""

import numpy as np
import math, sys, textwrap
import torch

#           System                                                                                                                                                                                                 

class WideBandDependencyModel:
    """
                                                                                                                                 .
                                                                :
      - core/stack.py:   compute_losses  (div, balance, ranking, reinforce, ...)
      - core/mirror.py:  alpha_update, homeostatic_boost, gate_logits
      - train.py:        spectral_alignment, phase_scaling

                                                                                                             .
    """

    def __init__(self):
        #           Default config (WideBand Mini)                                                                                     
        self.D = 896
        self.G = 8
        self.k = 32
        self.d = self.D // self.G  # 112
        self.n_layers = 12
        self.lr = 3e-4

        #           Parameters                                                                                                                                           
        self.params = {
            # Loss weights
            'div_weight':         0.087,   #               push log_scale variance
            'ranking_weight':     0.01,    # pairwise order ls_mean by gate_usage
            'balance_weight':     0.026,   #               HHI-based load balancing
            'reinforce_weight':   0.001,   # MSE(gate, usefulness)
            'gate_l1_weight':     0.0001,  # L1 penalty on expert gates
            'log_scale_l2_weight': 0.01,   # L2 on exp(log_scale) > 10

            # Init
            'log_scale_init_range': 0.6,   # linspace(-0.3, 0.3) = range 0.6

            # Alpha
            'alpha_lerp_rate':   0.01,     # lerp rate for alpha update
            'alpha_sigmoid_center': 2.2,   # sigmoid(2.2 - log(relative_var))

            # Spectral alignment
            'cos_sim_ce_aux':    0.0,      # cos_sim(CE grad, aux grad)     TYPICALLY 0 for orthog
            'cos_sim_ce_ranking': 0.0,     # same for ranking specifically

            # Balance suppressors
            'intra_weight_factor': math.sqrt(112 / 8),  # sqrt(d/G)     3.74
            'hhi_suppression':   1.0,      # how much HHI suppresses gate_var

            # Phase scaling
            'mirror_ratio_mean': 0.18,     # mirror_grad_norm / base_grad_norm (typical)
            'mirror_ratio_std':  0.1,      # std of ratio for sigmoid normalization

            # Homeostatic boost
            'boost_strength':    1.0,      # gate boost for unusual log_scale when ls_var<0.05

            # Learning
            'lr':                3e-4,

            # Ranking loss internal
            'rank_margin':       0.1,      # margin in pairwise hinge
        }

        #           Current metrics (computed from params)                                                       
        self._compute_metrics()

    def _compute_metrics(self):
        p = self.params

        # 1. log_scale variance from init
        #    linspace(-r/2, r/2) where r = log_scale_init_range
        #    var = r^2 / 12 (uniform dist variance)
        r = p['log_scale_init_range']
        self.metrics = {}
        self.metrics['ls_var_init'] = r**2 / 12

        # 2. Div gradient magnitude on log_scale (per step)
        #       div/   ls = -div_weight * 2 * (ls - ls.mean) / G
        #    Typical |ls - ls.mean| = sqrt(ls_var_init) = r/sqrt(12)
        #    |grad_div| = div_weight * 2 * sqrt(ls_var_init) / G
        ls_std = math.sqrt(self.metrics['ls_var_init'])
        intra = p['intra_weight_factor']

        # Cross-expert gradient
        grad_div_cross = p['div_weight'] * 2 * ls_std / self.G
        # Cross-dim gradient
        grad_div_intra = p['div_weight'] * intra * 2 * ls_std / self.d
        self.metrics['grad_div_magnitude'] = grad_div_cross + grad_div_intra

        #   ls per step from div
        self.metrics['delta_ls_per_step_div'] = self.lr * grad_div_cross
        self.metrics['delta_ls_2000steps_div'] = self.metrics['delta_ls_per_step_div'] * 2000
        self.metrics['ls_var_after_2000'] = (
            self.metrics['ls_var_init'] +
            self.metrics['delta_ls_2000steps_div']**2
        )

        # 3. Ranking loss effect on ls_mean
        #    ranking_loss = max(0, margin - (ls_high - ls_low))
        #    Only affects when ls_mean diff < margin
        #    Spectral alignment: scale = cos_sim * ||CE||/||rank||
        #    If cos_sim=0: scale=0     no ranking effect
        cs_rank = p['cos_sim_ce_ranking']
        ce_norm = 10.0  # typical CE grad norm
        rank_norm = 1.0  # typical ranking grad norm
        spectral_scale_rank = max(0, min(10.0, cs_rank)) * ce_norm / (rank_norm + 1e-8)
        self.metrics['spectral_scale_ranking'] = spectral_scale_rank
        self.metrics['ranking_grad_effective'] = p['ranking_weight'] * spectral_scale_rank
        self.metrics['ranking_working'] = spectral_scale_rank > 0.05

        # 4. Balance loss effect on gate
        #    Same spectral issue
        cs_bal = p['cos_sim_ce_aux']
        ce_norm = 10.0
        bal_norm = 2.0
        spectral_scale_bal = max(0, min(10.0, cs_bal)) * ce_norm / (bal_norm + 1e-8)
        self.metrics['spectral_scale_balance'] = spectral_scale_bal
        self.metrics['balance_grad_effective'] = p['balance_weight'] * spectral_scale_bal * p['hhi_suppression']

        # 5. Gate variance equilibrium
        #    gate_logits ~ N(0, sigma_init)     sigmoid     gate     0.5
        #    variance from init: w_gate ~ N(0, 1/sqrt(k))     gate_logits var = k * var(w_gate)     1
        #    sigmoid(0    1)     [0.27, 0.73]     gate_var     0.031
        #    balance pushes towards uniform     suppresses gate_var
        #    With balance_weight and spectral, effective suppression:
        bal_eff = self.metrics['balance_grad_effective']
        self.metrics['gate_var_equilibrium'] = 0.031 / (1 + bal_eff * 50)
        #    With no balance or spectral blocking: gate_var can grow
        self.metrics['gate_var_no_balance'] = 0.031 * (1 + p['reinforce_weight'] * 10)

        # 6. Spectral alignment for all aux (non-div)
        cs = p['cos_sim_ce_aux']
        # Only the div loss bypasses spectral alignment
        self.metrics['spectral_scale_all_aux'] = max(0, min(10.0, cs)) * ce_norm / (5.0 + 1e-8)
        self.metrics['all_aux_blocked'] = self.metrics['spectral_scale_all_aux'] < 0.05

        # 7. Gate homeostatic boost effect
        #    When ls_var < 0.05: boost = boost_strength * sigmoid(3.0 * ls_dev)
        #    ls_dev = ls.mean(dim=-1) - ls.mean()
        #    With current ls_var=0.036, boost     0.5 * boost_strength
        self.metrics['homeo_boost_active'] = self.metrics['ls_var_init'] < 0.05
        self.metrics['homeo_boost_strength'] = p['boost_strength'] * 0.5

        # 8. Phase scaling on mirror
        #    mir_s = sigmoid((ratio - ratio_ema) / (ratio_std + 1e-8))
        #    clamped to [0.2, 2.0]
        ratio = p['mirror_ratio_mean']
        ratio_ema = 0.15  # typical EMA after many steps (0.99 decay, starts at 0)
        ratio_std = p['mirror_ratio_std']
        z = (ratio - ratio_ema) / (ratio_std + 1e-8)
        mir_s = 1.0 / (1.0 + math.exp(-z))
        self.metrics['phase_mirror_scale'] = max(0.2, min(2.0, mir_s))
        self.metrics['phase_suppresses_mirror'] = self.metrics['phase_mirror_scale'] < 0.8

        # 9. Alpha adaptation rate
        #    alpha_target = sigmoid(2.2 - log(relative_var))
        #    lerp: alpha.data.lerp_(alpha_target, 0.01)
        #    residual_var comes from pred_error
        #    For typical pred_error: residual_var     0.5     relative_var     1 (all equal)
        #        alpha_target = sigmoid(2.2)     0.9
        #    Alpha delta per step: 0.01 * (0.9 - 0.85) = 0.0005
        self.metrics['alpha_target_typical'] = 1.0 / (1.0 + math.exp(-2.2))
        self.metrics['alpha_delta_per_step'] = p['alpha_lerp_rate'] * (0.9 - 0.85)
        self.metrics['alpha_delta_2000steps'] = self.metrics['alpha_delta_per_step'] * 2000

        # 10. LS_reg effect
        #    log_scale_reg = sum(exp(ls).clamp(max=10)) penalty
        #    Effect: prevents log_scale from exploding, indirectly limits variance
        self.metrics['ls_reg_suppression'] = p['log_scale_l2_weight'] * 2.0

        # 11. Combined: total effective gradient on log_scale
        self.metrics['total_ls_grad_per_step'] = (
            self.metrics['grad_div_magnitude'] * self.lr +
            self.metrics['ranking_grad_effective'] * self.lr * 0.1 +
            -self.metrics['ls_reg_suppression'] * self.lr * 0.01
        )

    def sensitivity(self, param_name, delta=0.1):
        """
        Compute    each_metric /    param for a given parameter.
        Returns dict {metric: derivative}
        """
        old_val = self.params[param_name]
        h = max(abs(old_val) * delta, 1e-10)

        self.params[param_name] = old_val + h
        self._compute_metrics()
        high = dict(self.metrics)

        self.params[param_name] = old_val - h
        self._compute_metrics()
        low = dict(self.metrics)

        self.params[param_name] = old_val
        self._compute_metrics()

        sens = {}
        for k in high:
            if k in low:
                sens[k] = (high[k] - low[k]) / (2 * h)
        return sens

    def display_sensitivity(self, param_name):
        """Print sensitivity of all metrics to a parameter."""
        sens = self.sensitivity(param_name)
        print(f"\n{'='*70}")
        print(f"SENSITIVITY:    metric /    {param_name}  (value={self.params[param_name]:.6f})")
        print(f"{'='*70}")
        for k, v in sorted(sens.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(v) > 1e-12:
                print(f"  {k:35s}  {v:+.6e}")

    def sweep_over(self, param_name, values):
        """Sweep a parameter over values, track all metrics."""
        results = []
        for v in values:
            self.params[param_name] = v
            self._compute_metrics()
            results.append({**self.metrics, param_name: v})
        # Restore
        return results

    def print_sweep_table(self, param_name, values, metrics=None):
        """Pretty-printed sweep table."""
        if metrics is None:
            # Auto-select most relevant metrics
            metrics = ['ls_var_init', 'grad_div_magnitude', 'delta_ls_per_step_div',
                       'spectral_scale_ranking', 'ranking_working', 'gate_var_equilibrium',
                       'homeo_boost_active', 'phase_mirror_scale']
        results = self.sweep_over(param_name, values)
        header = f"{param_name:>12s}" + "".join(f"{m:>18s}" for m in metrics)
        sep = "-" * len(header)
        print(f"\n{'='*len(header)}")
        print(f"Sweep: {param_name}")
        print(f"{'='*len(header)}")
        print(header)
        print(sep)
        for r in results:
            row = f"{r[param_name]:>12.4f}"
            for m in metrics:
                row += f"{r[m]:>18.6f}"
            print(row)
        print(sep)

    def full_matrix(self):
        """Print full dependency matrix: each param     each metric."""
        params = [k for k in self.params if k not in ('lr',)]
        metrics = ['ls_var_init', 'grad_div_magnitude', 'delta_ls_per_step_div',
                   'delta_ls_2000steps_div', 'spectral_scale_ranking',
                   'ranking_grad_effective', 'spectral_scale_balance',
                   'gate_var_equilibrium', 'gate_var_no_balance',
                   'homeo_boost_active', 'phase_mirror_scale',
                   'alpha_target_typical', 'total_ls_grad_per_step',
                   'all_aux_blocked', 'ranking_working']

        cols = len(metrics)
        # Column widths
        cw = [18] * cols

        print("\n" + "=" * (12 + sum(cw) + cols))
        h = f"{'PARAM':>12s}" + "".join(f"{m[:18]:>18s}" for m in metrics)
        print(h)
        print("=" * (12 + sum(cw) + cols))

        for p in params:
            sens = self.sensitivity(p)
            row = f"{p:>12s}"
            for m in metrics:
                v = sens.get(m, 0.0)
                if abs(v) > 1e-6:
                    row += f"{v:>18.2e}"
                else:
                    row += f"{'   ':>18s}"
            print(row)

        print("=" * (12 + sum(cw) + cols))

    def describe(self):
        """Print current state analysis."""
        p = self.params
        m = self.metrics
        print("\n" + "=" * 65)
        print("WideBand Mini     Current Parameter Analysis")
        print("=" * 65)
        print(f"\nCONFIG:")
        print(f"  D={self.D}, G={self.G}, d={self.d}, k={self.k}")
        print(f"  lr={self.lr:.1e}")
        print(f"\nKEY PARAMETERS:")
        for k, v in sorted(p.items()):
            print(f"  {k:30s} = {v}")
        print(f"\nCURRENT METRICS:")
        for k, v in sorted(m.items()):
            print(f"  {k:40s} = {v:.6f}" if isinstance(v, float) else f"  {k:40s} = {v}")

        print(f"\n{'=' * 65}")
        print(f"ANALYSIS:")
        print(f"{'=' * 65}")

        # ls_var analysis
        print(f"\n  1. log_scale variance: ls_var={m['ls_var_init']:.4f}")
        print(f"     Div grad magnitude: {m['grad_div_magnitude']:.2e}")
        print(f"       ls/step from div:  {m['delta_ls_per_step_div']:.2e}")
        print(f"       ls/2000 steps:     {m['delta_ls_2000steps_div']:.4f}")
        print(f"     ls_var after 2000:  {m['ls_var_after_2000']:.4f}")
        if m['delta_ls_per_step_div'] < 1e-7:
            print(f"     >>> PROBLEM: div gradient too weak for meaningful change")
            needed = 0.4 / 2000  # target ls_var range / steps
            needed_div = needed / (self.lr * 2 * ls_std / self.G)
            print(f"     >>> Need div_weight     {needed_div:.2f} for visible effect in 2000 steps")

        # Spectral analysis
        print(f"\n  2. Spectral alignment:")
        print(f"     cos_sim(CE, aux)={p['cos_sim_ce_aux']:.2f}")
        print(f"     All non-div aux blocked: {m['all_aux_blocked']}")
        print(f"     Ranking working:         {m['ranking_working']}")
        print(f"     Ranking effective grad:  {m['ranking_grad_effective']:.2e}")
        if m['all_aux_blocked']:
            print(f"     >>> CRITICAL: spectral alignment blocks ALL non-div aux losses")

        # Gate variance
        print(f"\n  3. Gate variance:")
        print(f"     Equilibrium (with balance): {m['gate_var_equilibrium']:.4f}")
        print(f"     Equilibrium (no balance):   {m['gate_var_no_balance']:.4f}")
        if m['gate_var_equilibrium'] < 0.03:
            print(f"     >>> balance holds gate_var near init regardless of training")

        # Phase scaling
        print(f"\n  4. Phase scaling:")
        print(f"     Mirror gradient scale: {m['phase_mirror_scale']:.4f}")
        if m['phase_suppresses_mirror']:
            print(f"     >>> Phase scaling SUPPRESSES mirror (mirror_ratio below EMA)")

        # Alpha
        print(f"\n  5. Alpha adaptation:")
        print(f"     alpha_target_typical: {m['alpha_target_typical']:.4f}")
        print(f"     alpha_delta/step:     {m['alpha_delta_per_step']:.2e}")
        print(f"     alpha_delta/2000:     {m['alpha_delta_2000steps']:.4f}")
        if m['alpha_delta_per_step'] < 1e-5:
            print(f"     >>> alpha adaptation very slow (lerp_rate too low or target     current)")

        # Total
        print(f"\n  6. Total gradient on log_scale (per step): {m['total_ls_grad_per_step']:.2e}")
        print(f"\n{'=' * 65}")

    #           Deadlock detection                                                                                                                                        

    def find_deadlocks(self):
        """Identify circular dependencies and deadlocks."""
        m = self.metrics
        deadlocks = []

        # Deadlock 1: ranking needs gate_var, gate_var needs ranking
        if m['ranking_working'] == 0:
            deadlocks.append((
                "RANKING-GATE DEADLOCK",
                "ranking requires gate_usage variance to sort experts, "
                "but ranking is blocked by spectral alignment. "
                "Even if ranking worked: gate_var=0.031     ranking cannot distinguish experts     ranking_loss=0."
            ))

        # Deadlock 2: spectral blocks all aux
        if m['all_aux_blocked']:
            deadlocks.append((
                "SPECTRAL BLOCKADE",
                f"cos_sim(CE, aux)={self.params['cos_sim_ce_aux']:.2f}     scale={m['spectral_scale_all_aux']:.4f}     "
                "all non-div aux losses have near-zero effect. "
                "Only div_loss (bypassed) has any chance of working."
            ))

        # Deadlock 3: div too weak
        if m['delta_ls_per_step_div'] < 1e-7:
            deadlocks.append((
                "DIV TOO WEAK",
                f"div_weight={self.params['div_weight']}     grad_div={m['grad_div_magnitude']:.2e}     "
                f"  ls/step={m['delta_ls_per_step_div']:.2e}. "
                f"After 2000 steps:   ls={m['delta_ls_2000steps_div']:.6f}. "
                f"Invisible to training."
            ))

        # Deadlock 4: balance suppresses gate
        if m['gate_var_equilibrium'] < m['gate_var_no_balance'] * 0.5:
            deadlocks.append((
                "BALANCE SUPPRESSION",
                f"balance_weight={self.params['balance_weight']} suppresses gate_var from "
                f"{m['gate_var_no_balance']:.4f} to {m['gate_var_equilibrium']:.4f}. "
                "Gate cannot specialize."
            ))

        # Deadlock 5: phase suppresses mirror
        if m['phase_suppresses_mirror']:
            deadlocks.append((
                "PHASE SUPPRESSION",
                f"mirror_ratio={self.params['mirror_ratio_mean']} < EMA={0.15:.2f}     "
                f"phase_scale={m['phase_mirror_scale']:.4f} < 1.0. "
                "Mirror gradients are being suppressed by phase scaling."
            ))

        # Deadlock 6: homeo boost needs ls_var < 0.05 (circular)
        if m['ls_var_init'] < 0.05:
            deadlocks.append((
                "HOMEOSTATIC CIRCLE",
                f"ls_var={m['ls_var_init']:.4f} < 0.05     boost ACTIVE ({m['homeo_boost_strength']:.2f}). "
                "But boost affects gate, not log_scale directly. "
                "ls_var stays < 0.05     boost stays active     gate changes     no ls_var change     eternal boost."
            ))

        return deadlocks


# --- Numerical Simulator --------------------------------------------------

class Simulator:
    """
    Numerical ODE simulator of WideBand Mini dynamics.
    Tracks per-expert values of ls, gate, alpha over N steps.
    Uses real gradient equations from stack.py / mirror.py / train.py.
    
    State:
      ls_mean          (G,)    -- mean log_scale per expert (across d dims)
      ls_std_cross     scalar  -- cross-expert std of ls_mean
      gate_logit       (G,)    -- logit of gate probability per expert
      alpha            (G,k)   -- predictive alpha per expert per k-dim
      residual_var_ema (G,k)   -- EMA of pred_error variance
      
    Metrics tracked at each step:
      ls_var, gate_var, |1-alpha|_mean, gate_entropy, alpha_diag_mean
    """

    def __init__(self, cfg=None):
        c = cfg or {}
        self.G = c.get('G', 8)
        self.d = c.get('d', 112)
        self.k = c.get('k', 32)
        self.lr = c.get('lr', 3e-4)
        
        # Params from config
        self.div_weight = c.get('div_weight', 0.087)
        self.ranking_weight = c.get('ranking_weight', 0.0)
        self.balance_weight = c.get('balance_weight', 0.026)
        self.reinforce_weight = c.get('reinforce_weight', 0.001)
        self.gate_l1_weight = c.get('gate_l1_weight', 0.0001)
        self.log_scale_l2_weight = c.get('log_scale_l2_weight', 0.01)
        self.alpha_lerp_rate = c.get('alpha_lerp_rate', 0.01)
        self.boost_strength = c.get('boost_strength', 1.0)
        self.gate_repulse_weight = c.get('gate_repulse_weight', 0.0)
        self.alpha_novelty_weight = c.get('alpha_novelty_weight', 0.0)
        self.gate_bias_scale = c.get('gate_bias_scale', 0.0)
        self.ls_init_range = c.get('ls_init_range', 0.0)  # 0 = use expert_asymmetry formula
        self.cos_sim_ce_aux = c.get('cos_sim_ce_aux', 0.0)
        self.cos_sim_ce_ranking = c.get('cos_sim_ce_ranking', 0.0)
        self.intra_weight = c.get('intra_weight', math.sqrt(112 / 8))  # sqrt(d/G)
        
        # Spectral scale for non-div aux
        self.spectral_scale_aux = max(0, min(10.0, self.cos_sim_ce_aux)) * 10.0 / 5.0
        self.spectral_scale_ranking = max(0, min(10.0, self.cos_sim_ce_ranking)) * 10.0 / 1.0
        
        # Phase scaling params
        self.mirror_ratio_ema = 0.0
        self.mirror_ratio_std = 0.1
        
        # --- State init ----------------------------------------------
        self.reset()
    
    def reset(self):
        # log_scale: expert_asymmetry init (matches real model with expert_asymmetry=True)
        # ls = log(0.05 * 1.5^g) for g in [0..G-1], expanded to d dims
        if self.ls_init_range > 0:
            r = self.ls_init_range
            ls_base = torch.linspace(-r/2, r/2, self.G)
        else:
            ls_vals = [math.log(0.05 * (1.5 ** g)) for g in range(self.G)]
            ls_base = torch.tensor(ls_vals)
        self.ls = ls_base.unsqueeze(1).expand(self.G, self.d).clone() + torch.randn(self.G, self.d) * 0.01
        
        # gate_logit: random init -> sigmoid gives gate_var ~ 0.03
        self.gate_logit = torch.randn(self.G) * 0.5
        # Gate bias per expert: linspace breaks symmetry immediately
        # gate_bias_scale=1.5 -> gates from 0.18 to 0.82, gate_var ~ 0.06
        if self.gate_bias_scale > 0:
            self.gate_logit = self.gate_logit + torch.linspace(-self.gate_bias_scale, self.gate_bias_scale, self.G)
        
        # alpha: tau-based init + expert_asymmetry override (matches real model)
        alpha_init = torch.zeros(self.G, self.k)
        for g in range(self.G):
            init_alpha = 0.85 + (g / max(self.G - 1, 1)) * 0.14
            alpha_init[g] = init_alpha
        self.alpha = alpha_init.clone()
        
        # residual_var_ema: starts at 0.1 per dim
        self.residual_var_ema = torch.ones(self.G, self.k) * 0.1
        
        # gate_ema for ranking
        self.gate_ema = torch.zeros(self.G)
        
        # Phase ratio tracking
        self.mirror_ratio_ema = 0.0
        self.mirror_ratio_std = 0.1
        
        # Metrics log
        self.history = {
            'step': [], 'ls_var': [], 'gate_var': [], 'alpha_dev': [],
            'ls_std_cross': [], 'gate_mean': [], 'gate_entropy': [],
            'homeo_boost': [], 'phase_scale': [], 'div_raw': [],
            'mirror_ratio': [], 'alpha_target_mean': [],
        }
    
    def step(self, t):
        """One simulated training step."""
        
        # --- 1. CE gradient on log_scale (modeled as random walk) ----
        ls_grad_ce = torch.randn(self.G, self.d) * 0.01  # random CE noise, small
        
        # --- 2. Div gradient ----------------------------------------
        # div = -var(sigmoid(ls)) — bounded in [0, 0.25]
        # Chain rule: d(div)/d(ls) = dsig * d(div)/d(sig) with dsig vanishing at saturation
        # Naturally self-limiting: once sig(ls) -> 0 or 1, gradient vanishes
        sig_ls = torch.sigmoid(self.ls)
        sig_center_exp = sig_ls - sig_ls.mean(dim=0, keepdim=True)
        sig_center_dim = sig_ls - sig_ls.mean(dim=-1, keepdim=True)
        dsig = sig_ls * (1 - sig_ls)  # sigmoid derivative — vanishes at saturation
        ls_grad_div = (-self.div_weight * 2 * dsig * sig_center_exp / self.G
                       -self.div_weight * self.intra_weight * 2 * dsig * sig_center_dim / self.d)
        
        # --- 3. L2 regularization on log_scale (matches real model) ----
        # reg = relu(ls - 2.3)^2 * w_ls  — soft cap at 2.3, no lower bound
        # grad: 2 * (ls - 2.3) * w_ls for ls > 2.3, else 0
        excess = (self.ls - 2.3).clamp(min=0)
        ls_grad_reg = -self.log_scale_l2_weight * 2 * excess
        
        # --- 4. Ranking gradient on ls_mean (bypasses spectral in real model) ----
        ls_grad_ranking = torch.zeros(self.G, self.d)
        if self.ranking_weight > 0:
            # ranking_loss = sum_i sum_j relu( -(ls_i - ls_j) ) * [gate_j > gate_i]
            # Full pairwise: expert pair contributes iff gate orders opposite to ls orders
            ls_mean = self.ls.mean(dim=-1)
            gate_usage = torch.sigmoid(self.gate_logit)
            ls_pair = ls_mean.unsqueeze(1) - ls_mean.unsqueeze(0)  # G×G, + = ls_i > ls_j
            gate_pair = gate_usage.unsqueeze(1) - gate_usage.unsqueeze(0)  # + = gate_i > gate_j
            # Mask: pairs where gate_i > gate_j BUT ls_i <= ls_j (opposite ordering)
            # relu(-ls_pair) * (gate_pair > 0)
            # gradient: for expert i, sum over j where gate_i > gate_j and ls_i < ls_j
            #   grad_i += -1 (push ls_i up to fix ordering)
            #    grad_j += +1 (push ls_j down to fix ordering)
            mask_down = (gate_pair > 0).float() * (ls_pair < 0).float()  # gate_i > gate_j, ls_i < ls_j
            grad_per_pair = -mask_down + mask_down.t()  # asymmetry: grad_i = -sum_down, grad_j = +sum_up
            ls_mean_grad_ranking = grad_per_pair.sum(dim=1)
            # Scale by ranking_weight (the loss weight, 0.01 by default)
            ls_grad_ranking = (ls_mean_grad_ranking * self.ranking_weight).unsqueeze(1).expand(-1, self.d)
        
        # Combined ls gradient — ranking bypasses spectral, div is its own gradient
        ls_grad = (ls_grad_ce + ls_grad_div + ls_grad_reg + ls_grad_ranking)
        self.ls = self.ls - self.lr * ls_grad
        
        # --- 5. Alpha update ----------------------------------------
        # pred_error -> residual_var -> alpha_target -> lerp
        # As model learns, pred_error decreases -> residual_var decreases
        # Add per-expert noise to break symmetry (intelligent fix: "expert curiosity")
        pred_error_scale = max(0.01, 0.5 * math.exp(-t / 5000))
        # Expert-specific residual scaling (different subspaces have different learnability)
        expert_noise = torch.exp(torch.randn(self.G, 1) * 0.1)  # log-normal per-expert
        residual_var = torch.ones(self.G, self.k) * pred_error_scale * expert_noise
        self.residual_var_ema.mul_(0.99).add_(residual_var, alpha=0.01)
        
        rv = self.residual_var_ema
        rv_mean = rv.mean(dim=0, keepdim=True)
        relative_var = rv / (rv_mean + 1e-10)
        alpha_target = torch.sigmoid(2.2 - torch.log(relative_var))
        self.alpha = self.alpha + self.alpha_lerp_rate * (alpha_target - self.alpha)
        
        # Alpha novelty push: push per-expert alpha apart with adaptive gain
        # Auto-boost when alpha_std is small to break symmetry
        alpha_per_expert = self.alpha.mean(dim=-1)  # (G,) — mean alpha per expert
        alpha_std = alpha_per_expert.std()
        boost = max(1.0, 0.1 / (alpha_std + 0.01))
        adapted_w = self.alpha_novelty_weight * boost
        alpha_center = alpha_per_expert - alpha_per_expert.mean()
        alpha_novelty_grad = (+adapted_w * 2
                              * alpha_center.unsqueeze(1).expand(-1, self.k) / self.G)
        self.alpha = self.alpha + alpha_novelty_grad  # direct update (not lr-scaled)
        self.alpha = torch.clamp(self.alpha, 0.01, 0.99)
        
        # --- 6. Gate update -----------------------------------------
        # Balance gradient on gate_logit:
        # ?balance/?logit = bal_weight * G/(G-1) * 2 * (gate - 1/G) * gate * (1-gate)
        gate = torch.sigmoid(self.gate_logit)
        bal_gate_grad = (self.balance_weight * self.G / (self.G - 1)
                         * 2 * (gate - 1.0/self.G) * gate * (1 - gate))
        
        # Reinforce gradient: MSE(gate, usefulness) -> ?/?logit = 2*(gate - usefulness)*gate*(1-gate)/n
        # usefulness ? gate (approximate) -> small random gradient
        usefulness = torch.sigmoid(torch.randn(self.G) * 0.5 + 0.5)
        reinf_gate_grad = (self.reinforce_weight * 2 * (gate - usefulness)
                          * gate * (1 - gate))
        
        # Gate L1: ?L1/?logit = gate_l1_weight * gate * (1-gate) / G
        l1_gate_grad = self.gate_l1_weight * gate * (1 - gate) / self.G
        
        # CE gradient on gate: NOT just noise. Gate controls mirror output,
        # mirror output affects next hidden state, hidden state affects CE.
        # dCE/d(logit) = dCE/d(mirror) * mirror_raw * gate * (1-gate)
        # mirror_raw magnitude scales with exp(log_scale), so experts with
        # larger ls get stronger CE gradient on their gate.
        ce_gate_grad_raw = torch.randn(self.G) * 0.01  # subspace-specific CE component
        # LS coupling: higher ls -> more mirror impact -> stronger CE gate gradient
        ls_activation = torch.exp(self.ls.mean(dim=-1)).detach()
        ls_coupling = 0.02 * (ls_activation - ls_activation.mean()) / (ls_activation.std() + 1e-8)
        ce_gate_grad = ce_gate_grad_raw + ls_coupling
        
        # --- Gate repulsion: inverse of balance, pushes gates apart ---
        # L_rep = -gate.var() * w_rep
        # dL/dlogit = -w_rep * 2 * (gate - gate.mean()) / G  (sigmoid derivative absorbed)
        # Direct gradient on logit (like div does on ls), not scaled by gate*(1-gate)
        gate_rep_grad = (-self.gate_repulse_weight * 2 * (gate - gate.mean()) / self.G)
        
        # Only balance + reinforce + l1 go through spectral alignment
        # Gate repulsion is direct (bypasses spectral alignment, like div)
        gate_grad = (ce_gate_grad
                     + self.spectral_scale_aux * bal_gate_grad
                     + self.spectral_scale_aux * reinf_gate_grad
                     + self.spectral_scale_aux * l1_gate_grad
                     + gate_rep_grad)
        
        # Homeostatic boost (if ls_var < 0.05)
        ls_mean = self.ls.mean(dim=-1)
        ls_var = self.ls.var().item()
        boost = 0.0
        if ls_var < 0.05:
            ls_dev = ls_mean - ls_mean.mean()
            boost = self.boost_strength * torch.sigmoid(3.0 * ls_dev)
            gate_grad = gate_grad + boost
        
        self.gate_logit = self.gate_logit - self.lr * gate_grad
        
        # Update gate_ema
        gate = torch.sigmoid(self.gate_logit)
        self.gate_ema.mul_(0.99).add_(gate.detach(), alpha=0.01)
        
        # --- 7. Phase scaling update ---------------------------------
        # ratio = mirror_grad_norm / base_grad_norm
        # Approximate: mirror_grad_norm ? ls change, base_grad_norm ? 0.1
        mirror_norm = ls_grad.norm().item()
        base_norm = 0.1
        ratio = mirror_norm / (base_norm + 1e-8)
        self.mirror_ratio_ema = 0.99 * self.mirror_ratio_ema + 0.01 * ratio
        self.mirror_ratio_std = 0.99 * self.mirror_ratio_std + 0.01 * abs(ratio - self.mirror_ratio_ema)
        z = (ratio - self.mirror_ratio_ema) / (self.mirror_ratio_std + 1e-8)
        phase_scale = max(0.2, min(2.0, 1.0 / (1.0 + math.exp(-z))))
        
        # --- 8. Log metrics ------------------------------------------
        gate = torch.sigmoid(self.gate_logit)
        ls_mean = self.ls.mean(dim=-1)
        
        # div_raw = -(sigmoid(ls).var(dim=0).mean() + intra * sigmoid(ls).var(dim=-1).mean())
        sig_ls = torch.sigmoid(self.ls)
        div_raw = -(sig_ls.var(dim=0).mean() + self.intra_weight * sig_ls.var(dim=-1).mean())
        
        # gate_entropy
        g = gate.clamp(1e-10, 1-1e-10)
        ent = -(g * g.log() + (1-g) * (1-g).log()).mean()
        
        self.history['step'].append(t)
        self.history['ls_var'].append(self.ls.var().item())
        self.history['gate_var'].append(gate.var().item())
        self.history['alpha_dev'].append(self.alpha.mean(dim=-1).std().item())
        self.history['ls_std_cross'].append(ls_mean.std().item())
        self.history['gate_mean'].append(gate.mean().item())
        self.history['gate_entropy'].append(ent.item())
        self.history['homeo_boost'].append(boost.mean().item() if isinstance(boost, torch.Tensor) else boost)
        self.history['phase_scale'].append(phase_scale)
        self.history['div_raw'].append(div_raw.item() if isinstance(div_raw, torch.Tensor) else div_raw)
        self.history['mirror_ratio'].append(ratio)
        self.history['alpha_target_mean'].append(alpha_target.mean().item())
    
    def run(self, n_steps=2000, print_every=500):
        for t in range(n_steps):
            self.step(t)
            if print_every > 0 and t % print_every == 0:
                gv = self.history['gate_var'][-1]
                lv = self.history['ls_var'][-1]
                ad = self.history['alpha_dev'][-1]
                gv_str = f"gate_var={gv:.4f}" if gv < 10 else f"gate_var={gv:.2f}"
                print(f"  step={t:>5}  ls_var={lv:.4f}  {gv_str}  |1-a|={ad:.4f}")
    
    def summary(self):
        h = self.history
        n = len(h['step'])
        if n == 0:
            print("No data")
            return
        print(f"\nSimulation: {n-1} steps")
        print(f"  ls_var:      {h['ls_var'][0]:.4f} -> {h['ls_var'][-1]:.4f}")
        print(f"  gate_var:    {h['gate_var'][0]:.4f} -> {h['gate_var'][-1]:.4f}")
        print(f"  |1-alpha|:   {h['alpha_dev'][0]:.4f} -> {h['alpha_dev'][-1]:.4f}")
        print(f"  gate_entropy:{h['gate_entropy'][0]:.4f} -> {h['gate_entropy'][-1]:.4f}")
        print(f"  phase_scale: {h['phase_scale'][-1]:.4f}")
        print(f"  homeo_boost: {h['homeo_boost'][-1]:.4f}")
        print()

    def plot(self, save_path=None):
        """ASCII plot of key metrics over time."""
        h = self.history
        metrics = [
            ('ls_var', 'ls_var'),
            ('gate_var', 'gate_var'),
            ('alpha_dev', '|1-alpha|'),
            ('gate_entropy', 'entropy'),
        ]
        steps = h['step']
        
        for key, label in metrics:
            vals = h[key]
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-10:
                print(f"  {label}: all={vals[0]:.4f}")
                continue
            # 50-char ASCII chart
            n_samp = min(len(vals), 50)
            idx = [int(i * (len(vals)-1) / (n_samp-1)) for i in range(n_samp)] if n_samp > 1 else [0]
            scaled = [(v - lo) / (hi - lo) for v in vals]
            chars = '_.-:=+*#@'
            bar = ''.join(chars[min(int(s * (len(chars)-1)), len(chars)-1)] for s in [scaled[i] for i in idx])
            print(f"  {label:12s} [{lo:.4f},{hi:.4f}] |{bar}|")
        print()

    def run_config_sweep(self, configs, n_steps=1000):
        """Run multiple configs and compare final metrics."""
        print(f"\n{'='*80}")
        print(f"Config sweep: {len(configs)} configs, {n_steps} steps each")
        print(f"{'='*80}")
        print(f"{'cfg':>8s} {'ls_var':>10s} {'gate_var':>10s} {'|1-a|':>10s} {'phase':>8s} {'boost':>8s}")
        print(f"{'-'*56}")
        results = []
        for i, c in enumerate(configs):
            sim = Simulator(c)
            sim.run(n_steps, print_every=0)
            r = {
                'ls_var': sim.history['ls_var'][-1],
                'gate_var': sim.history['gate_var'][-1],
                'alpha_dev': sim.history['alpha_dev'][-1],
                'phase_scale': sim.history['phase_scale'][-1],
                'boost': sim.history['homeo_boost'][-1],
            }
            results.append(r)
            print(f"{i:>8d} {r['ls_var']:>10.4f} {r['gate_var']:>10.4f} {r['alpha_dev']:>10.4f} {r['phase_scale']:>8.4f} {r['boost']:>8.4f}")
        print(f"{'-'*56}")
        return results
    
    def diagnose(self):
        """Find broken mechanisms from simulation output."""
        h = self.history
        issues = []
        
        ls_var_delta = h['ls_var'][-1] - h['ls_var'][0]
        if abs(ls_var_delta) < 0.001:
            issues.append(f"ls_var STUCK: {h['ls_var'][0]:.4f} -> {h['ls_var'][-1]:.4f} (delta={ls_var_delta:.4f})")
        
        gate_var_delta = h['gate_var'][-1] - h['gate_var'][0]
        if gate_var_delta < 0.001:
            issues.append(f"gate_var NOT GROWING: {h['gate_var'][0]:.4f} -> {h['gate_var'][-1]:.4f}")
        
        alpha_dev = h['alpha_dev'][-1]
        if alpha_dev < 0.02:
            issues.append(f"alpha DEVICE TOO LOW: |1-alpha|={alpha_dev:.4f} (no specialization)")
        
        if h['homeo_boost'][-1] > 0.2:
            issues.append(f"HOMEOSTATIC BOOST PERMANENT: active={h['homeo_boost'][-1]:.2f} at ls_var={h['ls_var'][-1]:.4f}")
        
        phase = h['phase_scale'][-1]
        if phase < 0.8:
            issues.append(f"PHASE SUPPRESSING MIRROR: scale={phase:.4f}")
        
        return issues


#           CLI                                                                                                                                                                                                          

def main():
    model = WideBandDependencyModel()
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]

    if '--table' in flags or '--matrix' in flags:
        model.full_matrix()
        return

    if '--sweep' in flags and len(args) >= 4:
        param, lo, hi, step = args[0], float(args[1]), float(args[2]), float(args[3])
        values = np.arange(lo, hi + step/2, step).tolist()
        model.print_sweep_table(param, values)
        return

    if '--sensitivity' in flags and len(args) >= 1:
        param = args[0]
        model.display_sensitivity(param)
        return

    if '--deadlocks' in flags:
        model.describe()
        print("\n\nDEADLOCK ANALYSIS:")
        print("=" * 65)
        for name, desc in model.find_deadlocks():
            print(f"\n       {name}")
            print(f"     {textwrap.fill(desc, width=60, subsequent_indent='     ')}")
        return

    if '--sim' in flags:
        n_steps = int(args[0]) if args else 2000
        sim = Simulator()
        print(f"\n{'='*60}")
        print(f"Numerical simulation: {n_steps} steps (current config)")
        print(f"{'='*60}")
        sim.run(n_steps)
        sim.summary()
        sim.plot()
        issues = sim.diagnose()
        if issues:
            print("  BROKEN MECHANISMS:")
            for iss in issues:
                print(f"    ! {iss}")
        print()
        return

    if '--sim-sweep' in flags:
        n_steps = int(args[0]) if args else 1000
        configs = [
            {},  # current
            {'div_weight': 1.0},
            {'div_weight': 3.0},
            {'balance_weight': 0.0, 'div_weight': 1.0},
            {'cos_sim_ce_aux': 0.1, 'cos_sim_ce_ranking': 0.05},
            {'cos_sim_ce_aux': 0.1, 'cos_sim_ce_ranking': 0.05, 'div_weight': 1.0},
            {'cos_sim_ce_aux': 0.1, 'cos_sim_ce_ranking': 0.05, 'div_weight': 1.0, 'balance_weight': 0.0},
            {'ls_init_range': 1.0, 'div_weight': 1.0, 'balance_weight': 0.0, 'cos_sim_ce_aux': 0.1, 'cos_sim_ce_ranking': 0.05},
        ]
        sim = Simulator()
        sim.run_config_sweep(configs, n_steps)
        return

    # Default: full analysis
    model.describe()

    print("\n\nDEADLOCK DETECTION:")
    print("=" * 65)
    for name, desc in model.find_deadlocks():
        print(f"\n       {name}")
        print(f"     {textwrap.fill(desc, width=60, subsequent_indent='     ')}")

    print("\n\nSENSITIVITY MATRIX:")
    model.full_matrix()


if __name__ == '__main__':
    main()
