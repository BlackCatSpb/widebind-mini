"""Grouped Cognitive Mirror: G experts, each with K-space self-consistency."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vsa_utils import fib_sigmoid_init


class GroupedCognitiveMirror(nn.Module):
    """
    Ансамбль из 32 экспертов-зеркал, каждый в своём d=128 подпространстве (D=4096).
    Для D=896: d=28, d/k=3.5:1. Для D=4096: d=128, d/k=4:1.

    Каждый эксперт:
      - Имеет свой K-space (k=32) внутри своего d (128 для D=4096, 28 для D=896)
      - Вычисляет 4 сигнала коррекции: temp, pred, smooth, sym
      - Все 4 сигнала суммируются по всем k размерностям (без lo/hi split)
      - Даёт полный градиент pred_error всем k размерностям
      - Имеет свой tanh_bias + skip_connection + log_scale
      - Имеет meta-gate: учится доверять/игнорировать эксперта

    Predictive mirror:
      - alpha: scalar per expert, pred_k = alpha_g * hp_{t-1}
      - Вместо W_pred (G×k×k = 16K params) для сильного градиента
      - pred_error = hp_t - pred(hp_{t-1}) — ошибка предсказания
      - Обучает зеркало динамике VSA-состояния

    K-space Merge:
      - Все 4 сигнала (temp, pred, smooth, sym) суммируются по всем k размерностям
      - Без lo/hi split: полный градиент для всех сигналов
      - Замена W_pred на alpha + полный k-градиент = W_pred наконец учится

    Gradient-Adaptive Gate:
      - delta_var: running EMA variance дельты K-space
      - Эксперт с высокой variance активен, с низкой — прижат
      - Дополняет внешний grad_norm сигнал внутренней метрикой

    Внешний сигнал подкрепления:
      - prev_grad_norm: норма градиента по подпространству (c предыдущего backward)
      - Устанавливается извне через cache_grad_norms(grad_h) после backward

    Skip connection (alpha=0.1):
      - mirror = tanh(linear) + alpha * linear
      - Обеспечивает per-dim градиент для log_scale даже при насыщении tanh
    """

    def __init__(self, D, G=32, k=32, w_pred_scale_init=3.0, log_scale_init_std=0.05,
                 delta_var_ema_min=0.8, delta_var_ema_max=0.99, tie_mirror_proj=False,
                 layer_idx=0, n_layers=32, has_private_mem=False,
                 expert_asymmetry=False, meta_trust=False, gate_bias_scale=0.0,
                 alpha_novelty_weight=0.0):
        super().__init__()
        assert D % G == 0
        self.D = D
        self.G = G
        self.k = k
        self.d = D // G
        self.tie_mirror_proj = tie_mirror_proj
        phi = math.log(1 + layer_idx) / math.log(max(n_layers, 2))
        self.register_buffer('phi', torch.tensor(phi))

        proj_std = 1.0 / (self.d * k) ** 0.25

        if expert_asymmetry:
            W_p = torch.empty(G, self.d, k)
            for g in range(G):
                nn.init.orthogonal_(W_p[g])
            self.W_proj = nn.Parameter(W_p)
        else:
            self.W_proj = nn.Parameter(torch.randn(G, self.d, k) * proj_std)
        if tie_mirror_proj:
            self.register_buffer('W_out', torch.zeros(G, k, self.d))
            with torch.no_grad():
                self.W_out.copy_(self.W_proj.permute(0, 2, 1))
            self._hook = self.register_forward_pre_hook(
                lambda mod, args: mod._sync_W_out())
        else:
            self.W_out = nn.Parameter(torch.randn(G, k, self.d) * proj_std)

        self.w_temp = nn.Parameter(torch.randn(G, k))
        self.w_global = nn.Parameter(torch.randn(G, k))

        self.conv_smooth = nn.Conv1d(G * k, G * k, 3, padding=0,
                                      groups=G * k, bias=False)
        with torch.no_grad():
            self.conv_smooth.weight.zero_()
            self.conv_smooth.weight[:, :, 1] = 1.0

        self.w_sym_u = nn.Parameter(torch.randn(G, k))
        self.w_sym_v = nn.Parameter(torch.randn(G, k))

        tau_min, tau_max = 2.0, 200.0
        if k > 1:
            frac = torch.arange(k, dtype=torch.float32) / (k - 1)
            tau_k = tau_min * (tau_max / tau_min) ** frac
        else:
            tau_k = torch.tensor([(tau_min + tau_max) / 2])
        alpha_init = torch.exp(-1.0 / tau_k).view(1, k).expand(G, -1).clone()
        if expert_asymmetry and G > 1:
            for g in range(G):
                init_alpha = 0.15 + (g / (G - 1)) * 0.25
                alpha_init[g] = init_alpha
        self.alpha_diag = nn.Parameter(alpha_init)
        self.w_pred_scale_legacy = nn.Parameter(torch.ones(G, k))
        self.tanh_bias = nn.Parameter(torch.zeros(G, k))
        n_signals = 5 if has_private_mem else 4
        self.register_buffer('_signal_norm_ema', torch.ones(n_signals, G, k), persistent=False)
        if expert_asymmetry and G > 1:
            ls_vals = [math.log(0.05 * (1.5 ** g)) for g in range(G)]
            ls_base = torch.tensor(ls_vals).unsqueeze(1).expand(G, self.d)
        else:
            ls_base = torch.linspace(-0.3, 0.3, G).unsqueeze(1).expand(G, self.d)
        self.log_scale = nn.Parameter(ls_base + torch.randn(G, self.d) * 0.01)

        gate_std = 1.0 / (self.k + 1) ** 0.5
        self.w_gate = nn.Parameter(torch.randn(G, self.k) * gate_std)
        self.b_gate = nn.Parameter(torch.zeros(G))
        self.w_delta_gate = nn.Parameter(torch.randn(G, self.k) / math.sqrt(self.k))
        gate_bias_val = torch.linspace(-gate_bias_scale, gate_bias_scale, G)
        self.gate_bias = nn.Parameter(gate_bias_val)
        self._alpha_novelty_weight = alpha_novelty_weight

        self.register_buffer('_prev_grad_norm', torch.zeros(G), persistent=False)
        self._expert_asymmetry = expert_asymmetry
        self._meta_trust = meta_trust
        self._has_private_mem = has_private_mem
        self._pm_write_delay = 5000
        if has_private_mem:
            self.register_buffer('_private_mem', torch.randn(G, self.k) * 0.01)
            self.register_buffer('_pm_step', torch.zeros(1, dtype=torch.long), persistent=False)
            self.w_help = nn.Parameter(torch.full((G, 1), math.log(3.0)))
            self.w_contra = nn.Parameter(torch.full((G,), 0.01))
            self.register_buffer('_concept_sim_ema', torch.eye(G), persistent=False)
            self.register_buffer('_behavior_div_ema', torch.zeros(G, G), persistent=False)
            self.register_buffer('_trust_matrix', torch.eye(G) * 0.5, persistent=False)
            if meta_trust:
                self.register_buffer('_prev_trust_matrix', torch.eye(G) * 0.5, persistent=False)
                self.register_buffer('_meta_private_mem', torch.zeros(G), persistent=False)
        self.register_buffer('_hp_grad', torch.zeros(G), persistent=False)
        self.register_buffer('_delta_var', torch.zeros(G), persistent=False)
        self.register_buffer('_fwd_count', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('_last_magnitude', torch.zeros(1), persistent=False)
        self.register_buffer('_last_gates', torch.zeros(G), persistent=False)
        self.register_buffer('_last_h_pool', torch.zeros(G, self.d), persistent=False)
        self.register_buffer('_gate_ema', torch.zeros(G), persistent=True)
        self.register_buffer('_alpha_override', torch.zeros(1), persistent=False)
        self._cached_pred_k = None
        self._cached_hp = None
        self._cached_pred_error_norm = None
        self._cached_contra = None
        self._cached_disagreement = None
        self._cached_contra_graph = None
        self._cached_contra_expert = None
        self._cached_concept_dendrogram = None
        self._cached_dominance = None
        self._cached_isolation = None
        self.register_buffer('_residual_var_ema', torch.ones(G, k) * 0.1, persistent=False)

        rho = 0.6 ** layer_idx
        log_skip_init = math.log(0.10) + (math.log(17.0) - math.log(0.10)) * rho
        self.log_skip_alpha = nn.Parameter(torch.full((G,), log_skip_init))
        log_mod_init = -2.30 + (-0.81 - (-2.30)) * rho
        self.log_dvar_mod_scale = nn.Parameter(torch.full((G,), log_mod_init))
        self.log_grad_mod_scale = nn.Parameter(torch.full((G,), log_mod_init))
        self.dvar_mod_bias = nn.Parameter(torch.full((G,), -0.01))
        self.grad_mod_bias = nn.Parameter(torch.full((G,), -0.01))
        self._delta_var_ema_min = delta_var_ema_min
        self._delta_var_ema_max = delta_var_ema_max

        self._signal_log_weights = nn.Parameter(torch.zeros(n_signals))

        _pos_g = torch.Generator().manual_seed(12345)
        self.register_buffer('_pos_id_buf',
            torch.sign(torch.randn(1, 4096, 1, k, generator=_pos_g)), persistent=False)

        self.usefulness_predictor = nn.Sequential(
            nn.Linear(k, k),
            nn.Tanh(),
            nn.Linear(k, 1),
        )
        self.mod_scale_mlp = nn.Parameter(torch.full((G,), math.log(2.0)))
        self.mod_scale_mem = nn.Parameter(torch.full((G,), math.log(2.0)))
        self.register_buffer('_usefulness_temp', torch.tensor(2.0), persistent=False)
        self.register_buffer('_damp_tau', torch.tensor(0.1), persistent=False)

    def _sync_W_out(self):
        with torch.no_grad():
            self.W_out.copy_(self.W_proj.permute(0, 2, 1))

    def forward(self, h, mem_all, global_state=None, diff=None,
                tanh_bias_mod=1.0, pred_scale_mod=None, context_mem=None,
                allow_write=None):
        self._fwd_count += 1
        self._progress = 1.0 - math.exp(-self._fwd_count.item() / 200.0)
        B, L, D = h.shape
        G, d, k = self.G, self.d, self.k

        h_g = h.reshape(B, L, G, d)
        mem_g = mem_all.reshape(B, L, G, d)
        mc_g = mem_g.mean(dim=1, keepdim=True)

        hp = torch.einsum('blgd,gdk->blgk', h_g, self.W_proj)
        if hp.requires_grad:
            hp.register_hook(lambda g: (
                self._prev_grad_norm.copy_(g.detach().norm(dim=-1).mean(dim=(0, 1))),
                None
            )[1])
        mc_k = torch.einsum('b l gd,gdk->b l gk', mc_g, self.W_proj)

        # ─── Bipolar pos_id binding: hp = hp ⊛ pos_id ───
        # Детерминированный буфер позиционных кодов (broadcast на B, G).
        hp = hp * self._pos_id_buf[:, :L]

        hp_prev = torch.cat([torch.zeros_like(hp[:, 0:1]), hp[:, :-1]], dim=1)

        temp_k = (hp - mc_k) * self.w_temp

        if global_state is not None:
            gs_k = torch.einsum('b l gd,gdk->b l gk',
                                global_state.reshape(1, 1, G, d), self.W_proj)
            temp_k = temp_k + (hp - gs_k) * self.w_global

        alpha_eff = self.alpha_diag
        override = self._alpha_override.item()
        if override > 0:
            alpha_eff = (1 - override) * alpha_eff + override * 1.0
        pred_k = hp_prev * alpha_eff.view(1, 1, G, k)
        _pred_k_aux = pred_k
        if pred_scale_mod is None:
            dv = self._delta_var
            dv_mean = dv.mean().clamp(min=1e-8)
            pred_scale_mod = (dv / dv_mean).clamp(0.1, 3.0)
        hp_norm = hp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        raw_pred_error = hp - pred_k
        if not self.training:
            damp = torch.sigmoid(-raw_pred_error.norm(dim=-1).mean() / self._damp_tau)
            alpha_eff = 1.0 + (alpha_eff - 1.0) * damp
            pred_k = hp_prev * alpha_eff.view(1, 1, G, k)
        pred_error = (hp - pred_k) / hp_norm * pred_scale_mod.view(G, 1)
        if self.training:
            with torch.no_grad():
                override = self._alpha_override.item()
                if override < 0.1:
                    residual_var = pred_error.var(dim=(0, 1), unbiased=False)
                    self._residual_var_ema.lerp_(residual_var, 0.01)
                    rv = self._residual_var_ema
                    rv_mean = rv.mean(dim=0, keepdim=True)
                    relative_var = rv / (rv_mean + 1e-10)
                    alpha_target = torch.sigmoid(2.2 - torch.log(relative_var))
                    self.alpha_diag.data.lerp_(alpha_target, 0.01)
                    if self._alpha_novelty_weight > 0 and G > 1:
                        alpha_per_expert = self.alpha_diag.mean(dim=-1)
                        alpha_std = alpha_per_expert.std()
                        boost = max(1.0, 0.1 / (alpha_std + 0.01))
                        adapted_w = self._alpha_novelty_weight * boost
                        alpha_center = alpha_per_expert - alpha_per_expert.mean()
                        novelty_push = (adapted_w * 2
                                        * alpha_center.unsqueeze(1).expand(-1, k) / G)
                        self.alpha_diag.data.add_(novelty_push)
                        self.alpha_diag.data.clamp_(0.01, 0.99)
        self._cached_pred_k = _pred_k_aux
        self._cached_hp = hp
        pred_error_norm = (raw_pred_error / hp_norm).norm(dim=(-2, -1))
        self._cached_pred_error_norm = pred_error_norm

        if self._has_private_mem:
            uncert = torch.sigmoid(pred_error.abs())
            q = hp * uncert
            keys = self._private_mem.detach().clone()
            if context_mem is not None:
                keys = context_mem * 0.3 + keys * 0.7
                keys = F.normalize(keys, dim=-1) * self._private_mem.norm(dim=-1, keepdim=True)
            attn_logits = q @ keys.T / math.sqrt(self.k)
            attn = torch.sigmoid(attn_logits)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)
            help_k_base = attn @ keys
            hp_n = hp.norm(dim=-1).clamp(min=1e-8)
            disagreement = (hp - help_k_base).norm(dim=-1) / hp_n
            contra = torch.sigmoid(disagreement - 1.0)
            trust = 1.0 - contra
            help_k = help_k_base * torch.sigmoid(self.w_help).unsqueeze(0).unsqueeze(0)
            help_k = help_k * trust.unsqueeze(-1)
            self._cached_contra = contra.detach()
            self._cached_disagreement = disagreement.detach()
        else:
            help_k = torch.zeros_like(hp)
            trust = torch.ones_like(hp.norm(dim=-1))

        if self._has_private_mem and self.training:
            with torch.no_grad():
                pm = self._private_mem
                pm_norm = pm.norm(dim=-1, keepdim=True).clamp(min=1e-10)
                pm_n = pm / pm_norm
                concept_sim = pm_n @ pm_n.T
                self._concept_sim_ema.mul_(0.99).add_(concept_sim, alpha=0.01)
                hp_avg = hp.mean(dim=(0, 1))
                hp_n = F.normalize(hp_avg, dim=-1)
                behavior_sim = hp_n @ hp_n.T
                behavior_div = 1.0 - behavior_sim
                self._behavior_div_ema.mul_(0.99).add_(behavior_div, alpha=0.01)
                trust_weights = attn.mean(dim=(0, 1))
                if self._meta_trust and self._has_private_mem:
                    self._prev_trust_matrix.copy_(self._trust_matrix)
                self._trust_matrix.mul_(0.99).add_(trust_weights, alpha=0.01)
                if self._meta_trust and self._has_private_mem:
                    delta_tm = self._trust_matrix - self._prev_trust_matrix
                    instability = delta_tm.abs().mean(dim=1)
                    self._meta_private_mem.mul_(0.9).add_(instability, alpha=0.1)
                contra_g = concept_sim * behavior_div
                self._cached_contra_graph = contra_g
                contra_expert = contra_g.mean(dim=-1)
                self._cached_contra_expert = contra_expert
                sim_vals = concept_sim[~torch.eye(self.G, dtype=torch.bool, device=concept_sim.device)]
                q_hi = sim_vals.float().quantile(0.75)
                q_lo = sim_vals.float().quantile(0.25)
                self._cached_concept_dendrogram = (q_hi.item(), q_lo.item())
                dominance = self._trust_matrix.sum(dim=0)
                isolation = 1.0 - (self._trust_matrix.sum(dim=-1) / self.G)
                self._cached_dominance = dominance
                self._cached_isolation = isolation

        _write = self._has_private_mem and (self.training if allow_write is None else allow_write)
        if _write:
            self._pm_step += 1
            _write = self._pm_step.item() >= self._pm_write_delay
        if _write:
            with torch.no_grad():
                conf = torch.sigmoid(-pred_error.abs().mean(dim=-1, keepdim=True))
                contra_u = contra.unsqueeze(-1)
                contra_expert_coll = self._cached_contra_expert.to(conf.device).view(1, 1, G, 1)
                isolation_coll = self._cached_isolation.to(conf.device).view(1, 1, G, 1)
                social_pressure = 1.0 - 0.5 * torch.sigmoid(contra_expert_coll.clamp(min=0) + isolation_coll)
                conf_plastic = conf * (1.0 - contra_u) * social_pressure
                temp_write = 0.5
                conf_soft = conf_plastic ** temp_write
                conf_bc = conf_soft * self.G / (conf_soft.sum(dim=-2, keepdim=True) + 1e-8)
                weighted_hp = (conf_bc * hp.detach()).mean(dim=(0, 1))
                pm_scale = self._private_mem.norm(dim=-1).mean().clamp(min=1e-8)
                warmup_rate = torch.sigmoid(3.0 - pm_scale)
                pm_decay = 0.999 - 0.009 * warmup_rate
                self._private_mem.mul_(pm_decay).add_(weighted_hp, alpha=1.0 - pm_decay)
                self._private_mem.clamp_(-10.0, 10.0)

        hp_perm = hp.permute(0, 2, 3, 1).reshape(B, G * k, L)
        hp_pad = F.pad(hp_perm, (2, 0))
        hp_smooth = self.conv_smooth(hp_pad)
        hp_smooth = hp_smooth.reshape(B, G, k, L).permute(0, 3, 1, 2)
        smooth_k = hp - hp_smooth

        sym_k = (hp * self.w_sym_u) * (hp_prev * self.w_sym_v)

        if self._has_private_mem:
            signals = [temp_k, pred_error, smooth_k, sym_k, help_k]
        else:
            signals = [temp_k, pred_error, smooth_k, sym_k]
        signals_normed = []
        decay_ema = 0.01
        for i, s in enumerate(signals):
            if self.training:
                with torch.no_grad():
                    rms = s.norm(dim=(-2, -1), keepdim=True).mean(dim=(0, 1), keepdim=True)
                    self._signal_norm_ema[i].mul_(1 - decay_ema).add_(rms.squeeze(), alpha=decay_ema)
            s_norm = s / (self._signal_norm_ema[i].unsqueeze(0).unsqueeze(0) + 1e-8)
            signals_normed.append(s_norm)

        w = torch.sigmoid(self._signal_log_weights)

        n_sig = len(signals)
        decorr = 0.0
        npairs = 0
        for i in range(n_sig):
            for j in range(i + 1, n_sig):
                si = (signals_normed[i] * w[i]).reshape(-1, signals_normed[i].shape[-2] * signals_normed[i].shape[-1])
                sj = (signals_normed[j] * w[j]).reshape(-1, signals_normed[j].shape[-2] * signals_normed[j].shape[-1])
                si_c = si - si.mean(dim=0, keepdim=True)
                sj_c = sj - sj.mean(dim=0, keepdim=True)
                cos_ij = (si_c * sj_c).sum(dim=-1) / (si_c.norm(dim=-1) * sj_c.norm(dim=-1) + 1e-8)
                decorr = decorr + cos_ij.pow(2).mean()
                npairs = npairs + 1
        if npairs > 0:
            decorr = decorr / npairs
        self._cached_decorr = decorr

        delta = sum(w[i] * signals_normed[i] for i in range(n_sig))
        delta = F.rms_norm(delta, (delta.shape[-1],))
        delta = delta + self.tanh_bias * tanh_bias_mod

        grad_mod = torch.exp(self.log_grad_mod_scale) * torch.tanh(self._prev_grad_norm + self.grad_mod_bias)
        if self.training:
            with torch.no_grad():
                dvar = delta.var(dim=(0, 1), unbiased=False).mean(dim=-1)
                if diff is not None:
                    ema_alpha = self._delta_var_ema_min + diff * (self._delta_var_ema_max - self._delta_var_ema_min)
                else:
                    ema_alpha = 0.9
                self._delta_var.mul_(ema_alpha).add_(dvar * (1.0 - ema_alpha))
        dvar_mod = torch.exp(self.log_dvar_mod_scale) * torch.tanh(self._delta_var + self.dvar_mod_bias)

        usefulness_logits = self.usefulness_predictor(delta).squeeze(-1)
        prog = getattr(self, '_progress', 0.0)
        temp = max(0.3, 3.0 * math.exp(-prog * 2.0))
        with torch.no_grad():
            threshold = usefulness_logits.median(dim=-1, keepdim=True).values
        usefulness = torch.sigmoid((usefulness_logits - threshold) / temp)

        mlp_mod = usefulness * torch.sigmoid(self.mod_scale_mlp).view(1, 1, G)
        mem_mod = usefulness * torch.sigmoid(self.mod_scale_mem).view(1, 1, G)

        linear = torch.einsum('blgk,gkd->blgd', delta, self.W_out)
        skip_alpha = torch.exp(self.log_skip_alpha).view(1, 1, G, 1)
        mirror = torch.tanh(linear) + skip_alpha * linear
        # Adaptive scale: prevents saturation when delta is large
        # scale ∝ 1/(1 + ‖delta‖) per expert (broadcast across d)
        delta_norm = delta.norm(dim=-1, keepdim=True).mean(dim=(0, 1), keepdim=True).clamp(min=1e-8)
        adapt_scale = 1.0 / (1.0 + 0.1 * delta_norm)  # (1, 1, G, 1)
        mirror = mirror * torch.exp(self.log_scale) * adapt_scale

        gate_signal = torch.abs(pred_error)
        gate_logits = torch.einsum('blgk,gk->blg', gate_signal, self.w_gate) + self.b_gate
        gate_logits = gate_logits + self.gate_bias.unsqueeze(0).unsqueeze(0)
        delta_gate = torch.einsum('blgk,gk->blg', delta, self.w_delta_gate)
        gate_logits = gate_logits + delta_gate
        gate_logits = gate_logits + grad_mod.unsqueeze(0).unsqueeze(0)
        gate_logits = gate_logits + dvar_mod.unsqueeze(0).unsqueeze(0)
        if self._has_private_mem:
            gate_logits = gate_logits + disagreement * self.w_contra.unsqueeze(0).unsqueeze(0)
            if self._cached_contra_expert is not None:
                ce = self._cached_contra_expert.to(gate_logits.device).unsqueeze(0).unsqueeze(0)
                gate_logits = gate_logits + ce
            spec = self._behavior_div_ema.mean(dim=-1)
            spec = spec / (spec.max() + 1e-10)
            cons = self._concept_sim_ema.mean(dim=-1)
            cons = cons / (cons.max() + 1e-10)
            gate_bonus = (spec * 0.5 + cons * 0.5) * self.w_contra * 0.1
            gate_logits = gate_logits + gate_bonus.unsqueeze(0).unsqueeze(0)
        if self._meta_trust and self._has_private_mem:
            p = self._meta_private_mem.unsqueeze(0).unsqueeze(0)
            gate_logits = gate_logits - 0.5 * p

        if self.training:
            ls = self.log_scale
            ls_dev = ls.mean(dim=-1) - ls.mean()
            ls_var = ls.var().item()
            if ls_var < 0.05:
                boost = 1.0 * torch.sigmoid(3.0 * ls_dev)
                gate_logits = gate_logits + boost.unsqueeze(0).unsqueeze(0)

        expert_gate = torch.sigmoid(gate_logits)
        self._cached_gate_l1 = expert_gate.mean()
        self._cached_gate_usage = expert_gate.mean(dim=(0, 1))
        if self.training:
            self._gate_ema.data.mul_(0.99).add_(self._cached_gate_usage.detach(), alpha=0.01)
        self._cached_usefulness = usefulness
        self._cached_gate = expert_gate.detach()

        mirror = mirror * expert_gate.unsqueeze(-1)
        mirror = mirror.reshape(B, L, D)

        self._last_magnitude.fill_(mirror.abs().mean().item())
        self._last_gates.copy_(expert_gate.detach().mean(dim=(0, 1)))
        self._last_h_pool.copy_(h_g.detach().mean(dim=(0, 1)))

        return mirror, mlp_mod, mem_mod

    def cache_grad_norms(self, grad_h=None):
        if grad_h is not None:
            with torch.no_grad():
                g_norms = grad_h.reshape(-1, self.G, self.d).norm(dim=-1).mean(dim=0)
                self._prev_grad_norm.copy_(g_norms)
        else:
            self._prev_grad_norm.copy_(self._hp_grad)

    @torch.no_grad()
    def debug_mind(self):
        info = {}
        if not self._has_private_mem:
            return info
        info['private_mem_norm'] = self._private_mem.norm(dim=-1).mean().item()
        info['w_help'] = torch.sigmoid(self.w_help).mean().item()
        info['w_contra'] = self.w_contra.mean().item()
        w = torch.sigmoid(self._signal_log_weights)
        w_norm = w / (w.sum() + 1e-10)
        for i, label in enumerate(['temp','pred','smooth','sym','help'][:len(w)]):
            info[f'signal_w_{label}'] = w_norm[i].item()
        if self._cached_contra_expert is not None:
            info['contra_expert'] = self._cached_contra_expert.mean().item()
            info['contra_expert_raw'] = self._cached_contra_expert.tolist()
        if self._cached_contra_graph is not None:
            info['contra_graph_mean'] = self._cached_contra_graph.mean().item()
        if self._cached_dominance is not None:
            info['dominance'] = self._cached_dominance.tolist()
        if self._cached_isolation is not None:
            info['isolation'] = self._cached_isolation.tolist()
        if self._cached_concept_dendrogram is not None:
            info['concept_q_hi'], info['concept_q_lo'] = self._cached_concept_dendrogram
        info['gate_ema'] = self._gate_ema.tolist()
        info['gate_ema_mean'] = self._gate_ema.mean().item()
        info['gate_selectivity'] = self._last_gates.std().item()
        info['ls_var'] = self.log_scale.var(dim=-1).tolist()
        info['ls_var_mean'] = self.log_scale.var(dim=-1).mean().item()
        if self._has_private_mem:
            tm = self._trust_matrix
            info['trust_max'] = tm.max().item()
            info['trust_min'] = tm[tm > 0].min().item() if (tm > 0).any() else 0.0
            info['trust_diag'] = tm.diag().mean().item()
            info['behavior_div'] = self._behavior_div_ema.mean(dim=-1).tolist()
            info['concept_sim'] = self._concept_sim_ema.mean(dim=-1).tolist()
            bd = self._behavior_div_ema.mean(dim=-1)
            cs = self._concept_sim_ema.mean(dim=-1)
            tr = self._trust_matrix.mean(dim=-1)
            info['spec_index'] = (self._last_gates * bd).tolist()
            info['cons_index'] = (cs * tr).tolist()
        return info
