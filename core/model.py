"""
WideBind: hybrid D-space LM with VSA memory + bottleneck bind.
"""

import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import WideBindConfig
from .zeckendorf_readout import ZeckendorfReadout


# ─── Utilities ──────────────────────────────────────────────────────────

def dct_basis(n):
    """DCT-II basis vectors of shape (n, n) — orthogonal rows."""
    k = torch.arange(n, dtype=torch.float32)
    v = k.unsqueeze(1) * (k.unsqueeze(0) + 0.5)
    basis = torch.cos(v * math.pi / n)
    basis[0, :] = basis[0, :] / math.sqrt(2)
    return basis * math.sqrt(2.0 / n)


def zeckendorf_codes(vocab=50000):
    """Fibonacci Zeckendorf binary codes for vocab tokens.
    Возвращает (V, K≈23) — длина кода зависит от vocab.
    """
    fib = [1, 2]
    while fib[-1] <= vocab:
        fib.append(fib[-1] + fib[-2])
    fib = fib[:-1]
    K = len(fib)
    codes = torch.zeros(vocab, K)
    for i in range(vocab):
        n = i + 1
        for j in range(K - 1, -1, -1):
            if n >= fib[j]:
                codes[i, j] = 1.0
                n -= fib[j]
    return codes


def sparse_block_codes(vocab=50000, K=32, S=6):
    """Sparse block codes: ровно S единиц из K на каждый токен.
    
    Использует комбинаторную систему счисления (combinadic) с
    фиксированной случайной перестановкой, чтобы все K бит были
    равномерно представлены среди vocab токенов.
    
    Гарантии:
      - C(K, S) ≥ vocab     (C(32,6)=906192 ≥ 50000 ✓)
      - Ровно S=6 активных бит на каждый токен
      - Каждый бит активен у ≈ vocab·S/K токенов (≈ 9375)
      - Детерминированность (seed=42)
    """
    from math import comb
    total = comb(K, S)
    # Фиксированная случайная перестановка всех C(K, S) индексов
    perm = torch.randperm(total, generator=torch.Generator().manual_seed(42))
    codes = torch.zeros(vocab, K)
    for v in range(vocab):
        idx = int(perm[v].item())
        n = idx
        for i in range(S, 0, -1):
            c = i - 1
            while comb(c + 1, i) <= n:
                c += 1
            codes[v, c] = 1.0
            n -= comb(c, i)
    return codes


# ─── VSA Prefix Scan ───────────────────────────────────────────────────

def vsa_prefix_scan(a, b, state=None):
    """Associative parallel prefix scan for VSA memory (chunked for stability).
    mem[t] = a[t] * mem[t-1] + b[t]  (element-wise)
    
    a: (B, L, D) or (B, L) — decay factors
    b: (B, L, D) — input increments
    state: (B, D) — initial state or None
    
    Returns: (B, L, D) full scan, (B, D) final state
    """
    B, L, D = b.shape
    if a.dim() == 2:
        a = a.unsqueeze(-1).expand(-1, -1, D)
    
    eps = 1e-10
    CHUNK = 32
    out = []
    s = state.clone() if state is not None else None
    for start in range(0, L, CHUNK):
        end = min(start + CHUNK, L)
        b_chunk = b[:, start:end]
        a_chunk = a[:, start:end]
        
        log_a_chunk = torch.log(a_chunk.clamp(min=eps))
        log_cum_chunk = torch.cumsum(log_a_chunk, dim=1)
        cum_decay_chunk = torch.exp(log_cum_chunk)
        inv_cum_decay_chunk = 1.0 / cum_decay_chunk.clamp(min=eps)
        
        weighted = b_chunk * inv_cum_decay_chunk
        cum_weighted = torch.cumsum(weighted, dim=1)
        
        if s is not None:
            result_chunk = cum_decay_chunk * s.unsqueeze(1) + cum_decay_chunk * cum_weighted
        else:
            result_chunk = cum_decay_chunk * cum_weighted
        
        out.append(result_chunk)
        s = result_chunk[:, -1]
    
    result = torch.cat(out, dim=1)
    return result, result[:, -1]


# ─── Embedding ──────────────────────────────────────────────────────────

class ZeckendorfEmbedding(nn.Module):
    """Token -> D-space via Zeckendorf codes + learned projection.
    
    Legacy: проекция K→D через Linear. Ранг матрицы эмбеддингов ≤ K=23.
    """
    def __init__(self, cfg):
        super().__init__()
        codes = zeckendorf_codes(cfg.vocab)
        K = codes.shape[1]
        self.register_buffer('codes', codes)
        self.proj = nn.Linear(K, cfg.D, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, tokens):
        return self.proj(self.codes[tokens])


class PartitionedEmbedding(nn.Module):
    """Token -> D-space via partitioned sparse codes.
    
    D делится на K сегментов, K = D // seg_size (точное деление).
    Каждый бит кода получает свой сегмент: h = Σ z_k · w_k.
    
    K=32, S=6: C(32,6)=906192 ≥ V=50000. Ровно 6 активных бит на токен.
    Per-token: 6 × d = 6×112 = 672 dims (18.8%), детерминированно.
    
    Математические свойства:
      - rank(E) = 3584 (полный ранг)
      - Segment ↔ mirror group: 1:1 alignment (32×112)
      - Равномерная частота бит: ~19% каждый
      - K=32 → bind compression 32→16: ровно 2 сегмента на bind-канал
    """
    def __init__(self, cfg):
        super().__init__()
        codes = sparse_block_codes(cfg.vocab, K=cfg.code_dim, S=cfg.code_sparsity)
        self.K = codes.shape[1]
        self.register_buffer('codes', codes)
        
        D = cfg.D
        assert D % self.K == 0, f'D={D} must be divisible by K={self.K}'
        d = D // self.K
        
        # Rank expansion: mixing matrix M (K×K) с ортогональной инициализацией
        # codes → sigmoid(M·codes) даёт плотные коэффициенты, каждый бит влияет на все сегменты
        self.embed_mix = nn.Parameter(torch.zeros(self.K, self.K))
        nn.init.orthogonal_(self.embed_mix)
        self.register_buffer('_mix_scale', torch.tensor(0.1), persistent=False)
        
        self.basis = nn.Parameter(torch.randn(self.K, d))
        nn.init.normal_(self.basis, std=0.02)
    
    def forward(self, tokens):
        codes = self.codes[tokens]  # (B, L, K), sparse binary
        # Dense mixing: sigmoid(scale · M · codes) → каждый бит влияет на все сегменты
        codes = torch.sigmoid(torch.einsum('blk,kj->blj', codes, self.embed_mix) * self._mix_scale)
        B, L = tokens.shape
        return torch.einsum('blk,kd->blkd', codes, self.basis).reshape(B, L, -1)


class LmHead(nn.Module):
    """D-space -> vocab logits via Zeckendorf code projection (legacy)."""
    def __init__(self, cfg):
        super().__init__()
        codes = zeckendorf_codes(cfg.vocab)
        K = codes.shape[1]
        self.register_buffer('codes', codes)
        self.proj = nn.Linear(cfg.D, K, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, h):
        return self.proj(h) @ self.codes.T


class PartitionedHead(nn.Module):
    """D-space -> vocab logits via segment-addressed readout + per-token bias.
    
    h ∈ ℝᴰ → split по тем же K сегментам, что и в PartitionedEmbedding.
    Каждый сегмент h_k сравнивается со своим readout r_k:
        logit_v = Σ_k z_{vk} · ⟨h_k, r_k⟩ + b_v
    
    b_v — learnable per-token bias (token frequency prior).
    K=32: каждый сегмент выровнен с mirror group (1:1).
    
    Если embed_basis передан (PartitionedEmbedding.basis), readout делится с ним
    (weight tying encode/decode). Иначе — собственный readout.
    """
    def __init__(self, cfg, embed_basis=None):
        super().__init__()
        codes = sparse_block_codes(cfg.vocab, K=cfg.code_dim, S=cfg.code_sparsity)
        self.K = codes.shape[1]
        self.register_buffer('codes', codes)
        
        D = cfg.D
        assert D % self.K == 0
        d = D // self.K
        
        if embed_basis is not None:
            self.readout = embed_basis  # shared reference
        else:
            self.readout = nn.Parameter(torch.randn(self.K, d))
            nn.init.normal_(self.readout, std=0.02)
        self.token_bias = nn.Parameter(torch.zeros(cfg.vocab))
    
    def forward(self, h):
        B, L, D = h.shape
        h_g = h.reshape(B, L, self.K, -1)  # (B, L, K, d)
        scores = torch.einsum('blkd,kd->blk', h_g, self.readout)
        return scores @ self.codes.T + self.token_bias.unsqueeze(0).unsqueeze(0)


# ─── Grouped Cognitive Mirror (32 эксперта) ────────────────────────────

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
      - mirror = tanh(linear + bias) + alpha * linear
      - Обеспечивает per-dim градиент для log_scale даже при насыщении tanh
    """
    def __init__(self, D, G=32, k=32, w_pred_scale_init=3.0, log_scale_init_std=0.05,
                 delta_var_ema_min=0.8, delta_var_ema_max=0.99, tie_mirror_proj=False,
                 layer_idx=0, n_layers=32, has_private_mem=False):
        super().__init__()
        assert D % G == 0
        self.D = D
        self.G = G
        self.k = k
        self.d = D // G
        self.tie_mirror_proj = tie_mirror_proj
        # φ — единая когнитивная координата глубины (логарифмическая)
        phi = math.log(1 + layer_idx) / math.log(max(n_layers, 2))
        self.register_buffer('phi', torch.tensor(phi))
        
        proj_std = 1.0 / (self.d * k) ** 0.25
        
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
        
        # Depthwise conv per group in K-space (CAUSAL: only past tokens)
        self.conv_smooth = nn.Conv1d(G * k, G * k, 3, padding=0,
                                      groups=G * k, bias=False)
        with torch.no_grad():
            self.conv_smooth.weight.zero_()
            self.conv_smooth.weight[:, :, 1] = 1.0  # all channels get center dirac (x_{t-1})
        
        self.w_sym_u = nn.Parameter(torch.randn(G, k))
        self.w_sym_v = nn.Parameter(torch.randn(G, k))
        
        # Predictive mirror: per-dim alpha per expert
        # pred_k = alpha_kg * hp_prev, per K-dimension timescale
        # Tau hierarchy: K-dimensions span exponential range [tau_min, tau_max]
        #   τ_k = tau_min * (tau_max/tau_min)^(k/(K-1))
        #   α_k = exp(-1/τ_k)
        # Each expert inherits the same tau distribution (learnable divergence)
        tau_min, tau_max = 2.0, 200.0
        if k > 1:
            frac = torch.arange(k, dtype=torch.float32) / (k - 1)
            tau_k = tau_min * (tau_max / tau_min) ** frac
        else:
            tau_k = torch.tensor([(tau_min + tau_max) / 2])
        alpha_init = torch.exp(-1.0 / tau_k).view(1, k).expand(G, -1).clone()
        self.alpha_diag = nn.Parameter(alpha_init)
        self.w_pred_scale_legacy = nn.Parameter(torch.ones(G, k) * w_pred_scale_init)
        self.tanh_bias = nn.Parameter(torch.zeros(G, k))
        # EMA norms for signal normalization (Proposal V-1)
        n_signals = 5 if has_private_mem else 4
        self.register_buffer('_signal_norm_ema', torch.ones(n_signals, G, k), persistent=False)
        # Asymmetric init: guaranties non-zero var(log_scale) from step 0.
        # Without it, diversity loss has zero gradient at init (cold start).
        ls_base = torch.linspace(-1.0, 1.0, G).unsqueeze(1).expand(G, self.d)
        self.log_scale = nn.Parameter(ls_base + torch.randn(G, self.d) * 0.05)
        
        # ─── K-space gate (per-token, per-expert from hp) ───
        # w_gate: (G, k) — maps |pred_error| to gate logit per expert
        gate_std = 1.0 / (self.k + 1) ** 0.5
        self.w_gate = nn.Parameter(torch.randn(G, self.k) * gate_std)
        self.b_gate = nn.Parameter(torch.zeros(G))
        # w_delta_gate: (G, k) — maps delta (correction) to gate logit
        self.w_delta_gate = nn.Parameter(torch.randn(G, self.k) * 0.01)
        
        # External gradient cache (устанавливается hook'ом после backward)
        self.register_buffer('_prev_grad_norm', torch.zeros(G), persistent=False)
        # Private memory bank: expert confident K-space states (cross-expert recall)
        self._has_private_mem = has_private_mem
        self._pm_write_delay = 5000  # minimum training forward steps before writes activate
        if has_private_mem:
            self.register_buffer('_private_mem', torch.randn(G, self.k) * 0.01)
            self.register_buffer('_pm_step', torch.zeros(1, dtype=torch.long), persistent=False)
            # w_help init = log(3.0) -> sigmoid ~0.75: strong initial presence, prevents cold-start suppression
            self.w_help = nn.Parameter(torch.full((G, 1), math.log(3.0)))  # per-expert scale for recalled help
            self.w_contra = nn.Parameter(torch.full((G,), 0.01))  # small positive: disagreement opens gate by default
            # Expert knowledge graph: concept similarity, behavior divergence, trust
            self.register_buffer('_concept_sim_ema', torch.eye(G), persistent=False)   # (G, G) — who shares concepts
            self.register_buffer('_behavior_div_ema', torch.zeros(G, G), persistent=False)  # (G, G) — who behaves differently
            self.register_buffer('_trust_matrix', torch.eye(G) * 0.5, persistent=False)     # (G, G) — who helps whom
        self.register_buffer('_hp_grad', torch.zeros(G), persistent=False)
        self.register_buffer('_delta_var', torch.zeros(G), persistent=False)  # running EMA of delta var
        self.register_buffer('_last_magnitude', torch.zeros(1), persistent=False)
        self.register_buffer('_last_gates', torch.zeros(G), persistent=False)
        self.register_buffer('_last_h_pool', torch.zeros(G, self.d), persistent=False)
        # Alpha override: set to 0.5 during warmup to force large pred_error
        # 0.0 = use learned alpha; >0 = override alpha for all experts
        self.register_buffer('_alpha_override', torch.zeros(1), persistent=False)
        # Cache for alpha auxiliary loss
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
        # Residual variance EMA for adaptive tau (self-organizing timescales)
        self.register_buffer('_residual_var_ema', torch.ones(G, k) * 0.1, persistent=False)
        
        # ─── Per-expert learned modulation (геометрическая init по φ) ───
        # skip_alpha: L0≈17, L31≈0.10 (из чекпоинта step 50000)
        # ρ=0.6: сенсорный слой L0≈17, глубокий L31≈0.10
        rho = 0.6 ** layer_idx
        log_skip_init = math.log(0.10) + (math.log(17.0) - math.log(0.10)) * rho
        self.log_skip_alpha = nn.Parameter(torch.full((G,), log_skip_init))
        # mod_scale: L0≈-0.81, L31≈-2.30 (из чекпоинта)
        log_mod_init = -2.30 + (-0.81 - (-2.30)) * rho
        self.log_dvar_mod_scale = nn.Parameter(torch.full((G,), log_mod_init))
        self.log_grad_mod_scale = nn.Parameter(torch.full((G,), log_mod_init))
        self.dvar_mod_bias = nn.Parameter(torch.full((G,), -0.01))
        self.grad_mod_bias = nn.Parameter(torch.full((G,), -0.01))
        self._delta_var_ema_min = delta_var_ema_min
        self._delta_var_ema_max = delta_var_ema_max

        # ─── Learnable signal weights (softmax-normalized) ───
        self._signal_log_weights = nn.Parameter(torch.zeros(n_signals))
        
        # ─── Self-organizing usefulness predictor (competitive) ───
        # Каждый эксперт предсказывает свою полезность по delta (K-space correction)
        # Softmax по G: эксперты конкурируют за право модулировать слой.
        # Только лучшие эксперты для данного токена получают высокий вес.
        # init: без Sigmoid — raw logits для softmax-конкуренции
        self.usefulness_predictor = nn.Sequential(
            nn.Linear(k, k),
            nn.Tanh(),
            nn.Linear(k, 1),
        )
        # Per-expert масштабы модуляции (learned log-scale)
        self.mod_scale_mlp = nn.Parameter(torch.full((G,), math.log(2.0)))
        self.mod_scale_mem = nn.Parameter(torch.full((G,), math.log(2.0)))
        # Softmax temperature: >1 = softer (uniform), <1 = sharper (winner-take-all)
        self.register_buffer('_usefulness_temp', torch.tensor(2.0), persistent=False)
        # Error-gated damping: порог резонансного демпфирования α на инференсе
        self.register_buffer('_damp_tau', torch.tensor(0.1), persistent=False)
    
    def _sync_W_out(self):
        with torch.no_grad():
            self.W_out.copy_(self.W_proj.permute(0, 2, 1))
    
    def forward(self, h, mem_all, global_state=None, diff=None,
                tanh_bias_mod=1.0, pred_scale_mod=None, context_mem=None,
                allow_write=None):
        B, L, D = h.shape
        G, d, k = self.G, self.d, self.k
        
        # Split into subspaces
        h_g = h.reshape(B, L, G, d)           # (B, L, G, d)
        mem_g = mem_all.reshape(B, L, G, d)
        mc_g = mem_g.mean(dim=1, keepdim=True)  # (B, 1, G, d)
        
        # Project each group to its K-space
        hp = torch.einsum('blgd,gdk->blgk', h_g, self.W_proj)    # (B, L, G, k)
        # Hook to capture gradient for grad_mod (only during training)
        if hp.requires_grad:
            hp.register_hook(lambda g: (
                self._prev_grad_norm.copy_(g.detach().norm(dim=-1).mean(dim=(0, 1))),
                None
            )[1])
        mc_k = torch.einsum('b l gd,gdk->b l gk', mc_g, self.W_proj)
        
        # hp_prev shared by sym_k and pred_error
        hp_prev = torch.cat([torch.zeros_like(hp[:, 0:1]), hp[:, :-1]], dim=1)
        
        # ─── Slow signals (lo half of K-space) ───
        # Temporal: deviation from memory centroid
        temp_k = (hp - mc_k) * self.w_temp  # (B, L, G, k)
        
        # Global: deviation from cross-layer state
        if global_state is not None:
            gs_k = torch.einsum('b l gd,gdk->b l gk',
                                global_state.reshape(1, 1, G, d), self.W_proj)
            temp_k = temp_k + (hp - gs_k) * self.w_global
        
        # Predictive: error in K-space self-prediction (t-1 -> t)
        # Per-dim alpha: each K-dimension has its own timescale
        # Alpha override smoothly interpolates: override=1 → identity (α=1),
        # override=0 → learned alpha_diag. Provides smooth warmup transition.
        alpha_eff = self.alpha_diag
        override = self._alpha_override.item()
        if override > 0:
            alpha_eff = (1 - override) * alpha_eff + override * 1.0
        pred_k = hp_prev * alpha_eff.view(1, 1, G, k)  # (B, L, G, k)
        _pred_k_aux = pred_k  # undamped — для aux loss, чтобы damping не боролся с ним
        if pred_scale_mod is None:
            dv = self._delta_var
            dv_mean = dv.mean().clamp(min=1e-8)
            pred_scale_mod = (dv / dv_mean).clamp(0.1, 3.0)
        # Нормализованная ошибка предсказания: relative к ||hp||, а не абсолютная
        hp_norm = hp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        raw_pred_error = hp - pred_k
        # Error-gated damping (только инференс): α → 1 когда ||pred_error|| велика
        # На тренировке (teacher forcing) δ_{t-1}=0, резонанс безвреден.
        if not self.training:
            damp = torch.sigmoid(-raw_pred_error.norm(dim=-1).mean() / self._damp_tau)
            alpha_eff = 1.0 + (alpha_eff - 1.0) * damp
            pred_k = hp_prev * alpha_eff.view(1, 1, G, k)
        # Без w_pred_scale — сигнал нормируется EMA вместе с остальными
        pred_error = (hp - pred_k) / hp_norm * pred_scale_mod.view(G, 1)
        # Adaptive tau: K-измерения с высокой ошибкой → короткое τ, с низкой → длинное.
        # alpha_target = sigmoid(2.2 - log(rel_var)):
        #   rel_var=1 (noise) → α=0.9 (init), rel_var=0.5 → α=0.95, rel_var=2 → α=0.82
        with torch.no_grad():
            override = self._alpha_override.item()
            if override < 0.1:
                residual_var = pred_error.var(dim=(0, 1), unbiased=False)
                self._residual_var_ema.lerp_(residual_var, 0.01)
                rv = self._residual_var_ema
                rv_mean = rv.mean(dim=-1, keepdim=True)
                relative_var = rv / (rv_mean + 1e-10)
                alpha_target = torch.sigmoid(2.2 - torch.log(relative_var))
                self.alpha_diag.data.lerp_(alpha_target, 0.0005)
        self._cached_pred_k = _pred_k_aux
        self._cached_hp = hp
        # Cache normalized pred_error norm per token для surprisal-gated i_gate
        pred_error_norm = (raw_pred_error / hp_norm).norm(dim=(-2, -1))  # (B, L)
        self._cached_pred_error_norm = pred_error_norm
        
        # ─── Private Memory: read via cross-expert attention (when uncertain) ───
        if self._has_private_mem:
            uncert = torch.sigmoid(pred_error.abs())  # (B, L, G, k)
            q = hp * uncert
            keys = self._private_mem.detach().clone()  # (G, k) — frozen snapshot for autograd
            if context_mem is not None:
                # Blend context override with learned memory (30% context, 70% learned)
                keys = context_mem * 0.3 + keys * 0.7
                keys = F.normalize(keys, dim=-1) * self._private_mem.norm(dim=-1, keepdim=True)
            attn = F.softmax(q @ keys.T / math.sqrt(self.k), dim=-1)  # (B, L, G, G)
            help_k_base = attn @ keys  # (B, L, G, k) — collective confident memory
            # ─── Contradiction gate: disagreement between expert hp and collective help_k ───
            hp_n = hp.norm(dim=-1).clamp(min=1e-8)  # (B, L, G)
            disagreement = (hp - help_k_base).norm(dim=-1) / hp_n  # relative: 0=agrees, >>1=contradicts
            contra = torch.sigmoid(disagreement - 1.0)  # sigmoid(rel_disagree - 1): <1=agrees, >1=contradicts
            trust = 1.0 - contra  # how much to trust help_k (low when contradictory)
            # Apply contradiction-aware scaling: 
            # high disagreement + confident expert → expert is confidently wrong → reduce help_k
            # high disagreement + uncertain expert → collective irrelevant → reduce help_k
            help_k = help_k_base * torch.sigmoid(self.w_help).unsqueeze(0).unsqueeze(0)
            help_k = help_k * trust.unsqueeze(-1)  # trust-weighted collective memory
            self._cached_contra = contra.detach()  # for analysis
            self._cached_disagreement = disagreement.detach()  # for analysis
        else:
            help_k = torch.zeros_like(hp)
            trust = torch.ones_like(hp.norm(dim=-1))  # no contradiction when disabled
        
        # ─── Expert Knowledge Graph update (uses OLD private_mem, runs before write) ───
        if self._has_private_mem and self.training:
            with torch.no_grad():
                pm = self._private_mem
                pm_norm = pm.norm(dim=-1, keepdim=True).clamp(min=1e-10)
                pm_n = pm / pm_norm  # safe normalize (no NaN on zero vectors)
                concept_sim = pm_n @ pm_n.T
                self._concept_sim_ema.mul_(0.999).add_(concept_sim, alpha=0.001)
                hp_avg = hp.mean(dim=(0, 1))
                hp_n = F.normalize(hp_avg, dim=-1)
                behavior_sim = hp_n @ hp_n.T
                behavior_div = 1.0 - behavior_sim
                self._behavior_div_ema.mul_(0.999).add_(behavior_div, alpha=0.001)
                trust_weights = attn.mean(dim=(0, 1))
                self._trust_matrix.mul_(0.999).add_(trust_weights, alpha=0.001)
                contra_g = concept_sim * behavior_div
                self._cached_contra_graph = contra_g
                contra_expert = contra_g.mean(dim=-1)
                self._cached_contra_expert = contra_expert
                sim_vals = concept_sim[~torch.eye(self.G, dtype=torch.bool, device=concept_sim.device)]
                q_hi = sim_vals.quantile(0.75)
                q_lo = sim_vals.quantile(0.25)
                self._cached_concept_dendrogram = (q_hi.item(), q_lo.item())
                dominance = self._trust_matrix.sum(dim=0)
                isolation = 1.0 - (self._trust_matrix.sum(dim=-1) / self.G)
                self._cached_dominance = dominance
                self._cached_isolation = isolation
        
        # ─── Private Memory: write confident K-space states (contradiction-aware) ───
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
                # Soft competition: temperature prevents winner-take-all monoculture
                temp_write = 0.5  # <1 softens competition (true soft), >1 sharpens
                conf_soft = conf_plastic ** temp_write
                conf_bc = conf_soft * self.G / (conf_soft.sum(dim=-2, keepdim=True) + 1e-8)
                weighted_hp = (conf_bc * hp.detach()).mean(dim=(0, 1))
                # Adaptive decay: fast warmup when memory is nascent, slow when stable
                pm_scale = self._private_mem.norm(dim=-1).mean().clamp(min=1e-8)
                warmup_rate = torch.sigmoid(3.0 - pm_scale)  # ~1.0 when pm~0, ~0.0 when pm>3
                pm_decay = 0.999 - 0.009 * warmup_rate  # [0.990, 0.999] — faster decay when memory is empty
                self._private_mem.mul_(pm_decay).add_(weighted_hp, alpha=1.0 - pm_decay)
                self._private_mem.clamp_(-10.0, 10.0)
        
        # ─── Fast signals (hi half of K-space) ───
        # Smoothness: local coherence in K-space (CAUSAL: pad left only)
        hp_perm = hp.permute(0, 2, 3, 1).reshape(B, G * k, L)  # (B, G*k, L)
        hp_pad = F.pad(hp_perm, (2, 0))  # pad 2 zeros on left, 0 on right
        hp_smooth = self.conv_smooth(hp_pad)  # (B, G*k, L) — kernel sees t-2, t-1, t
        hp_smooth = hp_smooth.reshape(B, G, k, L).permute(0, 3, 1, 2)  # (B, L, G, k)
        smooth_k = hp - hp_smooth
        
        # Symmetry: bilinear temporal interaction
        sym_k = (hp * self.w_sym_u) * (hp_prev * self.w_sym_v)
        
        # ─── EMA-нормировка сигналов (соизмеримость перед softmax) ───
        if self._has_private_mem:
            signals = [temp_k, pred_error, smooth_k, sym_k, help_k]
        else:
            signals = [temp_k, pred_error, smooth_k, sym_k]
        signals_normed = []
        decay = 0.001  # ~1000-step EMA
        for i, s in enumerate(signals):
            with torch.no_grad():
                rms = s.norm(dim=(-2, -1), keepdim=True).mean(dim=(0, 1), keepdim=True)  # (1, 1, G, k)
                self._signal_norm_ema[i].mul_(1 - decay).add_(rms.squeeze(), alpha=decay)
            s_norm = s / (self._signal_norm_ema[i].unsqueeze(0).unsqueeze(0) + 1e-8)
            signals_normed.append(s_norm)
        
        # ─── Learnable signal weights (softmax-normalized) ───
        n_sig = len(signals)
        w = torch.softmax(self._signal_log_weights, dim=0)  # {n_sig} weights summing to 1
        
        # ─── Merge all signals (weighted sum) ───
        delta = sum(w[i] * signals_normed[i] for i in range(n_sig))
        
        delta = F.rms_norm(delta, (delta.shape[-1],))  # norm over k
        delta = delta + self.tanh_bias * tanh_bias_mod
        
        # ─── Gate modulation signals (shared between gate & usefulness) ───
        grad_mod = torch.exp(self.log_grad_mod_scale) * torch.tanh(self._prev_grad_norm + self.grad_mod_bias)
        with torch.no_grad():
            dvar = delta.var(dim=(0, 1), unbiased=False).mean(dim=-1)  # (G,)
            if diff is not None:
                ema_alpha = self._delta_var_ema_min + diff * (self._delta_var_ema_max - self._delta_var_ema_min)
            else:
                ema_alpha = 0.9
            self._delta_var.mul_(ema_alpha).add_(dvar * (1.0 - ema_alpha))
        dvar_mod = torch.exp(self.log_dvar_mod_scale) * torch.tanh(self._delta_var + self.dvar_mod_bias)
        
        # ─── Self-organizing usefulness (sigmoid + adaptive threshold) ───
        # Каждый эксперт предсказывает свою полезность по delta (K-space correction).
        # Sigmoid + median threshold: конкуренция без zero-sum (sum≠1).
        # Эксперты выше медианы получают >0.5, ниже — <0.5.
        usefulness_logits = self.usefulness_predictor(delta).squeeze(-1)  # (B, L, G)
        temp = self._usefulness_temp.clamp(min=0.1)
        with torch.no_grad():
            threshold = usefulness_logits.median(dim=-1, keepdim=True).values  # (B, L, 1)
        usefulness = torch.sigmoid((usefulness_logits - threshold) / temp)
        # Homeostatic temperature (after warmup): бинарная энтропия управляет остротой
        with torch.no_grad():
            override = self._alpha_override.item()
            if override < 0.1:
                u_ent = -(usefulness * torch.log(usefulness + 1e-10) +
                          (1 - usefulness) * torch.log(1 - usefulness + 1e-10))
                u_ent_mean = u_ent.sum(dim=-1).mean()
                target_ent = 0.75 * G * 0.693  # ~0.75*G*log(2)
                temp_err = u_ent_mean - target_ent
                self._usefulness_temp.data.add_(-0.001 * temp_err * self._usefulness_temp.data)
                self._usefulness_temp.data.clamp_(min=0.3, max=4.0)
        
        # Per-expert modulation strengths (gated by self-assessment)
        mlp_mod = usefulness * torch.sigmoid(self.mod_scale_mlp).view(1, 1, G)  # (B, L, G)
        mem_mod = usefulness * torch.sigmoid(self.mod_scale_mem).view(1, 1, G)
        
        # Linear projection + skip connection
        linear = torch.einsum('blgk,gkd->blgd', delta, self.W_out)  # (B, L, G, d)
        skip_alpha = torch.exp(self.log_skip_alpha).view(1, 1, G, 1)
        mirror = torch.tanh(linear) + skip_alpha * linear
        mirror = mirror * torch.exp(self.log_scale)  # per-dim scale
        
        # ─── K-Space Gate (per-token, per-expert) ───
        gate_signal = torch.abs(pred_error)  # (B, L, G, k)
        gate_logits = torch.einsum('blgk,gk->blg', gate_signal, self.w_gate) + self.b_gate
        # Delta signal: how much correction is mirror applying (complements |pred_err|)
        delta_gate = torch.einsum('blgk,gk->blg', delta, self.w_delta_gate)
        gate_logits = gate_logits + delta_gate
        gate_logits = gate_logits + grad_mod.unsqueeze(0).unsqueeze(0)
        gate_logits = gate_logits + dvar_mod.unsqueeze(0).unsqueeze(0)
        # Contradiction signal: expert vs collective disagreement opens gate (arbiter)
        if self._has_private_mem:
            gate_logits = gate_logits + disagreement * self.w_contra.unsqueeze(0).unsqueeze(0)
            # Concept graph pressure: high contradiction → open gate more
            if self._cached_contra_expert is not None:
                ce = self._cached_contra_expert.to(gate_logits.device).unsqueeze(0).unsqueeze(0)
                gate_logits = gate_logits + ce  # collective contradiction raises gate
        
        expert_gate = torch.sigmoid(gate_logits)  # (B, L, G)
        # Cache gate L1 for auxiliary sparsity loss (still in graph for gradients)
        self._cached_gate_l1 = expert_gate.mean()
        # Cache per-expert mean gate for load balancing loss
        self._cached_gate_usage = expert_gate.mean(dim=(0, 1))  # (G,)
        # Cache for expert reinforcement loss (gate vs usefulness alignment)
        self._cached_usefulness = usefulness
        self._cached_gate = expert_gate.detach()
        
        mirror = mirror * expert_gate.unsqueeze(-1)
        mirror = mirror.reshape(B, L, D)
        
        self._last_magnitude.fill_(mirror.abs().mean().item())
        self._last_gates.copy_(expert_gate.detach().mean(dim=(0, 1)))
        self._last_h_pool.copy_(h_g.detach().mean(dim=(0, 1)))
        
        return mirror, mlp_mod, mem_mod
    
    def cache_grad_norms(self, grad_h=None):
        """Call after backward: store per-subspace gradient norm.
        Uses hp hook by default; falls back to explicit grad_h if provided."""
        if grad_h is not None:
            with torch.no_grad():
                g_norms = grad_h.reshape(-1, self.G, self.d).norm(dim=-1).mean(dim=0)
                self._prev_grad_norm.copy_(g_norms)
        else:
            self._prev_grad_norm.copy_(self._hp_grad)

    @torch.no_grad()
    def debug_mind(self):
        """Return a dict of meta-cognitive stats for generation interpretability.
        Works in eval mode — computes KG stats on-demand if not cached."""
        info = {}
        if not self._has_private_mem:
            return info
        info['private_mem_norm'] = self._private_mem.norm(dim=-1).mean().item()
        info['w_help'] = torch.sigmoid(self.w_help).mean().item()
        info['w_contra'] = self.w_contra.mean().item()
        w = torch.softmax(self._signal_log_weights, dim=0)
        for i, label in enumerate(['temp','pred','smooth','sym','help'][:len(w)]):
            info[f'signal_w_{label}'] = w[i].item()
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
        tm = self._trust_matrix
        info['trust_max'] = tm.max().item()
        info['trust_min'] = tm[tm > 0].min().item() if (tm > 0).any() else 0.0
        info['trust_diag'] = tm.diag().mean().item()
        return info


def migrate_bind_state_dict(sd, n_layers, mode="off", S=1):
    """Convert old (pre-BottleneckBind) state dict keys to new format.
    Old: layers.N.W_proj (D,K)  layers.N.W_out (K,D)  layers.N.w_u (K,)  layers.N.w_v (K,)
    New: layers.N.bind.W_proj.weight (K,D)  layers.N.bind.W_out (K,D|S,K,D)  layers.N.bind.w_u (S,K)  layers.N.bind.w_v (S,K)
    """
    import re
    map_sd = {}
    for key, val in sd.items():
        m = re.match(r'layers\.(\d+)\.(W_proj|W_out|w_u|w_v|w_bind_bias)$', key)
        if not m:
            map_sd[key] = val
            continue
        lidx, param = m.groups()
        new_key = f'layers.{lidx}.bind.{param}'
        if param == 'W_proj':
            new_key = f'layers.{lidx}.bind.{param}.weight'
            map_sd[new_key] = val.t().contiguous()
        elif param == 'w_u' or param == 'w_v':
            map_sd[new_key] = val.unsqueeze(0)
        else:
            map_sd[new_key] = val
    return map_sd


def _golden_shifts(K: int, S: int) -> list:
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    shifts, used = [], set()
    s = 1
    while len(shifts) < S:
        sh = int(math.floor(s * K / phi)) % K
        while sh in used or sh == 0:
            sh = (sh + 1) % K
        shifts.append(sh); used.add(sh); s += 1
    return shifts


class BottleneckBind(nn.Module):
    def __init__(self, D: int, K: int, cfg):
        super().__init__()
        self.D, self.K = D, K
        self.mode = getattr(cfg, "bind_twist_mode", "off")
        self.S = int(getattr(cfg, "bind_twist_S", 4))
        if self.mode == "off":
            self.S = 1
        self.ocular = getattr(cfg, "bind_twist_ocular", "tied")
        self.gated = bool(getattr(cfg, "bind_twist_gate", False)) and self.mode != "off"
        scheme = getattr(cfg, "bind_twist_scheme", "golden")
        tie_bind = bool(getattr(cfg, "tie_bind", True))

        self.w_bind_bias = nn.Parameter(torch.zeros(K))

        self.W_proj = nn.Linear(D, K, bias=False)

        shifts = _golden_shifts(K, self.S) if scheme == "golden" else _fibonacci_shifts(K, self.S)
        self.register_buffer("shifts", torch.tensor(shifts, dtype=torch.long), persistent=False)

        self.w_u = nn.Parameter(torch.empty(self.S, K))
        self.w_v = nn.Parameter(torch.empty(self.S, K))
        nn.init.normal_(self.w_u, 0.0, 1.0)
        nn.init.normal_(self.w_v, 0.0, 1.0)

        # For shift mode with tie_bind, separate W_out per shift is needed for full rank S*K
        if self.mode == "shift" and tie_bind and self.S > 1:
            self.ocular = "multi"

        if self.mode != "off" and self.ocular == "multi" and self.S > 1:
            self.W_out = nn.Parameter(torch.empty(self.S, K, D))
            nn.init.normal_(self.W_out, 0.0, 0.02)
            self._tied = False
        else:
            self.W_out = nn.Parameter(torch.empty(K, D))
            nn.init.normal_(self.W_out, 0.0, 0.02)
            self._tied = tie_bind
            if self._tied:
                self.W_proj.register_forward_pre_hook(self._tie_hook)

        if self.gated:
            self.w_gate_proj = nn.Linear(K, self.S, bias=True)
            nn.init.normal_(self.w_gate_proj.weight, 0.0, 0.02)
            nn.init.zeros_(self.w_gate_proj.bias)

        if self.mode == "cascade":
            self.mix_logit = nn.Parameter(torch.zeros(self.S))

    def _tie_hook(self, module, inp):
        with torch.no_grad():
            self.W_out.data.copy_(self.W_proj.weight.data)

    def _cross(self, left, right, shift):
        return left * torch.roll(right, shifts=int(shift), dims=-1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        hp = self.W_proj(h) + self.w_bind_bias

        if self.gated:
            g = torch.sigmoid(self.w_gate_proj(hp)).unsqueeze(-1)
        else:
            g = None

        if self.mode == "off":
            prod = (hp * self.w_u[0]) * (hp * self.w_v[0])
            return prod @ self.W_out

        if self.mode == "shift":
            if not self._tied and self.ocular == "multi":
                out = None
                for s in range(self.S):
                    prod = self._cross(hp * self.w_u[s], hp * self.w_v[s], self.shifts[s])
                    if g is not None:
                        prod = prod * g[:, :, s]
                    term = prod @ self.W_out[s]
                    out = term if out is None else out + term
                return out
            else:
                acc = None
                for s in range(self.S):
                    prod = self._cross(hp * self.w_u[s], hp * self.w_v[s], self.shifts[s])
                    if g is not None:
                        prod = prod * g[:, :, s]
                    acc = prod if acc is None else acc + prod
                return acc @ self.W_out

        if self.mode == "cascade":
            a = [None] * (self.S + 1)
            a[1] = hp * self.w_u[0]
            a[2] = hp * self.w_v[0] if self.S >= 2 else a[1]
            seed_norm = a[1].norm(dim=-1, keepdim=True).detach()
            for n in range(3, self.S + 1):
                crossed = self._cross(a[n-1] * self.w_u[n-1], a[n-2] * self.w_v[n-1], self.shifts[n-1])
                a[n] = F.normalize(crossed, dim=-1) * seed_norm

            mix = torch.softmax(self.mix_logit, dim=0)
            if not self._tied and self.ocular == "multi":
                out = None
                for n in range(1, self.S + 1):
                    w = mix[n-1]
                    if g is not None:
                        w = w * g[:, :, n-1]
                    term = a[n] * w.unsqueeze(-1) @ self.W_out[n-1]
                    out = term if out is None else out + term
                return out
            else:
                m = None
                for n in range(1, self.S + 1):
                    w = mix[n-1]
                    if g is not None:
                        w = w * g[:, :, n-1]
                    term = a[n] * w.unsqueeze(-1)
                    m = term if m is None else m + term
                return m @ self.W_out


def _fibonacci_shifts(K: int, S: int) -> list:
    shifts, used, a, b = [], set(), 1, 1
    guard = 0
    while len(shifts) < S and guard < 10 * S:
        sh = b % K
        if sh not in used and sh != 0:
            shifts.append(sh); used.add(sh)
        a, b = b, a + b; guard += 1
    if len(shifts) < S:
        for g in _golden_shifts(K, S):
            if g not in used:
                shifts.append(g); used.add(g)
            if len(shifts) == S:
                break
    return shifts


# ─── Grouped MLP ──────────────────────────────────────────────────────

class GroupedMLP(nn.Module):
    """
    Grouped bottleneck MLP with per-group expansion.

    Instead of D → D → D (rank-bounded by D), splits D into G groups
    and gives each group internal expansion (d → expand*d → d).
    Total rank still ≤ D, but each group learns richer features
    within its d-dim subspace.

    G=8, d=112, expand=4 → 4× per-group expansion at half the params
    of a full 896→896→896 MLP.
    """
    def __init__(self, D, expand, groups):
        super().__init__()
        assert D % groups == 0
        self.D = D
        self.G = groups
        self.d = D // groups
        d = self.d
        e = expand

        up_std = (2.0 / (d + e * d)) ** 0.5
        down_std = (2.0 / (e * d + d)) ** 0.5
        self.W_up = nn.Parameter(torch.randn(groups, d, e * d) * up_std)
        self.W_down = nn.Parameter(torch.randn(groups, e * d, d) * down_std)
        self.norm_w = nn.Parameter(torch.ones(D))

    def forward(self, h):
        B, L, D = h.shape
        h = F.rms_norm(h, (D,), self.norm_w)
        h = h.reshape(B, L, self.G, self.d)
        h = F.silu(torch.einsum('blgd,gdf->blgf', h, self.W_up))
        h = torch.einsum('blgf,gfd->blgd', h, self.W_down)
        # Cache per-group outputs for diversity loss (до усреднения)
        self._cached_group_out = h  # (B, L, G, d)
        return h.reshape(B, L, D)


# ─── WideBind Block ────────────────────────────────────────────────────

class WideBindBlock(nn.Module):
    """
    Hybrid block: D -> K (bottleneck bind) + VSA memory + Conv + Spectral + MLP.
    
    Key design decisions:
    - Pre-LN: RMS norm at block start
    - Bind: D->K projection, bilinear in K, K->D projection
    - Memory: VSA vector superposition (not covariance matrix)
    - Gates: per-dim element-wise
    - Conv: depthwise 48-tap
    - Spectral: DCT basis scaling
    - MLP: D -> bottleneck -> D with residual
    """
    
    def __init__(self, cfg: WideBindConfig, layer_idx: int):
        super().__init__()
        self.D = cfg.D
        self.K = cfg.bind_K
        self.layer_idx = layer_idx
        self.tie_bind = cfg.tie_bind
        
        # Pre-LN weight
        self.register_buffer('pre_ln_w', torch.ones(cfg.D))
        self.total_layers = cfg.n_layers
        
        # ─── Bind: D -> K -> BottleneckBind (twisted bilinear) ───
        self.bind = BottleneckBind(cfg.D, cfg.bind_K, cfg)

        # Cognitive Mirror (32 эксперта, grouped K-space)
        if getattr(cfg, 'mirror_k_staircase', False):
            # Иерархия k_l: 4/8/16 по третям глубины, d/k_l ∈ {32,16,8}
            n = cfg.n_layers
            l = layer_idx
            if l < n // 3:
                k = 4
            elif l < (2 * n) // 3:
                k = 8
            else:
                k = 16
        else:
            k = cfg.mirror_k
        self.mirror = GroupedCognitiveMirror(cfg.D, G=cfg.mlp_groups, k=k,
            w_pred_scale_init=cfg.w_pred_scale_init, log_scale_init_std=cfg.log_scale_init_std,
            delta_var_ema_min=cfg.delta_var_ema_min, delta_var_ema_max=cfg.delta_var_ema_max,
            tie_mirror_proj=cfg.tie_mirror_proj,
            layer_idx=layer_idx, n_layers=cfg.n_layers,
            has_private_mem=cfg.private_mem)
        
        # ─── VSA Memory (multi-scale VSA: S=4 фиксированных τ) ───
        self._n_scales = 4
        tau_s = torch.tensor([8, 32, 128, 512], dtype=torch.float32)
        self.register_buffer('_tau_s', tau_s)
        self.w_i = nn.Parameter(torch.randn(cfg.D))          # content-dependent write gate (shared across scales)
        self.w_d = nn.Parameter(torch.randn(cfg.D) * cfg.w_d_init_std)    # content-dependent decay modulation
        self.w_q = nn.Parameter(torch.full((cfg.D,), 1.0 / math.sqrt(cfg.D)))  # warm read: mem_read ≈ mem_all at init
        self.w_q_leaf = nn.Parameter(torch.full((cfg.D,), 1.0 / math.sqrt(cfg.D)))  # leaf-level within-chunk read
        self.w_q_ctx = nn.Parameter(torch.full((cfg.D,), 0.5 / math.sqrt(cfg.D)))  # cross-chunk context read
        self.w_mem2v = nn.Parameter(torch.randn(cfg.D))
        # Per-scale per-channel combination weights (logits for softmax)
        self.scale_w = nn.Parameter(torch.zeros(self._n_scales, cfg.D))
        # Linear decay across layers: shallow → short memory, deep → long
        # Per-channel (D,) — can differentiate via gradient when vsa_b_d_smooth < 1.0
        layer_frac = layer_idx / max(cfg.n_layers - 1, 1)
        b_d_init = 2.0 + 3.0 * layer_frac  # L0: τ≈7, L23: τ≈63, L31: τ≈150
        self.b_i = nn.Parameter(torch.full((cfg.D,), -2.5))   # i_gate ~0.08 init
        self.b_d = nn.Parameter(torch.full((cfg.D,), b_d_init))
        # Surprisal-gated write coefficient γ_l: растёт с τ
        # γ_l = γ_max · σ((ln τ_l - ln 32) / 1.0)
        tau_l = math.exp(b_d_init)
        gamma_max = 0.5
        gamma_init = gamma_max * (1.0 / (1.0 + math.exp(-(math.log(tau_l) - math.log(32.0)))))
        self.gamma_surprisal = nn.Parameter(torch.full((), gamma_init))

        # First moment
        self.w_k_mu = nn.Parameter(torch.randn(cfg.D))
        self.w_q_mu = nn.Parameter(torch.randn(cfg.D))
        self.w_mu_mem = nn.Parameter(torch.randn(cfg.D))
        
        # ─── Conv ───
        self.conv = nn.Conv1d(cfg.D, cfg.D, kernel_size=cfg.conv_kernel,
                              padding=cfg.conv_kernel - 1, groups=cfg.D, bias=False)
        nn.init.normal_(self.conv.weight, std=cfg.conv_init_std)
        
        # ─── Spectral (self-organizing frequency filters) ───
        self.register_buffer('V_dct', dct_basis(cfg.D))
        base = 0.5 + layer_idx / max(cfg.n_layers - 1, 1)
        # Per-dim variation: low frequencies get slight boost, high get slight cut
        # Creates natural 1/f-like distribution encouraging frequency band separation
        freq_scale = torch.linspace(1.0, 0.5, cfg.D)  # DC amp=1, Nyquist=0.5
        per_dim = freq_scale * 0.2  # 20% variation across freq spectrum
        lam = torch.full((cfg.D,), base) + per_dim
        self.lambda_k = nn.Parameter(lam)
        
        # ─── MLP (grouped: per-group 4× expansion, half params) ───
        self.mlp = GroupedMLP(cfg.D, expand=cfg.mlp_expand, groups=cfg.mlp_groups)
    
    def forward(self, h, state=None, global_state=None,
                mem2v_scale=1.0, diff=None, noise_scale=0.0,
                tanh_bias_mod=1.0, pred_scale_mod=None, spectral_mod=1.0,
                context_mem=None, allow_write=None):
        mem_state = mu_state = conv_state = None
        if state is not None:
            mem_state, mu_state, conv_state = state
        B, L, D = h.shape
        # Clear stale mirror cache
        self.mirror._cached_pred_error_norm = None
        K = self.K
        device = h.device
        
        # ─── Pre-LN ───
        h = F.rms_norm(h, (D,), self.pre_ln_w)
        
        # ─── Conv ───
        if conv_state is None:
            conv_state = torch.zeros(B, D, self.conv.padding[0], device=device, dtype=h.dtype)
        h_perm = h.transpose(1, 2)
        h_conv = self.conv(torch.cat([conv_state, h_perm], dim=-1))
        h_conv = h_conv[..., :L].transpose(1, 2)
        conv_state_out = h_perm[:, :, -(self.conv.padding[0]):]
        h = h + h_conv
        self._cache_conv_out = h_conv.detach()
        
        # ─── Bind: BottleneckBind ───
        bind_out = self.bind(h)
        
        # ─── VSA Memory (multi-scale: S=4 фиксированных τ) ───
        S = self._n_scales
        d_s = torch.exp(-1.0 / self._tau_s.to(device))  # (S,) — fixed τ-scales
        # Surprisal-gated write: i_gate = softplus(linear + γ·||ê||₂)
        igate_logit = h * self.w_i + self.b_i
        mir = self.mirror
        if hasattr(mir, '_cached_pred_error_norm') and mir._cached_pred_error_norm is not None:
            pen = mir._cached_pred_error_norm  # (B, L)
            igate_logit = igate_logit + self.gamma_surprisal * pen.unsqueeze(-1)
        i_gate = F.softplus(igate_logit)                    # (B, L, D)
        d_mod = torch.sigmoid(h * self.w_d + self.b_d)      # (B, L, D) — content mod of decay
        if noise_scale > 0 and self.training:
            noise = 1.0 + noise_scale * torch.randn_like(i_gate)
            i_gate = i_gate * noise
        
        # Vectorize over S scales: (B, L, D) → (B, L, S*D)
        d_s_vec = d_s.view(1, 1, S, 1).expand(B, L, S, D).reshape(B, L, S * D)
        d_mod_vec = d_mod.unsqueeze(2).expand(-1, -1, S, -1).reshape(B, L, S * D)
        decay = d_s_vec * d_mod_vec  # each scale: d_s · content_mod
        
        mem_input = h * i_gate  # (B, L, D)
        input_vec = mem_input.unsqueeze(2).expand(-1, -1, S, -1).reshape(B, L, S * D)
        
        eps = 1e-10
        CHUNK = 32
        
        # fp32 guard for log-space scan (critical under AMP for long memory)
        _dtype = decay.dtype
        decay_f32 = decay.float()
        input_vec_f32 = input_vec.float()
        if mem_state is not None:
            mem_state_f32 = mem_state.reshape(B, S * D).float()
        else:
            mem_state_f32 = None
        
        def _scan_chunk(b_chunk, d_chunk):
            """Parallel chunk scan from zero state.
            Returns intra-chunk VSA (B, chunk_len, S*D), final state (B, 1, S*D),
            cumulative decay (B, chunk_len, S*D).
            """
            log_a = torch.log(d_chunk.clamp(min=eps))
            log_cum = torch.cumsum(log_a, dim=1)
            cum_decay = torch.exp(log_cum)
            inv_cum = 1.0 / cum_decay.clamp(min=eps)
            weighted = b_chunk * inv_cum
            cum_w = torch.cumsum(weighted, dim=1)
            intra = cum_decay * cum_w
            final = intra[:, -1:]
            return intra, final, cum_decay
        
        def _combine_chunks(chunk_data, initial_state):
            """2nd-level: cross-chunk prefix scan over K chunk states.
            chunk_data: list of (intra, final, cum_decay) per chunk
            Returns combined (B, L, S*D), final_state (B, S*D), leaf (B, L, S*D).
            """
            inter_decay = torch.cat([cd[:, -1:] for _, _, cd in chunk_data], dim=1)
            inter_input = torch.cat([f for _, f, _ in chunk_data], dim=1)
            s = initial_state.clone() if initial_state is not None else torch.zeros_like(inter_input[:, 0])
            cross_states = []
            for k in range(len(chunk_data)):
                cross_states.append(s.unsqueeze(1))  # state at start of chunk k
                s = inter_decay[:, k] * s + inter_input[:, k]  # state at end of chunk k
            cross = torch.cat(cross_states, dim=1)
            combined_pieces = []
            leaf_pieces = []
            for k, (intra_k, _, cum_decay_k) in enumerate(chunk_data):
                cross_k = cross[:, k:k+1]
                combined_pieces.append(cross_k * cum_decay_k + intra_k)
                leaf_pieces.append(intra_k)
            combined = torch.cat(combined_pieces, dim=1)
            leaf = torch.cat(leaf_pieces, dim=1)
            return combined, combined[:, -1], leaf
        
        # Level 1: parallel chunk scans from zero
        chunks = []
        for start in range(0, L, CHUNK):
            end = min(start + CHUNK, L)
            intra, final, cum_decay = _scan_chunk(input_vec_f32[:, start:end], decay_f32[:, start:end])
            chunks.append((intra, final, cum_decay))
        
        mem_all_vec, mem_state_out_vec, mem_leaf_vec = _combine_chunks(chunks, mem_state_f32)
        # Cast back to original dtype
        mem_all_vec = mem_all_vec.to(_dtype)
        mem_state_out_vec = mem_state_out_vec.to(_dtype)
        mem_leaf_vec = mem_leaf_vec.to(_dtype)
        
        mem_all_vec = mem_all_vec.view(B, L, S, D)  # (B, L, S, D)
        mem_leaf_vec = mem_leaf_vec.view(B, L, S, D)  # leaf: within-chunk only
        
        # Weighted combination: softmax over scales per channel
        w = F.softmax(self.scale_w, dim=0)  # (S, D)
        mem_all = (mem_all_vec * w.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # (B, L, D)
        mem_leaf = (mem_leaf_vec * w.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # (B, L, D) — без кросс-чанк контекста
        # Dual read: leaf (within-chunk, 100% покрытие) + context (cross-chunk)
        mem_read = mem_all * self.w_q + mem_leaf * self.w_q_leaf + mem_all * self.w_q_ctx
        mem_state_out = mem_state_out_vec.reshape(B, S * D)
        
        # First moment (same multi-scale decay, scaled input)
        if mu_state is not None:
            mu_state = mu_state.reshape(B, S * D)
        mu_input_vec = (mem_input * self.w_k_mu).unsqueeze(2).expand(-1, -1, S, -1).reshape(B, L, S * D)
        mu_chunks = []
        for start in range(0, L, CHUNK):
            end = min(start + CHUNK, L)
            intra, final, cum_decay = _scan_chunk(mu_input_vec[:, start:end], decay[:, start:end])
            mu_chunks.append((intra, final, cum_decay))
        mu_all_vec, mu_state_out_vec, _ = _combine_chunks(mu_chunks, mu_state)
        mu_all_vec = mu_all_vec.view(B, L, S, D)
        mu_all = (mu_all_vec * w.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        mu_read = mu_all * self.w_q_mu
        mem_read = mem_read + mu_read * self.w_mu_mem
        mu_state_out = mu_state_out_vec.reshape(B, S * D)
        
        # ─── Mirror (self-consistency: local + global) ───
        mirror, mlp_mod, mem_mod = self.mirror(
            h, mem_all, global_state=global_state, diff=diff,
            tanh_bias_mod=tanh_bias_mod, pred_scale_mod=pred_scale_mod,
            context_mem=context_mem, allow_write=allow_write)
        
        # ─── Output (adaptive memory scale, per-group modulation) ───
        # mem_mod: per-token, per-expert gating of memory contribution
        mm = mem_mod  # (B, L, G)
        mm = mm.unsqueeze(-1)  # (B, L, G, 1)
        g = self.mirror.G
        d = self.mirror.d
        mem_modulated = (mem_read.reshape(B, L, g, d) * mm).reshape(B, L, D)
        enhanced = bind_out + mem_modulated * self.w_mem2v * mem2v_scale + mirror
        self._cache_bind_out = (bind_out + mem_modulated * self.w_mem2v * mem2v_scale).detach()
        self._cache_mirror_out = mirror.detach()
        h = h + enhanced
        
        # ─── Spectral (adaptive: diff modulates frequency shaping) ───
        h_dct = h @ self.V_dct.T
        h = h + (h_dct * self.lambda_k * spectral_mod) @ self.V_dct
        
        # ─── MLP (per-group modulation by mlp_mod) ───
        h_mlp = self.mlp(h)
        mm2 = mlp_mod.unsqueeze(-1)  # (B, L, G, 1)
        h_mlp = (h_mlp.reshape(B, L, g, d) * mm2).reshape(B, L, D)
        h = h + h_mlp
        
        return h, (mem_state_out, mu_state_out, conv_state_out)


# ─── WideBind Stack ────────────────────────────────────────────────────

class WideBindStack(nn.Module):
    """Stack of WideBindBlock layers with embedding and lm_head."""
    
    def __init__(self, cfg: WideBindConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = PartitionedEmbedding(cfg)
        if getattr(cfg, 'zeckendorf_readout', False):
            self.lm_head = ZeckendorfReadout(cfg)
        else:
            self.lm_head = PartitionedHead(cfg, embed_basis=self.embed.basis)
        
        self.layers = nn.ModuleList([
            WideBindBlock(cfg, i) for i in range(cfg.n_layers)
        ])
        
        self.register_buffer('final_norm_w', torch.ones(cfg.D))
        # EMA for exploration (smoothed over ~500 steps)
        self.register_buffer('_expl_ema', torch.zeros(1), persistent=False)
    
    def forward(self, h, state=None, global_state=None, pred_weight=None, adaptive=True,
                context_mem=None, allow_write=None):
        """h: (B, L, D) — pre-embedded tokens
           state: per-layer memory states from previous forward (or None)
           global_state: cross-layer EMA self-model (or None, created fresh)
           pred_weight: adaptive alpha auxiliary loss weight (or None to compute)
           adaptive: if True, run AdaptiveController (training); if False, skip for speed (inference)
        """
        if state is None:
            state = [None] * len(self.layers)
        B, L, D = h.shape
        
        # ─── Adaptive gate biases from mirror stats (per-layer) ───
        if adaptive:
            with torch.no_grad():
                expl_raw, diff = AdaptiveController.stats(self.layers,
                    expl_thresh=self.cfg.exploration_threshold,
                    diff_thresh=self.cfg.differentiation_threshold)
                self._expl_ema.mul_(0.998).add_(expl_raw * (1.0 - 0.998))
                global_expl = self._expl_ema.clamp(0.0, 1.0).item()
                
                self._pred_weight = (pred_weight if pred_weight is not None
                    else AdaptiveController.pred_weight(self.layers,
                        min_val=0.05, max_val=1.0))
                
                for i, layer in enumerate(self.layers):
                    l_expl, l_diff = AdaptiveController.layer_stats(layer,
                        expl_thresh=self.cfg.exploration_threshold,
                        diff_thresh=self.cfg.differentiation_threshold)
                    
                    b_i_val = AdaptiveController.layer_b_i(layer, expl=l_expl)
                    b_d_max = getattr(self.cfg, 'vsa_b_d_max', 12.0)
                    b_d_val = AdaptiveController.layer_b_d(layer, expl=l_expl,
                        b_d_max=b_d_max)
                    smooth = getattr(self.cfg, 'vsa_b_d_smooth', 0.999)
                    if smooth >= 1.0:
                        layer.b_i.fill_(b_i_val)
                        layer.b_d.fill_(b_d_val)
                    else:
                        b_d_t = torch.tensor(b_d_val, device=layer.b_d.device, dtype=layer.b_d.dtype)
                        b_i_t = torch.tensor(b_i_val, device=layer.b_i.device, dtype=layer.b_i.dtype)
                        layer.b_d.data.lerp_(b_d_t, 1.0 - smooth)
                        layer.b_i.data.lerp_(b_i_t, 1.0 - smooth)
        
        # Global self-model: running EMA of layer memory centroids
        # Per-layer EMA rates proportional to 1/τ (Proposal V)
        n_layers = len(self.layers)
        if global_state is None:
            global_state = torch.zeros(n_layers, 1, D, device=h.device, dtype=h.dtype)
        if global_state.dim() == 2:
            global_state = global_state.unsqueeze(0).expand(n_layers, -1, -1).clone()
        elif global_state.shape[0] != n_layers:
            global_state = global_state[0:1].expand(n_layers, -1, -1).clone()
        # Calibrate c so L31 (τ≈149) matches current α≈0.976
        c_ema = (1.0 - 0.976) * 149.0  # ≈3.576
        new_state = []
        self._pred_cache = []
        for i, (layer, s) in enumerate(zip(self.layers, state)):
            if adaptive:
                l_expl, l_diff = AdaptiveController.layer_stats(layer,
                    expl_thresh=self.cfg.exploration_threshold,
                    diff_thresh=self.cfg.differentiation_threshold)
                mem2v_scale = AdaptiveController.layer_w_mem2v_scale(layer,
                    min_val=self.cfg.w_mem2v_scale_min, max_val=self.cfg.w_mem2v_scale_max,
                    diff=l_diff)
                nscale = AdaptiveController.layer_noise_scale(layer,
                    min_val=self.cfg.noise_scale_min, max_val=self.cfg.noise_scale_max,
                    diff=l_diff)
                tanh_bias_mod = AdaptiveController.tanh_bias_modulation(layer, expl=l_expl)
                spectral_mod = AdaptiveController.spectral_modulation(layer, diff=l_diff)
                pred_scale_mod = AdaptiveController.pred_scale_mod(layer)
            else:
                l_expl = l_diff = 0.5
                mem2v_scale = 1.0
                nscale = 0.0
                tanh_bias_mod = 1.0
                spectral_mod = 1.0
                pred_scale_mod = None
            
            gs_i = global_state[i:i+1].clone()  # (1, 1, D)
            h, s_out = layer(h, s, global_state=gs_i,
                             mem2v_scale=mem2v_scale, diff=l_diff, noise_scale=nscale,
                             tanh_bias_mod=tanh_bias_mod, pred_scale_mod=pred_scale_mod,
                             spectral_mod=spectral_mod,
                             context_mem=context_mem, allow_write=allow_write)
            if s_out is not None:
                mem_state_out = s_out[0]  # (B, S*D) — multi-scale memory state
                B = h.shape[0]
                S = layer._n_scales
                # Per-layer EMA alpha from τ
                lf = i / max(n_layers - 1, 1)
                tau_l = math.exp(2.0 + 3.0 * lf)
                alpha_l = 1.0 - c_ema / (tau_l + 1e-8)
                alpha_l = max(0.85, min(0.999, alpha_l))
                # Weighted combination of scales для global state
                w = F.softmax(layer.scale_w, dim=0)  # (S, D)
                mem_combined = (mem_state_out.reshape(B, S, layer.D) * w.unsqueeze(0)).sum(dim=1)
                mem_avg = mem_combined.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, D)
                global_state[i:i+1] = alpha_l * global_state[i:i+1] + (1.0 - alpha_l) * mem_avg
                s_out = tuple(t.detach() for t in s_out)
            new_state.append(s_out)
            if adaptive:
                mir = layer.mirror
                if mir._cached_pred_k is not None and mir._cached_hp is not None:
                    self._pred_cache.append((mir._cached_pred_k, mir._cached_hp))
        
        return F.rms_norm(h, (self.cfg.D,), self.final_norm_w), new_state, global_state
    
    def embed_tokens(self, tokens):
        """Token indices -> D-space vectors."""
        return self.embed(tokens)
    
    def compute_loss(self, h, targets, pred_weight=None):
        """h: (B, L, D) -> logits -> cross-entropy + alpha auxiliary loss
        pred_weight: if None, uses adaptive value from forward pass.
        """
        if isinstance(self.lm_head, ZeckendorfReadout):
            B, L, D = h.shape
            log_probs = self.lm_head.log_probs_for_target(
                h.reshape(-1, D), targets.reshape(-1))
            ce_loss = -log_probs.mean()
        else:
            logits = self.lm_head(h)
            ce = F.cross_entropy(logits.reshape(-1, self.cfg.vocab),
                                 targets.reshape(-1), reduction='none')
            # PAD/EOS masking: ignore special tokens (0=PAD, 2=EOS)
            mask = (targets.reshape(-1) != 0) & (targets.reshape(-1) != 2)
            ce = ce * mask.float()
            # Surprisal-weighted loss: w_t = (CE_t / mean(CE))^γ
            sw = getattr(self.cfg, 'surprisal_weight', 0.0)
            if self.training and sw > 0:
                with torch.no_grad():
                    w = (ce / (ce.mean() + 1e-8)).clamp(max=10.0) ** sw
                ce_loss = (ce * w).sum() / mask.sum().clamp(min=1)
            else:
                ce_loss = ce.sum() / mask.sum().clamp(min=1)
        pw = pred_weight if pred_weight is not None else getattr(self, '_pred_weight', 0.1)
        # alpha auxiliary loss: predict K-space state directly
        pred_loss = 0.0
        n_pred = 0
        cache = getattr(self, '_pred_cache', [])
        for pred_k, hp in cache:
            pred_loss = pred_loss + F.mse_loss(pred_k, hp.detach())
            n_pred = n_pred + 1
        if n_pred > 0:
            pred_loss = pred_loss / n_pred
        
        # Gate L1 sparsity: encourages expert specialization
        gate_l1 = 0.0
        n_gates = 0
        for layer in self.layers:
            g = getattr(layer.mirror, '_cached_gate_l1', None)
            if g is not None:
                gate_l1 = gate_l1 + g
                n_gates = n_gates + 1
        if n_gates > 0:
            gate_l1 = gate_l1 / n_gates
        
        # Expert reinforcement: align gate with usefulness (self-consistency)
        # High usefulness → high gate should follow (reinforcing correct self-assessment)
        reinforce_loss = 0.0
        n_reinf = 0
        for layer in self.layers:
            u = getattr(layer.mirror, '_cached_usefulness', None)
            g = getattr(layer.mirror, '_cached_gate', None)
            if u is not None and g is not None:
                reinforce_loss = reinforce_loss + F.mse_loss(u, g)
                n_reinf = n_reinf + 1
        if n_reinf > 0:
            reinforce_loss = reinforce_loss / n_reinf
        
        # Load balancing: encourage uniform expert usage across tokens
        # CV of per-expert usage = std/mean — lower means all experts used equally
        balance_loss = 0.0
        n_bal = 0
        for layer in self.layers:
            usage = getattr(layer.mirror, '_cached_gate_usage', None)
            if usage is not None:
                usage_p = usage / (usage.sum() + 1e-10)
                ue = -(usage_p * torch.log(usage_p + 1e-10)).sum()
                logG = math.log(usage.shape[-1])
                balance_loss = balance_loss + (logG - ue) / logG
                n_bal = n_bal + 1
        if n_bal > 0:
            balance_loss = balance_loss / n_bal
        
        # Diversity loss: decorrelate per-group MLP outputs
        # ||cov(||group_out||_g) - I||_F² → каждая группа ортогональна другим
        diversity_loss = 0.0
        n_div = 0
        for layer in self.layers:
            group_out = getattr(layer.mlp, '_cached_group_out', None)
            if group_out is not None:
                B, L, G, d = group_out.shape
                y = group_out.norm(dim=-1).reshape(-1, G)  # (B*L, G)
                y = y - y.mean(dim=0, keepdim=True)
                cov = y.T @ y / (y.shape[0] - 1 + 1e-10)
                div = F.mse_loss(cov, torch.eye(G, device=cov.device))
                diversity_loss = diversity_loss + div
                n_div = n_div + 1
        if n_div > 0:
            diversity_loss = diversity_loss / n_div
        
        # Nuclear norm regularization for bind W_proj
        # stochastic estimate: ||W||_* ≈ mean(||Wv||₂) · sqrt(K)
        nuc_loss = 0.0
        n_nuc = 0
        nuc_iters = 5
        for layer in self.layers:
            W = getattr(layer, 'W_proj', None)
            if W is not None:
                v = torch.randn(W.shape[1], nuc_iters, device=W.device)
                Wv = W @ v
                nuc = Wv.norm(dim=0).mean() * math.sqrt(W.shape[1])
                nuc_loss = nuc_loss + nuc
                n_nuc = n_nuc + 1
        if n_nuc > 0:
            nuc_loss = nuc_loss / n_nuc
        
        # Orthogonality regularization for bottleneck bind: ||Ŵ^T Ŵ - I||_F²
        orth_loss = 0.0
        n_orth = 0
        for layer in self.layers:
            W = getattr(layer, 'W_proj', None)
            if W is not None:
                W_hat = W / W.norm(dim=0, keepdim=True).clamp(min=1e-8)
                gram = W_hat.T @ W_hat  # (K, K)
                orth = F.mse_loss(gram, torch.eye(gram.shape[0], device=gram.device))
                orth_loss = orth_loss + orth
                n_orth = n_orth + 1
        if n_orth > 0:
            orth_loss = orth_loss / n_orth
        
        # w_m2v hierarchy by τ (Proposal IV): push w_m2v toward target ∝ σ(ln τ)
        w_m2v_loss = 0.0
        n_m2v = 0
        w_m2v_weight = getattr(self.cfg, 'w_m2v_hierarchy_weight', 0.0)
        if w_m2v_weight > 0:
            for i, layer in enumerate(self.layers):
                wm = getattr(layer, 'w_mem2v', None)
                if wm is not None:
                    lf = i / max(len(self.layers) - 1, 1)
                    tau_l = math.exp(2.0 + 3.0 * lf)
                    target = getattr(self.cfg, 'w_m2v_hierarchy_target', 1.0)
                    target_m2v = target / (1.0 + math.exp(-(math.log(tau_l) - math.log(32.0))))
                    w_m2v_loss = w_m2v_loss + F.mse_loss(wm.mean(), torch.tensor(target_m2v, device=wm.device))
                    n_m2v = n_m2v + 1
            if n_m2v > 0:
                w_m2v_loss = w_m2v_loss / n_m2v
        # Branch balance: equalize log-variance of conv/bind/mirror branches
        branch_loss = 0.0
        n_branch = 0
        branch_weight = getattr(self.cfg, 'branch_balance_weight', 0.0)
        if branch_weight > 0:
            for layer in self.layers:
                conv = getattr(layer, '_cache_conv_out', None)
                bnd = getattr(layer, '_cache_bind_out', None)
                mir = getattr(layer, '_cache_mirror_out', None)
                if conv is not None and bnd is not None and mir is not None:
                    vc = conv.norm(dim=-1).var() + 1e-10
                    vb = bnd.norm(dim=-1).var() + 1e-10
                    vm = mir.norm(dim=-1).var() + 1e-10
                    branch_loss = branch_loss + (torch.log(vc) - torch.log(vb)).pow(2)
                    branch_loss = branch_loss + (torch.log(vc) - torch.log(vm)).pow(2)
                    branch_loss = branch_loss + (torch.log(vb) - torch.log(vm)).pow(2)
                    n_branch = n_branch + 3
            if n_branch > 0:
                branch_loss = branch_loss / n_branch
        l1_weight = getattr(self.cfg, 'gate_l1_weight', 0.001)
        reinforce_weight = getattr(self.cfg, 'reinforce_weight', 0.01)
        balance_weight = getattr(self.cfg, 'balance_weight', 0.01)
        diversity_weight = getattr(self.cfg, 'diversity_weight', 0.001)
        nuc_weight = getattr(self.cfg, 'nuclear_weight', 1e-5)
        orth_weight = getattr(self.cfg, 'orth_weight', 1e-4)
        # Mirror diversity: push var(log_scale) up (expert specialization)
        div_w = getattr(self.cfg, 'div_weight', 0.0)
        div_loss = 0.0
        if div_w > 0:
            all_ls = torch.cat([layer.mirror.log_scale for layer in self.layers])  # (L*G, d)
            div_loss = -div_w * all_ls.var()
        # Signal balance: entropy regularization on signal weights (encourages uniform use of all signals)
        signal_entropy = 0.0
        n_sig = 0
        for layer in self.layers:
            w = torch.softmax(layer.mirror._signal_log_weights, dim=0)
            signal_entropy = signal_entropy - (w * torch.log(w + 1e-10)).sum()
            n_sig = n_sig + 1
        if n_sig > 0:
            signal_entropy = signal_entropy / n_sig
        signal_entropy_weight = getattr(self.cfg, 'signal_entropy_weight', 0.001)
        log_scale_l2_weight = getattr(self.cfg, 'log_scale_l2_weight', 0.01)
        log_scale_reg = 0.0
        n_ls = 0
        for layer in self.layers:
            ls = layer.mirror.log_scale
            excess = (ls - 2.3).clamp(min=0)
            log_scale_reg = log_scale_reg + excess.pow(2).mean()
            n_ls = n_ls + 1
        if n_ls > 0:
            log_scale_reg = log_scale_reg / n_ls
        return ce_loss + pw * pred_loss + l1_weight * gate_l1 + reinforce_weight * reinforce_loss \
            + balance_weight * balance_loss + diversity_weight * diversity_loss \
            + nuc_weight * nuc_loss + orth_weight * orth_loss \
            + w_m2v_weight * w_m2v_loss + branch_weight * branch_loss + div_loss \
            - signal_entropy_weight * signal_entropy \
            + log_scale_l2_weight * log_scale_reg
    
    def param_count(self):
        return sum(p.numel() for p in self.parameters())
    
    def param_groups(self, lr=None, weight_decay=None, gate_lr_mult=None):
        """Optimizer parameter groups with λ_d LR hierarchy or legacy flat groups.
        
        When cfg.lambda_lr_hierarchy=True (default), groups follow λ_d^p:
          p=-2: embedding, readout       (0.29×)
          p=-1: MLP cores, bind W_proj   (0.54×)
          p= 0: conv, norm, W_out, head  (1.00×)
          p=+1: mirror projections, α    (1.84×)
          p=+2: gates, w_i, b_i, etc     (3.38×)
          vsa:  b_d, b_i                 (λ^{-4} ≈ 0.087×)
        """
        cfg = self.cfg
        lr = lr or cfg.lr
        wd = weight_decay or cfg.weight_decay
        
        if getattr(cfg, 'lambda_lr_hierarchy', False):
            from .lambda_utils import lambda_d
            lam = lambda_d(cfg.lambda_d)
            mlr = {
                'embed': lam ** (-2),
                'mlp': lam ** (-1),
                'vsa': lam ** (-4),
                'mirror': lam ** (1),
                'gate': lam ** (2),
            }
            groups = {
                'embed':    {'params': [], 'lr': lr * mlr['embed'], 'weight_decay': 0},
                'embed_wd': {'params': [], 'lr': lr * mlr['embed'], 'weight_decay': wd},
                'mlp':      {'params': [], 'lr': lr * mlr['mlp'],   'weight_decay': 0},
                'mlp_wd':   {'params': [], 'lr': lr * mlr['mlp'],   'weight_decay': wd},
                'mirror':   {'params': [], 'lr': lr * mlr['mirror'],'weight_decay': 0},
                'mirror_wd':{'params': [], 'lr': lr * mlr['mirror'],'weight_decay': wd},
                'gate':     {'params': [], 'lr': lr * mlr['gate'],  'weight_decay': 0},
                'gate_wd':  {'params': [], 'lr': lr * mlr['gate'],  'weight_decay': wd},
                'vsa':      {'params': [], 'lr': lr * mlr['vsa'],   'weight_decay': 0},
                'default':  {'params': [], 'lr': lr,                'weight_decay': 0},
                'default_wd':{'params': [], 'lr': lr,               'weight_decay': wd},
            }
            for name, p in self.named_parameters():
                if '.b_d' in name or '.b_i' in name:
                    groups['vsa']['params'].append(p)
                elif name.startswith('embed.') or name.startswith('lm_head.readout') or name.startswith('lm_head.proj'):
                    k = 'embed_wd' if p.ndim >= 2 else 'embed'
                    groups[k]['params'].append(p)
                elif '.mlp.' in name or name.endswith('.W_proj') or name.endswith('.W_out'):
                    # Block-level W_proj/W_out (not mirror) → mlp speed
                    k = 'mlp_wd' if p.ndim >= 2 else 'mlp'
                    groups[k]['params'].append(p)
                elif any(g in name for g in ['.w_gate', '.b_gate', '.w_delta_gate', '.b_delta_gate',
                                              '.w_i', '.w_d', '.w_q', '.w_q_leaf', '.w_q_ctx', '.w_mem2v',
                                              '.w_k_mu', '.w_q_mu', '.w_mu_mem',
                                              '.w_u', '.w_v']):
                    k = 'gate_wd' if p.ndim >= 2 else 'gate'
                    groups[k]['params'].append(p)
                elif any(g in name for g in ['.mirror.alpha_diag', '.mirror.w_pred_scale_legacy',
                                              '.log_skip_alpha', '.mirror.W_proj', '.mirror.W_out',
                                              '.mirror.w_temp', '.mirror.w_global',
                                              '.mirror.log_scale', '.mirror.tanh_bias',
                                              '.log_dvar_mod_scale', '.dvar_mod_bias',
                                              '.log_grad_mod_scale', '.grad_mod_bias']):
                    k = 'mirror_wd' if p.ndim >= 2 else 'mirror'
                    groups[k]['params'].append(p)
                else:
                    k = 'default_wd' if p.ndim >= 2 else 'default'
                    groups[k]['params'].append(p)
            return [v for v in groups.values() if v['params']]
        
        # ─── Legacy groups (lambda_lr_hierarchy=False) ───
        gate_lr_mult = cfg.gate_lr_mult if gate_lr_mult is None else gate_lr_mult
        decay = []
        no_decay = []
        gate_decay = []
        gate_no_decay = []
        vsa_bias = []
        for name, p in self.named_parameters():
            if '.b_d' in name or '.b_i' in name:
                vsa_bias.append(p)
                continue
            is_gate = any(g in name for g in ['.w_i', '.w_d', '.w_q', '.w_q_leaf', '.w_q_ctx', '.w_mem2v',
                                               '.w_k_mu', '.w_q_mu', '.w_mu_mem',
                                               '.w_u', '.w_v',
                                               '.tanh_bias', '.log_scale',
                                               '.mirror.W_proj', '.mirror.W_out',
                                               '.mirror.w_temp', '.mirror.w_global',
                                                '.mirror.alpha_diag', '.mirror.w_pred_scale_legacy',
                                               '.mirror.w_gate', '.mirror.b_gate',
                                               '.log_dvar_mod_scale', '.dvar_mod_bias',
                                               '.log_grad_mod_scale', '.grad_mod_bias',
                                               '.log_skip_alpha'])
            if is_gate:
                if p.ndim < 2 or 'w_pred_scale_legacy' in name:
                    gate_no_decay.append(p)
                else:
                    gate_decay.append(p)
            else:
                if p.ndim < 2:
                    no_decay.append(p)
                else:
                    decay.append(p)
        groups = [
            {'params': decay, 'lr': lr, 'weight_decay': wd},
            {'params': no_decay, 'lr': lr, 'weight_decay': 0},
        ]
        if gate_decay:
            groups.append({'params': gate_decay, 'lr': lr * gate_lr_mult, 'weight_decay': wd})
        if gate_no_decay:
            groups.append({'params': gate_no_decay, 'lr': lr * gate_lr_mult, 'weight_decay': 0})
        if vsa_bias:
            vsa_lr_mult = getattr(cfg, 'vsa_b_lr_mult', 0.1)
            groups.append({'params': vsa_bias, 'lr': lr * vsa_lr_mult, 'weight_decay': 0})
        return groups


# ─── Adaptive Controller ──────────────────────────────────────────────

class AdaptiveController:
    """
    Computes ALL adaptive hyperparameters from cognitive mirror state.

    Two fundamental signals drive every parameter:
    ──────────────────────────────────────────────────────────
    exploration = min(1, |mirror| / λ⁻²)
        How much correction is the mirror applying.
        High → model is actively adjusting, needs aggressive config.
        Low → model is stable, needs conservative config.

    differentiation = min(1, var(log_scale) / λ⁻⁴)
        How specialized has the mirror become (per-dim scaling).
        High → mirror has learned which dims to trust/suppress.
        Low → mirror hasn't differentiated, still exploring.

    λ_d hierarchy (d=3): λ₃ ≈ 1.839, λ⁻² ≈ 0.296, λ⁻⁴ ≈ 0.087
    All range defaults below are λ_d d=3 derived.

    Key design: ALL methods work at per-layer AND global resolution.
    ``layer_stats(layer)`` → per-layer (expl, diff)
    ``stats(blocks)`` → global average   (backward compat)

    New intelligent adaptivity:
    ──────────────────────────
    - ``pred_weight(blocks)`` — alpha loss weight scales with diff
      (more temporal learning when mirror has specialized)
    - ``tanh_bias_modulation(layer)`` — tanh_bias amplified by exploration
      (more asymmetric correction when actively exploring)
    - ``spectral_modulation(layer)`` — lambda_k amplified by differentiation
      (more aggressive freq shaping when experts are specialized)
    - ``pred_scale_mod(layer)`` — per-expert w_pred_scale modulation
      from delta_var (experts with volatile dynamics get more
      temporal teaching signal)

    Mathematically derived ranges (λ_d d=3):
    ────────────────────────────────────────
    b_d ∈ [b_d_min, b_d_max] per layer, where b_d_min = 2.0 + 3.0*layer_frac
         expl=1 → b_d = b_d_min (shortest memory)
         expl=0 → b_d = b_d_max (longest memory, configurable vsa_b_d_max)
         L0: τ≈[7, 150] (default b_d_max=5.0), up to τ≈160K (b_d_max=12.0)
         Per-channel via gradient: b_d is (D,) with lerp-slow push to controller target
    b_i  ∈ [-3.0, -1.5] → i_gate ≈ [0.047, 0.18] (write rate via softplus)
    w_mem2v_scale ∈ [0.544, 1.0]  (memory contribution, λ⁻¹ to 1)
    ema_alpha ∈ [0.974, 0.992]  (cross-layer memory, 1-λ⁻⁶ to 1-λ⁻⁸)
    noise_scale ∈ [0.0076, 0.026]  (parameter noise, λ⁻⁸ to λ⁻⁶)
    pred_weight ∈ [0.026, 0.296]  (alpha loss weight, λ⁻⁶ to λ⁻²)
    tanh_bias_mod ∈ [1.0, 1.5]  (exploration amplification)
    spectral_mod ∈ [0.913, 1.087]  (differentiation, 1±λ⁻⁴)
    """
    @staticmethod
    def layer_stats(layer, expl_thresh=0.296, diff_thresh=0.087):
        """Per-layer (exploration, differentiation) from a single block."""
        m = layer.mirror
        ls = m.log_scale.data
        var = ls.var().item()
        mag = m._last_magnitude.item()
        return min(1.0, mag / expl_thresh), min(1.0, var / diff_thresh)

    @staticmethod
    def stats(blocks, expl_thresh=0.296, diff_thresh=0.087):
        """Global average (exploration, differentiation) across all layers."""
        expl_sum = diff_sum = 0.0
        for layer in blocks:
            e, d = AdaptiveController.layer_stats(layer, expl_thresh, diff_thresh)
            expl_sum += e
            diff_sum += d
        n = len(blocks)
        return expl_sum / n, diff_sum / n

    # ─── Per-layer methods ────────────────────────────────────────

    @staticmethod
    def layer_b_d(layer, expl=None, b_d_max=5.0):
        """Per-layer decay bias. Layer uses its own exploration."""
        if expl is None:
            expl, _ = AdaptiveController.layer_stats(layer)
        lf = getattr(layer, 'layer_idx', 0) / max(getattr(layer, 'total_layers', 32) - 1, 1)
        b_d_min = 2.0 + 3.0 * lf
        b_d_val = b_d_max - expl * (b_d_max - b_d_min)
        return max(2.0, min(b_d_max, b_d_val))

    @staticmethod
    def layer_b_i(layer, expl=None):
        """Per-layer write gate bias. Нормировка: i_gate ∝ 1/τ.
        
        i_gate = softplus(b_i_l). Равновесная норма памяти:
            ‖M_l‖ = i_gate · ‖h‖ · τ_l
        Для ‖M_l‖ = const по слоям: i_gate ∝ 1/τ_l.
        
        Базовое значение: i_gate_ref = 0.182 при τ_ref ≈ 32.
        c = 0.182 · 32 ≈ 5.83.
        i_gate_l = c / τ_l  →  b_i_l = softplus⁻¹(c / τ_l)
        """
        if expl is None:
            expl, _ = AdaptiveController.layer_stats(layer)
        # Базовый b_i от exploration
        b_i_base = -3.0 + expl * 1.5
        # c = 0.182 · 32 ≈ 5.83
        c = 5.83
        lf = getattr(layer, 'layer_idx', 0) / max(getattr(layer, 'total_layers', 32) - 1, 1)
        # τ_l ≈ 8 · (1 + 3.5 · lf)  (аппроксимация линейного роста 8→149 по слоям)
        tau_l = 8.0 + 141.0 * lf
        i_target = min(1.0, c / tau_l)  # насыщение на 1.0 для L0
        # softplus⁻¹(x) = log(exp(x)-1), но для численной стабильности:
        # b_i = log(exp(i_target) - 1) ≈ log(i_target) для малых i_target
        b_i_tau = math.log(max(i_target, 1e-6))
        return b_i_base + b_i_tau

    @staticmethod
    def layer_w_mem2v_scale(layer, min_val=0.544, max_val=1.0, diff=None):
        """Per-layer memory contribution."""
        if diff is None:
            _, diff = AdaptiveController.layer_stats(layer)
        return max_val - diff * (max_val - min_val)

    @staticmethod
    def layer_noise_scale(layer, min_val=0.0076, max_val=0.026, diff=None):
        """Per-layer parameter noise."""
        if diff is None:
            _, diff = AdaptiveController.layer_stats(layer)
        return max_val - diff * (max_val - min_val)

    @staticmethod
    def layer_ema_alpha(layer, min_val=0.974, max_val=0.992, diff=None):
        """Per-layer EMA rate (for per-layer global_state aggregation)."""
        if diff is None:
            _, diff = AdaptiveController.layer_stats(layer)
        return min_val + diff * (max_val - min_val)

    # ─── New intelligent adaptivity ───────────────────────────────

    @staticmethod
    def pred_weight(blocks, min_val=0.026, max_val=0.296):
        """Adaptive alpha auxiliary loss weight.

        When mirror has differentiated (high diff), temporal prediction
        is more meaningful → increase pred_weight to drive alpha learning.
        When mirror hasn't specialized, pred would be noise → keep low.
        """
        _, diff = AdaptiveController.stats(blocks)
        return min_val + diff * (max_val - min_val)

    @staticmethod
    def tanh_bias_modulation(layer, expl=None):
        """Scale tanh_bias by exploration.

        High exploration → more asymmetric correction needed → amplify.
        Range: [1.0, 1.296] (at most 1+λ⁻² boost).
        """
        if expl is None:
            expl, _ = AdaptiveController.layer_stats(layer)
        return 1.0 + 0.296 * expl

    @staticmethod
    def spectral_modulation(layer, diff=None):
        """Modulate spectral lambda_k by differentiation.

        High diff → mirror has learned structure → amplify spectral
        contrast (more aggressive frequency shaping).
        Low diff → flatten spectral response (conservative).
        Range: [0.913, 1.087] = 1 ± λ⁻⁴.
        """
        if diff is None:
            _, diff = AdaptiveController.layer_stats(layer)
        return 1.0 + 0.087 * (diff - 0.5) * 2.0  # 0.913 at diff=0, 1.087 at diff=1

    @staticmethod
    def pred_scale_mod(layer):
        """Per-expert w_pred_scale modulation from delta_var.
        
        Experts with volatile K-space dynamics (high delta_var relative
        to layer average) get more temporal teaching signal.
        Uses tanh-based soft normalization instead of division to avoid NaN.
        Range: [0.5, 2.0] centered at 1.0.
        """
        dv = layer.mirror._delta_var
        dv_centered = dv - dv.mean()
        return (1.0 + 0.5 * torch.tanh(dv_centered)).clamp(0.1, 3.0)

    # ─── Global (backward-compat) wrappers ────────────────────────

    @staticmethod
    def b_d(blocks, b_d_max=5.0):
        expl, _ = AdaptiveController.stats(blocks)
        return b_d_max - expl * 2.0

    @staticmethod
    def b_i(blocks):
        expl, _ = AdaptiveController.stats(blocks)
        return -3.0 + expl * 1.5

    @staticmethod
    def w_mem2v_scale(blocks, min_val=0.544, max_val=1.0):
        _, diff = AdaptiveController.stats(blocks)
        return max_val - diff * (max_val - min_val)

    @staticmethod
    def ema_alpha(blocks, min_val=0.974, max_val=0.992):
        _, diff = AdaptiveController.stats(blocks)
        return min_val + diff * (max_val - min_val)

    @staticmethod
    def noise_scale(blocks, min_val=0.0076, max_val=0.026):
        _, diff = AdaptiveController.stats(blocks)
        return max_val - diff * (max_val - min_val)


class MirrorLRScheduler:
    """LR scheduler modulated by cognitive mirror state dynamics.

    Growth-ratio multipliers (neutral at growth=1):
      var/alpha/gate growth  →  LR up when specialization grows, down when stalled
    mag_factor (cap): |mirror| above threshold → LR reduced (counter-cyclical)
    Loss damping (persistent): val_loss regression >2% → _loss_lr_factor halved
      (ReduceLROnPlateau semantics; resets to 1.0 on new best).
    """
    def __init__(self, model, optimizer, base_lr=None, warmup=1000,
                 target_var=0.161, mag_threshold=0.296, lr_min_ratio=0.026,
                 max_decay_steps=2584, var_min_for_lr_decay=0.008,
                 cfg=None):
        if cfg is not None:
            base_lr = base_lr or cfg.lr
            warmup = getattr(cfg, 'warmup_steps', warmup)
            target_var = getattr(cfg, 'target_var', target_var)
            mag_threshold = getattr(cfg, 'mag_threshold', mag_threshold)
            lr_min_ratio = getattr(cfg, 'lr_min_ratio', lr_min_ratio)
            max_decay_steps = getattr(cfg, 'max_decay_steps', max_decay_steps)
            var_min_for_lr_decay = getattr(cfg, 'var_min_for_lr_decay', var_min_for_lr_decay)
        self.model = model
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup = warmup
        self.target_var = target_var
        self.mag_threshold = mag_threshold
        self.lr_min_ratio = lr_min_ratio
        self.max_decay_steps = max_decay_steps
        self.var_min_for_lr_decay = var_min_for_lr_decay
        self._step = 0
        self._last_log = 0
        self._init_var = None
        self._init_1malpha = None
        self._init_gate_var = None
        self._best_val_loss = float('inf')
        self._loss_lr_factor = 1.0  # persistent damping factor (1.0 = no damping)
        self._pending_val_loss = None
        self._train_loss_tracker = []  # rolling training loss window for trend detection
        self._train_loss_lr_factor = 1.0

    def _mirror_stats(self):
        var_sum = 0.0
        mag_sum = 0.0
        alpha_sum = 0.0
        gate_var_sum = 0.0
        n = len(self.model.layers)
        for layer in self.model.layers:
            m = layer.mirror
            ls = m.log_scale.data
            var_sum += ls.var().item()
            mag_sum += m._last_magnitude.item()
            alpha = m.alpha_diag.data
            alpha_sum += (1.0 - alpha).abs().mean().item()
            gate_var_sum += m._last_gates.var().item()
        return var_sum / n, mag_sum / n, alpha_sum / n, gate_var_sum / n

    def report_train_loss(self, train_loss):
        """Report training loss for LR damping. Detects sustained increase, reduces LR."""
        self._train_loss_tracker.append(train_loss)
        if len(self._train_loss_tracker) > 500:
            self._train_loss_tracker.pop(0)
        if len(self._train_loss_tracker) >= 300:
            old = sum(self._train_loss_tracker[-300:-200]) / 100
            recent = sum(self._train_loss_tracker[-100:]) / 100
            if recent > old * 1.05 and self._train_loss_lr_factor > 0.1:
                self._train_loss_lr_factor = max(0.1, self._train_loss_lr_factor * 0.7)
                print(f'  TRAIN LR DAMPED: recent={recent:.4f} > old={old:.4f}, '
                      f'factor {self._train_loss_lr_factor:.3f}')

    def report_val_loss(self, val_loss):
        """Report validation loss for LR damping. Called from training code after eval."""
        self._pending_val_loss = val_loss

    def _consume_pending_val_loss(self):
        """Update persistent damping factor from reported val_loss."""
        if self._pending_val_loss is None:
            return
        vl = self._pending_val_loss
        self._pending_val_loss = None
        if vl < self._best_val_loss:
            if self._loss_lr_factor < 1.0:
                print(f'  LR RESTORED: val_loss={vl:.4f} new best, factor 1.0')
            self._best_val_loss = vl
            self._loss_lr_factor = 1.0
        elif vl > self._best_val_loss * 1.02:
            old = self._loss_lr_factor
            self._loss_lr_factor = max(0.1, self._loss_lr_factor * 0.5)
            if self._loss_lr_factor < old:
                print(f'  LR DAMPED: val_loss={vl:.4f} > best={self._best_val_loss:.4f}, '
                      f'factor {old:.3f} -> {self._loss_lr_factor:.3f}')

    def step(self):
        self._step += 1
        self._consume_pending_val_loss()
        # Alpha override: smoothly blended into learned alpha_diag by forward()
        # override=1.0 → 0.0 provides smooth transition from identity to learned
        warmup_end = self.warmup
        blend_steps = 50
        if self._step < warmup_end + blend_steps:
            if self._step < warmup_end:
                mult = self._step / max(warmup_end, 1)
                override = max(0.0, 1.0 - mult * 0.7)  # 1.0 → 0.3
            else:
                blend = (self._step - warmup_end) / blend_steps  # 0 → 1
                mult = 1.0 - blend * 0.3  # плавно 1.0 → 0.7
                override = 0.3 * max(0.0, 1.0 - blend)  # 0.3 → 0.0
            # Temperature annealing during warmup (homeostatica接管 after warmup)
            temp_max, temp_min = 2.0, 0.5
            if self._step < warmup_end:
                t = self._step / max(warmup_end, 1)
                temp = temp_max - t * (temp_max - temp_min)
            else:
                blend = min(1.0, (self._step - warmup_end) / blend_steps)
                temp = temp_min + (1.0 - blend) * (temp_max - temp_min) * 0.3
            for layer in self.model.layers:
                layer.mirror._alpha_override.fill_(override)
                layer.mirror._usefulness_temp.fill_(max(temp, 0.1))
        else:
            for layer in self.model.layers:
                layer.mirror._alpha_override.fill_(0.0)
            var, mag, mean_1malpha, gate_var = self._mirror_stats()

            if self._init_var is None:
                self._init_var = var + 1e-10
                self._init_1malpha = mean_1malpha + 1e-10
                self._init_gate_var = gate_var + 1e-10

            # EMA smoothing to reduce noise (τ~100 steps at 0.99)
            if not hasattr(self, '_var_ema'):
                self._var_ema = var
                self._1malpha_ema = mean_1malpha
                self._gate_var_ema = gate_var
            ema = 0.99
            self._var_ema = ema * self._var_ema + (1 - ema) * var
            self._1malpha_ema = ema * self._1malpha_ema + (1 - ema) * mean_1malpha
            self._gate_var_ema = ema * self._gate_var_ema + (1 - ema) * gate_var
            var, mean_1malpha, gate_var = self._var_ema, self._1malpha_ema, self._gate_var_ema

            # Counter-cyclical multipliers: LR down when volatility grows, up when stagnant
            var_growth = var / self._init_var
            var_mult = min(2.0, max(0.5, 1.0 / max(var_growth, 1e-10)))

            alpha_growth = mean_1malpha / self._init_1malpha
            alpha_mult = min(2.0, max(0.5, 1.0 / max(alpha_growth, 1e-10)))

            gate_growth = gate_var / self._init_gate_var
            gate_mult = min(2.0, max(0.5, 1.0 / max(gate_growth, 1e-10)))

            # Magnitude cap: |mirror| above threshold reduces LR (counter-cyclical)
            mag_factor = min(1.0, max(0.2, self.mag_threshold / max(mag, 1e-10)))

            mirror_mult = min(var_mult, alpha_mult, gate_mult) * mag_factor
            mult = max(0.05, min(1.0, mirror_mult))

            # Persistent loss damping applied every step
            mult *= self._loss_lr_factor * self._train_loss_lr_factor

            if self._step - self._last_log >= 500:
                self._last_log = self._step
                print(f'  lr_adapt: var(ls)={var:.6f} |1-a|={mean_1malpha:.6f} '
                      f'gate_var={gate_var:.6f} |mirror|={mag:.4f} '
                      f'mult={mult:.4f} damp={self._loss_lr_factor:.3f} lr={self.base_lr*mult:.2e}')

        for pg in self.optimizer.param_groups:
            pg['lr'] = self.base_lr * mult

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {
            'step': self._step,
            'last_log': self._last_log,
            'type': 'MirrorLRScheduler',
            'init_var': self._init_var,
            'init_1malpha': self._init_1malpha,
            'init_gate_var': self._init_gate_var,
            'best_val_loss': self._best_val_loss,
            'loss_lr_factor': self._loss_lr_factor,
            'train_loss_lr_factor': self._train_loss_lr_factor,
            'train_loss_tracker': self._train_loss_tracker,
        }

    def load_state_dict(self, sd):
        self._step = sd.get('step', 0)
        self._last_log = sd.get('last_log', 0)
        self._init_var = sd.get('init_var')
        self._init_1malpha = sd.get('init_1malpha')
        self._init_gate_var = sd.get('init_gate_var')
        self._best_val_loss = sd.get('best_val_loss', float('inf'))
        self._loss_lr_factor = sd.get('loss_lr_factor', 1.0)
        self._train_loss_lr_factor = sd.get('train_loss_lr_factor', 1.0)
        self._train_loss_tracker = sd.get('train_loss_tracker', [])

    def reset_for_new_data(self, reset_warmup_steps=2000):
        """Call when dataset changes (e.g. switching from ADVENTUR to FANTASY).
        Resets loss damping and reruns warmup to prevent spurious LR damping."""
        self._best_val_loss = float('inf')
        self._loss_lr_factor = 1.0
        self._train_loss_lr_factor = 1.0
        self._train_loss_tracker = []
        self._pending_val_loss = None
        self._init_var = None
        self._init_1malpha = None
        self._init_gate_var = None
        if hasattr(self, '_var_ema'):
            del self._var_ema
            del self._1malpha_ema
            del self._gate_var_ema


# ─── Verify ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    cfg = WideBindConfig(n_layers=24, D=896, bottleneck=896, bind_K=32, mlp_groups=8)
    model = WideBindStack(cfg).to(device)
    n = model.param_count()
    print(f'  D=896 G=8: params={n:,} ({n/1e6:.2f}M)')
    
    print()
    cfg = WideBindConfig(n_layers=4, D=896, bottleneck=896, bind_K=32)
    model = WideBindStack(cfg).to(device)
    
    x = torch.randint(0, cfg.vocab, (2, 16), device=device)
    h = model.embed_tokens(x)
    out, state, _ = model(h)
    loss = model.compute_loss(out[:, :-1], x[:, 1:])
    loss.backward()
    
    total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    out_std = out.std().item()
    print(f'Output: {out.shape}  std={out_std:.4f}')
    print(f'Loss: {loss.item():.4f}  Grad: {total_grad:.4f}')
    print('OK' if not math.isnan(loss.item()) and total_grad > 0 else 'FAIL')
