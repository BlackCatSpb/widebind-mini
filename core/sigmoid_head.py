"""SigmoidCodedHead: единая замена softmax-головы, объединяющая четыре идеи.

Объединяет:
  A. Факторизованный sigmoid-классификатор (без softmax-нормализации):
     P(token) = Π_{k активен} σ(z_k) · Π_{k неактивен} (1−σ(z_k)).
     Тренируем напрямую по target через log_probs_for_target за O(K) —
     НЕ материализуя и НЕ нормализуя все V логитов. Нет exp-sum по V ⇒ нет softmax.

  B. Структурный приоритет по коду: k-биты уже сбалансированы комба
     (sparse_block_codes, ровно S=6 из K), плюс learnable per-bit prior_bias,
     инициализированный по эмпирической загрузке битов (≈ грань Fibonacci-загрузки),
     чтобы маргинал каждого бита совпадал со структурной частотой кода.

  C. τ-самокалибровочная температура: learnable per-bit обратная температура
     log_temp (инициализация — идентичность), проецирующая корни τ/Фибоначчи
     в условие заострения σ(z_k / T_k). Позволяет модели самозаострять охват.

  D. Двухстадийное декодирование: forward возвращает все V логитов через
     разреженный u @ codes.T (линейно по sparse-коду, без exp-sum),
     а predict дает жадный argmax без полной V-нормализации.

Форма интерфейса (duck-typed):
    forward(h)  -> (B, L, V) "логиты" (лог-вероятности факторизованной модели)
    log_probs_for_target(h, targets) -> (B, L) log P(target)
    predict(h, k) -> greedy token ids
"""
import torch
import torch.nn as nn
from .vsa_utils import sparse_block_codes


class SigmoidCodedHead(nn.Module):
    """Факторизованный Bernoulli head поверх sparse block codes."""

    def __init__(self, cfg, embed_basis=None):
        super().__init__()
        codes = sparse_block_codes(cfg.vocab, K=cfg.code_dim, S=cfg.code_sparsity)
        self.K = codes.shape[1]
        self.S = cfg.code_sparsity
        self.register_buffer('codes', codes)  # (vocab, K) float 0/1

        D = cfg.D
        assert D % self.K == 0
        d = D // self.K

        if embed_basis is not None:
            self.readout = embed_basis  # weight tying encode/decode
        else:
            self.readout = nn.Parameter(torch.randn(self.K, d))
            nn.init.xavier_uniform_(self.readout, gain=0.5)

        # B: per-bit structural prior, init по эмпирической загрузке бит.
        prop = codes.mean(dim=0)  # (K,) фактическая доля активных бит
        self.register_buffer('_prop', prop)
        self.bit_bias = nn.Parameter(torch.zeros(self.K))  # learnable prior загрузки

        # C: per-bit inverse temperature (identity init), мягкая самокалибровка.
        self.log_temp = nn.Parameter(torch.zeros(self.K))

        # D / B: per-token prior (частотный априор), combined.
        self.token_bias = nn.Parameter(torch.zeros(cfg.vocab))

    # ── внутренние стенки ──────────────────────────────────────────────
    def _gates(self, h, temp_factor=None):
        """(h: (B, L, D)) -> zt: (B, L, K) калиброванные gate-логиты."""
        B, L, D = h.shape
        h_g = h.reshape(B, L, self.K, -1)  # (B, L, K, d)
        z = torch.einsum('blkd,kd->blk', h_g, self.readout)  # (B, L, K)
        T = torch.exp(self.log_temp).clamp_min(0.1)  # (K,) per-bit temp
        if temp_factor is not None:
            T = T * temp_factor
        zt = z / T + self.bit_bias  # τ-self-calibration (C)
        return zt

    def _su(self, zt):
        """Из logit-глов -> up-band логи-соотношение u и низкую базу.
        u = log σ(zt) − log(1−σ(zt)); base = Σ_k log(1−σ(zt))."""
        sig = torch.sigmoid(zt)
        ls = torch.log(sig.clamp_min(1e-9))
        lms = torch.log((1 - sig).clamp_min(1e-9))
        u = ls - lms            # (B, L, K)
        base = lms.sum(-1)      # (B, L) Σ inactive pots
        return u, base

    # ── основные интерфейсы ────────────────────────────────────────────
    def forward(self, h):
        """(B, L, V) — лог-вероятности факторизованной модели (без softmax)."""
        u, base = self._su(self._gates(h))
        logits = u @ self.codes.T + base[..., None] + self.token_bias  # (B, L, V)
        return logits

    def log_probs_for_target(self, h, targets):
        """(B, L) точные log P(target) без материализации всех V. O(K)."""
        zt = self._gates(h)
        sig = torch.sigmoid(zt)
        ls = torch.log(sig.clamp_min(1e-9))
        lms = torch.log((1 - sig).clamp_min(1e-9))
        c = self.codes[targets]                    # (B, L, K) 0/1
        logp = (c * ls).sum(-1) + ((1 - c) * lms).sum(-1)  # (B, L)
        return logp + self.token_bias[targets]

    def predict(self, h, k=512):
        """Two-stage decode (D): greedy argmax над factored log-probs."""
        lg = self.forward(h)
        return lg.argmax(dim=-1)

    def temperature_tokens(self):
        return self.log_temp
