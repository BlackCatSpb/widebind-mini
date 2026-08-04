"""CognitiveCodedHead: factored-Bernoulli head with six principled modifiers.

Math
----
For each bit k with pre-sigmoid logit z_k the factored Bernoulli assigns
    P(v) = prod_k sigma(z_k)^{c_vk} * (1 - sigma(z_k))^{1 - c_vk}.
Since logit(sigma(z)) = z the log-likelihood of token v equals
    z . c_v  +  base,            base := sum_k log(1 - sigma(z_k)).
No softmax is applied over the hidden/logit space; all six cognitive modifiers
act on z BEFORE the sigmoid.  The only optional softmax is a token-level
NORMALIZATION over the codes (`normalize`), which is the unique honest fix for
calibration:

    sparse_block_codes(V, K, S) picks V tokens out of C(K,S) bit-patterns.
    An unnre normalized factored Bernoulli puts comparable mass on every
    S-active pattern, so ~C(K,S)-V valid-mass leaks off the vocabulary.
    With K=32, S=6 the support is C(32,6)=906192: an unnormalized head starts
    ~log(906192/65536)=2.6 nats above softmax and cannot close that gap
    without negative examples.  normalize=True divides by the code-space
    partition; still cheap because the scores live in K=32 dims.

The six degrees of freedom
    1 temperature   log_temp_base + w_res*err - w_stab*stab   -> T multiplies z
    2 coded prior   bit_bias (from code marginals) + memory-attention offset
    3 social vote    dominance / contra-graph -> additive bounded offset
    4 resonance      embedding-basis mismatch -> soft gate on z
    5 context mod    learned per-bit code modulation before the sigmoid
    6 dynamic cues   per-token shift + per-token bias (learned token priors)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vsa_utils import sparse_block_codes


def _is_tensor(x):
    return isinstance(x, torch.Tensor)


class CognitiveCodedHead(nn.Module):
    def __init__(self, cfg, embed_basis=None, k_mirror=32):
        super().__init__()
        codes = sparse_block_codes(cfg.vocab, K=cfg.code_dim, S=cfg.code_sparsity)
        self.K = codes.shape[1]
        self.S = cfg.code_sparsity
        self.d = cfg.D // self.K
        self.vocab = cfg.vocab
        self.normalize = bool(getattr(cfg, 'head_normalize', True))
        self._k_mirror = k_mirror
        self.register_buffer('codes', codes)            # (V, K) float 0/1

        # readout: tied embedding basis (K,d) or a free K-by-d projector
        if embed_basis is not None:
            self.readout = embed_basis
            self.tie_readout = True
        else:
            self.readout = nn.Parameter(torch.randn(self.K, self.d))
            nn.init.xavier_uniform_(self.readout, gain=0.5)
            self.tie_readout = False

        # 1. adaptive temperature (per-bit inverse temperature)
        self.log_temp_base = nn.Parameter(torch.zeros(self.K))
        self.w_res = nn.Parameter(torch.tensor(0.5))
        self.w_stab = nn.Parameter(torch.tensor(0.1))

        # 2. contextual prior from private memory
        prop = codes.float().mean(dim=0).clamp(1e-7, 1 - 1e-7)
        self.bit_bias = nn.Parameter(torch.log(prop / (1 - prop)))   # logit(prop)
        self.W_q_prior = nn.Parameter(torch.randn(self.d, 1) * 0.01)
        self.W_k_prior = nn.Parameter(torch.randn(k_mirror, 1) * 0.01)
        self.alpha_prior = nn.Parameter(torch.tensor(0.2))
        self.w_prior_scale = nn.Parameter(torch.ones(1))

        # 3. socially-weighted voting (additive bounded offset, NOT a shrink)
        self.beta_social = nn.Parameter(torch.tensor(0.1))

        # 4. resonance against the embedding basis (gate on signal mismatch)
        self.w_energy = nn.Parameter(torch.tensor(0.1))
        self.resonance_floor = 0.5

        # 5. context modulation (pre-sigmoid)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.W_code_mod = nn.Parameter(torch.randn(self.K, self.d, 1) * 0.01)

        # 6. dynamic per-token shift + per-token prior
        self.token_shift_embed = nn.Embedding(cfg.vocab, 8)
        self.proj_shift = nn.Linear(8, self.K, bias=False)
        nn.init.normal_(self.token_shift_embed.weight, std=0.01)
        self.token_bias = nn.Parameter(torch.zeros(cfg.vocab))

        # cognitive state (absent -> neutral defaults)
        self._pred_error = None
        self._private_mem = None
        self._trust_matrix = None
        self._contra_graph = None
        self._dominance = None

    # -- cognitive state hook (optional, cross-layer aggregation) -----------
    def set_cognitive_state(self, pred_error=None, private_mem=None,
                            trust_matrix=None, contra_graph=None,
                            dominance=None):
        """Contract: pred_error (K,); private_mem (K,k_mirror);
        trust/contra (K,K); dominance (K,)."""
        self._pred_error = pred_error
        self._private_mem = private_mem
        self._trust_matrix = trust_matrix
        self._contra_graph = contra_graph
        self._dominance = dominance

    # -----------------------------------------------------------------------
    def _compute_z(self, h, B, L, device):
        """Pre-sigmoid logit z (B,L,K) after the six modifiers; plus base."""
        h_g = h.reshape(B, L, self.K, self.d)                    # (B,L,K,d)
        z_raw = torch.einsum('blkd,kd->blk', h_g, self.readout)  # (B,L,K)

        # 1. temperature ---------------------------------------------------------
        if self._pred_error is not None:
            pe = self._pred_error.float()
            e_pred = pe.mean(dim=(0, 1)) if pe.ndim > 1 else pe
        else:
            e_pred = torch.zeros(self.K, device=device, dtype=h.dtype)
        if self._private_mem is not None and self._private_mem.shape[1] == self._k_mirror:
            stab = self._private_mem.float().var(dim=1)          # (K,)
        else:
            stab = torch.zeros(self.K, device=device, dtype=h.dtype)
        tau = self.log_temp_base + self.w_res * e_pred - self.w_stab * stab
        T = torch.exp(tau).clamp(0.3, 5.0)                        # (K,)

        # 2. coded prior ---------------------------------------------------------
        if self._private_mem is not None and self._private_mem.shape[1] == self._k_mirror:
            pm = self._private_mem.float()
        else:
            pm = torch.zeros(self.K, self._k_mirror, device=device, dtype=h.dtype)
        key = pm @ self.W_k_prior                                # (K,1)
        query = torch.einsum('blkd,dq->blk', h_g, self.W_q_prior).squeeze(-1)  # (B,L,K)
        attn = torch.einsum('blk,km->blm', query, key).squeeze(-1)   # (B,L)  [FIXED einsum]
        prior = self.bit_bias + self.alpha_prior * torch.tanh(
            attn.unsqueeze(-1) * self.w_prior_scale)             # (B,L,K)

        # 3. social offset (additive, bounded) -----------------------------------
        if self._dominance is not None:
            dom = self._dominance.float()
        else:
            dom = torch.ones(self.K, device=device, dtype=h.dtype)
        if self._contra_graph is not None:
            contra_avg = self._contra_graph.float().mean(dim=1)  # (K,)
        else:
            contra_avg = torch.zeros(self.K, device=device, dtype=h.dtype)
        social_bias = torch.tanh(self.beta_social * (dom - contra_avg))  # (K,)

        # 4. resonance: gate stronger when signal matches the embedding basis --
        if self.tie_readout and _is_tensor(self.readout) and self.readout.ndim == 2 \
                and self.readout.shape[-1] == self.d:
            wb = self.readout.detach()
            energy = ((h_g - wb.unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=-1)  # (B,L,K)
            res = (1.0 + self.w_energy * torch.tanh(-energy)).clamp(self.resonance_floor, 2.0)
        else:
            res = 1.0

        # 5. context modulation (pre-sigmoid)
        ctx = torch.tanh(torch.einsum('blkd,kdm->blk', h_g, self.W_code_mod))  # (B,L,K)

        # apply: temperature, resonance, context, then additive priors/social
        z = z_raw * res
        z = z / T.unsqueeze(0).unsqueeze(0)
        z = z * (1.0 + self.gamma * ctx)
        z = z + prior + social_bias.unsqueeze(0).unsqueeze(0)

        base = F.logsigmoid(-z).sum(dim=-1)                      # (B,L) sum log(1-σ(z))
        return z, base

    # per-token dynamic cues ----------------------------------------------------
    def _shift_all(self):
        delta = self.proj_shift(self.token_shift_embed.weight)   # (V,K)
        return (self.codes * delta).sum(dim=1)                   # (V,)

    def _shift_targets(self, token_ids):
        delta = self.proj_shift(self.token_shift_embed(token_ids))  # (B,L,K)
        c = self.codes[token_ids]                                # (B,L,K)
        return (c * delta).sum(dim=-1)                           # (B,L)

    # public API ---------------------------------------------------------
    def forward(self, h):
        """Log-probabilities over all tokens (B,L,V), calibrated if normalize."""
        B, L, _ = h.shape
        z, base = self._compute_z(h, B, L, h.device)
        raw = z @ self.codes.T + base.unsqueeze(-1) + self.token_bias + self._shift_all()
        if self.normalize:
            raw = raw - raw.logsumexp(dim=-1, keepdim=True)
        return raw

    def log_probs_for_target(self, h, targets):
        """Log P(target) (B,L). O(K·V) when normalizing (K-dim scores are cheap)."""
        B, L, _ = h.shape
        z, base = self._compute_z(h, B, L, h.device)
        c = self.codes[targets]                                  # (B,L,K)
        score = (c * z).sum(dim=-1) + base + self.token_bias[targets] + self._shift_targets(targets)
        if not self.normalize:
            return score
        raw = z @ self.codes.T + base.unsqueeze(-1) + self.token_bias + self._shift_all()
        logZ = raw.logsumexp(dim=-1)                             # (B,L)
        return score - logZ

    def compute_loss(self, h, targets):
        return -self.log_probs_for_target(h, targets).mean()

    def predict(self, h, k=1):
        return self.forward(h).argmax(dim=-1)

    def sample(self, h, temperature=1.0):
        logits = self.forward(h)
        return torch.multinomial(F.softmax(logits / temperature, dim=-1)
                                 .reshape(-1, self.vocab), 1).reshape(h.size(0), h.size(1))