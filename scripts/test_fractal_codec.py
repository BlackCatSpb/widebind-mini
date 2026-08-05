"""Synthetic test: baseline codec vs Fibonacci fractal crossings.

Task: N unique tokens, each token has a hierarchical "signature" across K dims.
Baseline: 32 independent codes (current).
Fractal: Fibonacci-cascade crossings (proposed).

Measures: convergence speed, final loss, memory usage.
"""
import sys, os, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.amp_codec import SignedAmpHead, SignedAmpEmbedding, _LOG2PI
from core.config import WideBandConfig


def fib_sequence(k_max):
    """Fibonacci indices < k_max: 1, 2, 3, 5, 8, 13, 21, ..."""
    fibs = [1, 2]
    while fibs[-1] + fibs[-2] < k_max:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs


class FractalFibHead(SignedAmpHead):
    """Codec head with Fibonacci-cascade crossings between code positions.

    Instead of independent a_k = tanh(gain * proj_k(h)), we create
    a tree where each node crosses its projection with two ancestors:
        a_{fib[i]} = z_{fib[i]} ⊛ (a_{fib[i-1]}, a_{fib[i-2]})
    where z_k = tanh(gain * proj_k(h)) and ⊛ is bilinear cross.
    """

    def __init__(self, cfg, embed_basis=None, embed_proto=None):
        super().__init__(cfg, embed_basis=embed_basis, embed_proto=embed_proto)
        self.fib_indices = [f for f in fib_sequence(self.K) if f < self.K]
        self.root_indices = [k for k in range(self.K) if k not in self.fib_indices]
        # Cross-mixing weights per fib node (learned)
        self.cross_w = nn.Parameter(torch.ones(len(self.fib_indices), 2) * 0.5)

    def _amps_fractal(self, h_g):
        """Compute amplitudes with Fibonacci crossings."""
        B, L, K = h_g.shape
        z_raw = self._proj(h_g)
        if getattr(self.cfg, 'amp_pred', False):
            z_raw = torch.einsum('...k,kj->...j', z_raw, self.pred_w)
        T = torch.exp(self.log_gain).clamp(0.1, 4.0)
        z = torch.tanh(z_raw * T)

        a = torch.zeros_like(z)
        done = set()

        # Root nodes: independent
        for k in self.root_indices:
            if k < K:
                a[:, :, k] = z[:, :, k]
                done.add(k)

        # Fibonacci nodes: cross with two ancestors
        for idx, fib_k in enumerate(self.fib_indices):
            if fib_k >= K:
                break
            i1 = idx - 1
            i2 = idx - 2
            a1 = a[:, :, self.fib_indices[i1]] if i1 >= 0 else torch.ones_like(z[:, :, 0])
            a2 = a[:, :, self.fib_indices[i2]] if i2 >= 0 else torch.ones_like(z[:, :, 0])
            w1 = torch.sigmoid(self.cross_w[idx, 0])
            w2 = torch.sigmoid(self.cross_w[idx, 1])
            cross = z[:, :, fib_k] * (w1 * a1 + w2 * a2 * torch.roll(a1, 1, dims=-1))
            a[:, :, fib_k] = torch.tanh(cross)
            done.add(fib_k)

        remaining = [k for k in range(K) if k not in done]
        for k in remaining:
            a[:, :, k] = z[:, :, k]

        return a, z_raw * T

    def margin_loss(self, h, targets, h_emb=None):
        N, D = h.shape
        a, _ = self._amps_fractal(h.reshape(N, self.K, -1))
        pe = None if h_emb is None else self._code(h_emb.reshape(N, self.K, -1))
        son, soff, semb = self._sigmas()
        z = self.codes[targets].float()
        alpha = torch.tanh(self._proto_ref[0][targets]) * z
        on = -0.5 * ((a - alpha) / son) ** 2 - torch.log(son) - 0.5 * _LOG2PI
        off = -0.5 * ((a - self.o) / soff) ** 2 - torch.log(soff) - 0.5 * _LOG2PI
        logp = (z * on + (1 - z) * off).sum(-1) + self._bias()[targets]
        mu, nu, p = self._active_stats()
        e_on = -0.5 * ((a - mu) ** 2 + nu) / son ** 2 - torch.log(son) - 0.5 * _LOG2PI
        e_off = -0.5 * ((a - self.o) / soff) ** 2 - torch.log(soff) - 0.5 * _LOG2PI
        e_q = (p * e_on + (1.0 - p) * e_off).sum(-1)
        if pe is not None:
            code = -0.5 * ((pe - alpha) / semb) ** 2 - torch.log(semb) - 0.5 * _LOG2PI
            logp = logp + (z * code).sum(-1)
        return -(logp - e_q)


def make_synthetic_task(n_tokens, k_code, sparsity, seed=42):
    """Create sparse codes for n_tokens with hierarchical structure."""
    g = torch.Generator().manual_seed(seed)
    from core.vsa_utils import sparse_block_codes
    codes = sparse_block_codes(n_tokens, K=k_code, S=sparsity)
    prot = (torch.rand(n_tokens, k_code, generator=g) - 0.5) * 2 * 0.2
    return codes, prot


def _make_codec_modules(codes, prot, k_code, D, fractal=False):
    """Properly construct embed + head modules."""
    from core.embedding import RotaryEmbedding

    class EmbedMini(nn.Module):
        def __init__(self):
            super().__init__()
            self.K = k_code
            self.register_buffer('codes', codes)
            basis = torch.randn(k_code, D // k_code) * 0.5 / math.sqrt(D // k_code)
            basis = F.normalize(basis, dim=-1)
            self.basis = nn.Parameter(basis)
            self.proto = nn.Parameter(prot.clone())
            self._scale = 1.0
            self.rope = RotaryEmbedding(D, theta=1e6)
        def forward(self, tokens):
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            alpha = torch.tanh(self.proto[tokens]) * self.codes[tokens]
            out = torch.einsum('blk,kd->blkd', alpha, self.basis).reshape(tokens.shape[0], tokens.shape[1], -1)
            return self.rope(out)

    class HeadMiniBase(nn.Module):
        def __init__(self, embed):
            super().__init__()
            self.K = k_code
            self.register_buffer('codes', codes)
            self._basis_ref = [embed.basis]
            self._proto_ref = [embed.proto]
            self.log_gain = nn.Parameter(torch.full((k_code,), math.log(0.5)))
            self.log_gain_emb = nn.Parameter(torch.full((k_code,), math.log(0.5)))
            self.pred_w = nn.Parameter(torch.eye(k_code))
            self.o = nn.Parameter(torch.zeros(k_code))
            self.log_sigma_on = nn.Parameter(torch.full((k_code,), math.log(0.3)))
            self.log_sigma_off = nn.Parameter(torch.full((k_code,), math.log(0.3)))
            self.log_sigma_emb = nn.Parameter(torch.full((k_code,), math.log(0.3)))
            self.token_bias = nn.Parameter(torch.zeros(codes.shape[0]))
            self.cfg = type('C', (), {'amp_pred': True, 'amp_sigma_min': 0.2})()
        def _proj(self, h_g):
            return torch.einsum('...kd,kd->...k', h_g, self._basis_ref[0])
        def _amps(self, h_g):
            z = self._proj(h_g)
            z = torch.einsum('...k,kj->...j', z, self.pred_w)
            T = torch.exp(self.log_gain).clamp(0.1, 4.0)
            return torch.tanh(z * T), z * T
        def _code(self, h_emb_g):
            T = torch.exp(self.log_gain_emb).clamp(0.1, 4.0)
            return self._proj(h_emb_g) * T
        def _sigmas(self):
            return (torch.clamp(F.softplus(self.log_sigma_on) + 0.05, min=0.2, max=1.0),
                    torch.clamp(F.softplus(self.log_sigma_off) + 0.05, min=0.2, max=1.0),
                    torch.clamp(F.softplus(self.log_sigma_emb) + 0.05, min=0.2, max=1.0))
        def _bias(self):
            return torch.clamp(self.token_bias - self.token_bias.mean(), min=-4.0, max=4.0)
        def _alpha_of(self, tokens):
            return torch.tanh(self._proto_ref[0][tokens]) * self.codes[tokens]
        def _active_stats(self):
            alpha = torch.tanh(self._proto_ref[0]) * self.codes
            n = self.codes.sum(0).clamp_min(1)
            mu = (alpha * self.codes).sum(0) / n
            nu = ((alpha * self.codes).square().sum(0) / n - mu.square()).clamp_min(0.0)
            return mu, nu, self.codes.mean(0)
        def margin_loss(self, h, targets, h_emb=None):
            N, D2 = h.shape
            a, _ = self._amps(h.reshape(N, self.K, -1))
            pe = None if h_emb is None else self._code(h_emb.reshape(N, self.K, -1))
            son, soff, semb = self._sigmas()
            z = self.codes[targets].float()
            alpha = torch.tanh(self._proto_ref[0][targets]) * z
            on = -0.5 * ((a - alpha) / son) ** 2 - torch.log(son) - 0.5 * _LOG2PI
            off = -0.5 * ((a - self.o) / soff) ** 2 - torch.log(soff) - 0.5 * _LOG2PI
            logp = (z * on + (1 - z) * off).sum(-1) + self._bias()[targets]
            mu, nu, p = self._active_stats()
            e_on = -0.5 * ((a - mu) ** 2 + nu) / son ** 2 - torch.log(son) - 0.5 * _LOG2PI
            e_off = -0.5 * ((a - self.o) / soff) ** 2 - torch.log(soff) - 0.5 * _LOG2PI
            e_q = (p * e_on + (1.0 - p) * e_off).sum(-1)
            if pe is not None:
                code = -0.5 * ((pe - alpha) / semb) ** 2 - torch.log(semb) - 0.5 * _LOG2PI
                logp = logp + (z * code).sum(-1)
            return -(logp - e_q)

    class HeadMiniFractal(HeadMiniBase):
        def __init__(self, embed):
            super().__init__(embed)
            self.fib_indices = [f for f in fib_sequence(self.K) if f < self.K]
            self.cross_w = nn.Parameter(torch.ones(len(self.fib_indices), 2) * 0.5)

        def _amps(self, h_g):
            z_raw = self._proj(h_g)
            z_raw = torch.einsum('...k,kj->...j', z_raw, self.pred_w)
            T = torch.exp(self.log_gain).clamp(0.1, 4.0)
            z = torch.tanh(z_raw * T)
            K = z.shape[-1]
            done = set()
            root_indices = [k for k in range(K) if k not in self.fib_indices]
            parts = [None] * K
            for k in root_indices:
                parts[k] = z[..., k]
                done.add(k)
            for idx, fib_k in enumerate(self.fib_indices):
                if fib_k >= K:
                    break
                a1 = parts[self.fib_indices[idx - 1]] if idx >= 1 else torch.ones_like(z[..., 0])
                w1 = torch.sigmoid(self.cross_w[idx, 0])
                rolled = torch.cat([a1[..., -1:], a1[..., :-1]], dim=-1)
                cross = z[..., fib_k] * (w1 * a1 + (1 - w1) * rolled)
                parts[fib_k] = torch.tanh(cross)
                done.add(fib_k)
            for k in range(K):
                if k not in done:
                    parts[k] = z[..., k]
            a = torch.stack(parts, dim=-1)
            return a, z_raw * T

    embed = EmbedMini()
    head = HeadMiniFractal(embed) if fractal else HeadMiniBase(embed)
    return embed, head


def train_baseline(data, codes, prot, k_code, steps=200, D=128, lr=1e-3):
    embed, head = _make_codec_modules(codes, prot, k_code, D, fractal=False)
    params = list(embed.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    losses = []
    for step in range(steps):
        logits = torch.randint(0, codes.shape[0], (256,))
        h = embed(logits)
        loss = head.margin_loss(h.reshape(-1, D), logits.reshape(-1), h.reshape(-1, D)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def train_fractal(data, codes, prot, k_code, steps=200, D=128, lr=1e-3):
    embed, head = _make_codec_modules(codes, prot, k_code, D, fractal=True)
    params = list(embed.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    losses = []
    for step in range(steps):
        logits = torch.randint(0, codes.shape[0], (256,))
        h = embed(logits)
        loss = head.margin_loss(h.reshape(-1, D), logits.reshape(-1), h.reshape(-1, D)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def make_hierarchical_task(n_tokens, k_code, n_categories, n_per_cat, seed=42):
    """Create codes with hierarchical structure: category -> subcategory -> token.
    The fractal tree should exploit this hierarchy."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.zeros(n_tokens, k_code)
    prot = torch.zeros(n_tokens, k_code)

    # First n_categories bits encode category (one-hot)
    # Next level encodes subcategory within category
    # Remaining bits encode individual token
    cat_bits = min(n_categories, k_code // 3)
    subcat_bits = min(n_categories * 2, k_code // 3)

    idx = 0
    for cat in range(n_categories):
        for sub in range(n_per_cat):
            if idx >= n_tokens:
                break
            # Category bits (shared across category)
            codes[idx, cat % cat_bits] = 1.0
            # Subcategory bits
            sub_bit = cat_bits + (sub + cat * n_per_cat) % subcat_bits
            if sub_bit < k_code:
                codes[idx, sub_bit] = 1.0
            # Token-specific bits
            token_sparsity = max(1, 6 - 2)  # 4 individual bits
            n_free = max(0, k_code - cat_bits - subcat_bits)
            if n_free > 0:
                n_individual = min(token_sparsity, n_free)
                for b in range(n_individual):
                    bit_idx = cat_bits + subcat_bits + ((idx * 7 + b * 13) % n_free)
                    if bit_idx < k_code:
                        codes[idx, bit_idx] = 1.0
            idx += 1

    prot = codes * 0.8 + (torch.rand(n_tokens, k_code, generator=g) - 0.5) * 0.05
    return codes, prot


def train_baseline_limited(data, codes, prot, k_code, steps, D, lr, train_tokens):
    """Train on subset, return loss on ALL tokens."""
    embed, head = _make_codec_modules(codes, prot, k_code, D, fractal=False)
    params = list(embed.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    for step in range(steps):
        logits = train_tokens[torch.randint(0, len(train_tokens), (256,))]
        h = embed(logits)
        loss = head.margin_loss(h.reshape(-1, D), logits.reshape(-1), h.reshape(-1, D)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Eval on ALL tokens
    with torch.no_grad():
        all_logits = torch.arange(codes.shape[0])
        h = embed(all_logits)
        all_loss = head.margin_loss(h.reshape(-1, D), all_logits.reshape(-1), h.reshape(-1, D)).mean()
    return all_loss.item()


def train_fractal_limited(data, codes, prot, k_code, steps, D, lr, train_tokens):
    embed, head = _make_codec_modules(codes, prot, k_code, D, fractal=True)
    params = list(embed.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    for step in range(steps):
        logits = train_tokens[torch.randint(0, len(train_tokens), (256,))]
        h = embed(logits)
        loss = head.margin_loss(h.reshape(-1, D), logits.reshape(-1), h.reshape(-1, D)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        all_logits = torch.arange(codes.shape[0])
        h = embed(all_logits)
        all_loss = head.margin_loss(h.reshape(-1, D), all_logits.reshape(-1), h.reshape(-1, D)).mean()
    return all_loss.item()


def run_comparison():
    K = 32
    SPARSITY = 6
    N_TOKENS = 512
    N_CATEGORIES = 8
    N_PER_CAT = 64
    D = 128
    STEPS = 100
    LR = 3e-3
    TRAIN_FRAC = 0.5

    print(f"=== Fractal Fibonacci Codec Test (Generalization) ===")
    print(f"K={K}, S={SPARSITY}, V={N_TOKENS}, D={D}, steps={STEPS}")
    print(f"Train on {TRAIN_FRAC*100:.0f}%, eval on 100%")
    print(f"Fibonacci indices (<{K}): {fib_sequence(K)}")
    print()

    codes, prot = make_hierarchical_task(N_TOKENS, K, N_CATEGORIES, N_PER_CAT)

    n_train = int(N_TOKENS * TRAIN_FRAC)
    train_tokens = torch.randperm(N_TOKENS)[:n_train]

    t0 = time.time()
    eval_base = train_baseline_limited(None, codes, prot, K, STEPS, D, LR, train_tokens)
    t_base = time.time() - t0

    t0 = time.time()
    eval_fractal = train_fractal_limited(None, codes, prot, K, STEPS, D, LR, train_tokens)
    t_fractal = time.time() - t0

    print(f"Baseline:  eval_loss={eval_base:.4f}  time={t_base:.2f}s")
    print(f"Fractal:   eval_loss={eval_fractal:.4f}  time={t_fractal:.2f}s")
    winner = 'FRACTAL' if eval_fractal < eval_base else 'BASELINE'
    print(f"Winner: {winner} (delta={eval_fractal-eval_base:+.4f})")

    return eval_base, eval_fractal


if __name__ == '__main__':
    run_comparison()
