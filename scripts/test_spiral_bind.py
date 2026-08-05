"""SpiralBind test: complex cross-mixing vs shift cross-mixing.

SpiralBind:
  u_s = hp * (w_u_re_s + i * w_u_im_s)   # complex
  v_s = hp * (w_v_re_s + i * w_v_im_s)   # complex
  v_rot = v_s * exp(i * theta_s)          # phase rotation
  prod_s = u_s * v_rot_s                  # complex multiply
  out = [Re(prod); Im(prod)] @ W_out      # single projection

Key claim: 4× expressiveness (rank 128 vs 32) at same FLOPS.
"""
import sys, os, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class SpiralBind(nn.Module):
    """Complex cross-mixing with phase rotations. Multi-ocular variant."""

    def __init__(self, D, K, S=4):
        super().__init__()
        self.D, self.K, self.S = D, K, S
        self.W_proj = nn.Linear(D, K, bias=True)
        self.hp_norm = nn.RMSNorm(K)

        # Complex weights: 4 real params per (spiral, channel)
        self.w_u_re = nn.Parameter(torch.randn(S, K) * 0.3)
        self.w_u_im = nn.Parameter(torch.zeros(S, K))
        self.w_v_re = nn.Parameter(torch.randn(S, K) * 0.3)
        self.w_v_im = nn.Parameter(torch.zeros(S, K))

        # Per-channel time constants (learned, scale-independent)
        self.tau = nn.Parameter(torch.rand(K) * 0.5 + 0.1)

        # Per-spiral output projections: S × 2K × D
        self.W_out = nn.Parameter(torch.empty(S, 2 * K, D))
        nn.init.xavier_uniform_(self.W_out, gain=0.5)

    def forward(self, h):
        hp = self.hp_norm(self.W_proj(h))
        B, L, K = hp.shape
        tau_norm = (self.tau / (self.tau.max() + 1e-8)).clamp(0.01, 1.0)

        out = None
        for s in range(self.S):
            c_s = (1 + math.sqrt(5)) / 2 * (s + 1)
            theta = 2 * math.pi * c_s * (tau_norm ** 0.5)
            cos_t = torch.cos(theta).unsqueeze(0).unsqueeze(0)
            sin_t = torch.sin(theta).unsqueeze(0).unsqueeze(0)

            u_re = hp * self.w_u_re[s]
            u_im = hp * self.w_u_im[s]
            v_re = hp * self.w_v_re[s]
            v_im = hp * self.w_v_im[s]

            vr_re = v_re * cos_t - v_im * sin_t
            vr_im = v_re * sin_t + v_im * cos_t
            del cos_t, sin_t

            prod_re = u_re * vr_re - u_im * vr_im
            prod_im = u_re * vr_im + u_im * vr_re

            out_s = torch.cat([prod_re, prod_im], dim=-1)
            term = out_s @ self.W_out[s]
            out = term if out is None else out + term

        return out

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class InterferenceBind(nn.Module):
    """Multi-scale interference mixing via learned wave frequencies.

    For each spiral s:
      phase = hp @ W_phase_s  (learned phase offsets)
      freq  = exp(W_freq_s)    (learned frequencies, per channel)
      wave_s = sin(freq * hp + phase) * amplitude_s
    Cross-mixing: wave_u ⊙ wave_v (interference)
    Output: Σ_s (interference_s @ W_out_s)

    This creates rich non-linear interactions using only real ops.
    Frequencies follow geometric progression (like Fourier but learned).
    """

    def __init__(self, D, K, S=4):
        super().__init__()
        self.D, self.K, self.S = D, K, S
        self.W_proj = nn.Linear(D, K, bias=True)
        self.hp_norm = nn.RMSNorm(K)

        # Per-spiral: phase proj + freq + amplitude
        self.W_phase = nn.Parameter(torch.randn(S, K) * 0.1)
        freq_init = torch.log(torch.arange(1, K + 1, dtype=torch.float32) / K).unsqueeze(0).expand(S, -1).clone()
        self.W_freq = nn.Parameter(freq_init)
        self.W_amp = nn.Parameter(torch.ones(S, K))

        # Per-channel time constants (from VSA integration)
        self.tau = nn.Parameter(torch.rand(K) * 0.5 + 0.1)

        # Per-spiral output projections
        self.W_out = nn.Parameter(torch.empty(S, K, D))
        nn.init.xavier_uniform_(self.W_out, gain=0.5)

    def forward(self, h):
        hp = self.hp_norm(self.W_proj(h))
        tau_norm = (self.tau / (self.tau.max() + 1e-8)).clamp(0.01, 1.0)

        out = None
        for s in range(self.S):
            freq = torch.exp(self.W_freq[s]).unsqueeze(0).unsqueeze(0)
            phase = self.W_phase[s].unsqueeze(0).unsqueeze(0)
            amp = self.W_amp[s].unsqueeze(0).unsqueeze(0)

            theta = freq * hp + phase + tau_norm.unsqueeze(0).unsqueeze(0) * math.pi
            wave = amp * torch.sin(theta)

            # Interference: self-mixing (u=v=wave creates harmonics)
            wave_rolled = torch.roll(wave, 1, dims=-1).contiguous()
            interference = wave * wave_rolled * torch.cos(theta * 0.5)

            term = interference @ self.W_out[s]
            out = term if out is None else out + term

        return out

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class DeepInterferenceBind(nn.Module):
    """Interference with multi-scale frequency stacks + non-linear mixing.

    Key improvement: multiple frequency bands per spiral (like wavelet),
    and multiplicative mixing between bands (like FM synthesis).
    """

    def __init__(self, D, K, S=4, n_bands=3):
        super().__init__()
        self.D, self.K, self.S, self.n_bands = D, K, S, n_bands
        self.W_proj = nn.Linear(D, K, bias=True)
        self.hp_norm = nn.RMSNorm(K)

        # Multiple frequency bands per spiral
        self.W_freq = nn.Parameter(torch.randn(S, n_bands, K) * 0.1)
        self.W_phase = nn.Parameter(torch.randn(S, n_bands, K) * 0.1)
        self.W_amp = nn.Parameter(torch.ones(S, n_bands, K))

        # Cross-band mixing weights
        mix_init = torch.stack([torch.eye(n_bands) for _ in range(S)])
        self.W_mix = nn.Parameter(mix_init)

        # Per-channel time constants
        self.tau = nn.Parameter(torch.rand(K) * 0.5 + 0.1)

        # Per-spiral output projections
        self.W_out = nn.Parameter(torch.empty(S, K, D))
        nn.init.xavier_uniform_(self.W_out, gain=0.5)

    def forward(self, h):
        hp = self.hp_norm(self.W_proj(h))
        tau_norm = (self.tau / (self.tau.max() + 1e-8)).clamp(0.01, 1.0)

        out = None
        for s in range(self.S):
            waves = []
            theta_last = None
            for b in range(self.n_bands):
                freq = torch.exp(self.W_freq[s, b]).unsqueeze(0).unsqueeze(0)
                phase = self.W_phase[s, b].unsqueeze(0).unsqueeze(0)
                amp = self.W_amp[s, b].unsqueeze(0).unsqueeze(0)

                theta = freq * hp + phase + tau_norm.unsqueeze(0).unsqueeze(0) * math.pi
                theta_last = theta
                waves.append(amp * torch.sin(theta))

            # Cross-band mixing: each band gets weighted sum of all bands
            waves_stacked = torch.stack(waves, dim=0)
            mix_w = F.softmax(self.W_mix[s], dim=-1)
            mixed_all = torch.einsum('ij,jbkl->ibkl', mix_w, waves_stacked)
            mixed = mixed_all.mean(dim=0)

            # Interference: channel-wise shift + multiply
            mixed_rolled = torch.roll(mixed, 1, dims=-1).contiguous()
            interference = mixed * mixed_rolled * torch.cos(theta_last * 0.5)

            term = interference @ self.W_out[s]
            out = term if out is None else out + term

        return out

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class ShiftBind(nn.Module):
    """Current shift multi-ocular for comparison."""

    def __init__(self, D, K, S=4):
        super().__init__()
        self.D, self.K, self.S = D, K, S
        self.W_proj = nn.Linear(D, K, bias=True)
        self.hp_norm = nn.RMSNorm(K)

        shifts = self._golden_shifts(K, S)
        self.register_buffer("shifts", torch.tensor(shifts, dtype=torch.long))

        self.w_u = nn.Parameter(torch.randn(S, K) * 0.3)
        self.w_v = nn.Parameter(torch.randn(S, K) * 0.3)
        self.W_out = nn.Parameter(torch.empty(S, K, D))
        nn.init.xavier_uniform_(self.W_out, gain=0.5)

    def _golden_shifts(self, K, S):
        phi = (1 + 5 ** 0.5) / 2
        shifts, used = [], set()
        s = 1
        while len(shifts) < S:
            sh = int(math.floor(s * K / phi)) % K
            while sh in used or sh == 0:
                sh = (sh + 1) % K
            shifts.append(sh)
            used.add(sh)
            s += 1
        return shifts

    def forward(self, h):
        hp = self.hp_norm(self.W_proj(h))
        out = None
        for s in range(self.S):
            prod = (hp * self.w_u[s]) * torch.roll(hp * self.w_v[s], int(self.shifts[s]), dims=-1)
            term = prod @ self.W_out[s]
            out = term if out is None else out + term
        return out

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def test_rank():
    """Compare effective rank of the cross-mixing matrices."""
    D, K, S = 128, 64, 4
    torch.manual_seed(42)

    binds = {
        "spiral": SpiralBind(D, K, S),
        "interference": InterferenceBind(D, K, S),
        "deep_interference": DeepInterferenceBind(D, K, S),
        "shift": ShiftBind(D, K, S),
    }

    print(f"=== Rank Analysis (K={K}, S={S}) ===")
    h_batch = torch.randn(100, 64, D)
    ranks = {}
    for name, bind in binds.items():
        with torch.no_grad():
            out = bind(h_batch).reshape(100 * 64, D)
        rank = torch.linalg.matrix_rank(out.float()).item()
        ranks[name] = rank
        print(f"{name:14s}: rank={rank:3d}  params={bind.param_count()/1e3:.1f}K  shape={tuple(bind(h_batch).shape)}")
    print()


def test_generalization():
    """Train both on a subset, eval on all tokens."""
    D, K, S = 128, 64, 4
    N_TOKENS = 512
    TRAIN_FRAC = 0.5
    STEPS = 200
    LR = 3e-3

    torch.manual_seed(42)

    codes = torch.zeros(N_TOKENS, K)
    for v in range(N_TOKENS):
        active = torch.randperm(K)[:6]
        codes[v, active] = 1.0
    prot = codes * 0.8 + (torch.rand(N_TOKENS, K) - 0.5) * 0.05

    class CodeBook(nn.Module):
        def __init__(self):
            super().__init__()
            self.basis = nn.Parameter(F.normalize(torch.randn(K, D // K), dim=-1))
            self.proto = nn.Parameter(prot.clone())
        def forward(self, tokens):
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            alpha = torch.tanh(self.proto[tokens]) * codes[tokens]
            out = torch.einsum('blk,kd->blkd', alpha, self.basis).reshape(tokens.shape[0], tokens.shape[1], -1)
            return out

    n_train = int(N_TRAIN := N_TOKENS * TRAIN_FRAC)
    train_idx = torch.randperm(N_TOKENS)[:n_train]

    results = {}
    for name, BindClass in [("interference", InterferenceBind), ("deep_interference", DeepInterferenceBind), ("shift", ShiftBind)]:
        book = CodeBook()
        bind = BindClass(D, K, S)
        opt = torch.optim.Adam(list(book.parameters()) + list(bind.parameters()), lr=LR)

        for step in range(STEPS):
            idx = train_idx[torch.randint(0, n_train, (64,))]
            h = book(idx).reshape(-1, D)
            bind_out = bind(h.unsqueeze(0)).squeeze(0)
            loss = F.mse_loss(bind_out, h) + 0.001 * bind.W_proj.weight.norm()
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            all_idx = torch.arange(N_TOKENS)
            h_all = book(all_idx).reshape(-1, D)
            bind_out = bind(h_all.unsqueeze(0)).squeeze(0)
            error = F.mse_loss(bind_out, h_all)
            results[name] = error.item()

    print(f"=== Generalization (train {TRAIN_FRAC*100:.0f}%, eval 100%) ===")
    for name, err in results.items():
        print(f"{name:14s}: recon error = {err:.4f}")
    winner = min(results, key=results.get)
    print(f"Winner: {winner}")
    print()


def test_flops():
    """Measure actual FLOPS for both."""
    import time
    D, K, S = 448, 64, 4

    binds = {
        "spiral": SpiralBind(D, K, S),
        "interference": InterferenceBind(D, K, S),
        "deep_interference": DeepInterferenceBind(D, K, S),
        "shift": ShiftBind(D, K, S),
    }
    if torch.cuda.is_available():
        binds = {k: v.cuda() for k, v in binds.items()}
    h = torch.randn(4, 128, D).cuda() if torch.cuda.is_available() else torch.randn(4, 128, D)

    for _ in range(10):
        for b in binds.values():
            b(h)

    n_runs = 100
    times = {}
    for name, bind in binds.items():
        t0 = time.time()
        for _ in range(n_runs):
            bind(h)
        times[name] = (time.time() - t0) / n_runs

    print(f"=== Speed (batch=4, seq=128, D={D}, K={K}) ===")
    for name, bind in binds.items():
        print(f"{name:14s}: {times[name]*1000:.2f} ms  params={bind.param_count()/1e3:.1f}K")
    print()


if __name__ == '__main__':
    test_rank()
    test_generalization()
    test_flops()
