"""
Collective Concept Layer — prototype for WideBand Mini.

Concepts live in their OWN slot bank (S shared slots in K-space), separate from
private_mem. Experts READ from the bank only when the main-layer signal is
insufficient (uncertainty gate) AND the concept is consistent with the current
context (contradiction gate). The contradiction gate is the "pink elephant"
guard: a gap in the data ("elephant has no color") must NOT become a confident
"pink elephant" — a born-but-unconfirmed concept stays silent until evidence
supports it.

Mechanism:
  WRITE (no_grad, gated by layer maturity):
    - model's OWN quality signal: mirror residual-var EMA. When it settles below
      a fraction of its initial value, the layer is "mature" -> mining on.
    - confident + novel tokens (low pred_error, gap from all slots) refine/seed slots.
  READ (differentiable through W_o, gates detached):
    - relevance  a_s     = softmax(cos(shared_hp, m_s))         (which concepts)
    - occupancy  w(U_s)  = normalized slot usage EMA            (born? confirmed?)
    - uncertainty gate   = sigmoid(kappa*(pred_error - theta))  (main signal weak?)
    - contradiction gate = sigmoid(gain*(cos(readout, context) - thresh))
      (concept must align with current context or it is suppressed)
    - scale = sigmoid(learnable) — the model learns how much to trust concepts.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CollectiveConceptLayer(nn.Module):
    def __init__(self, D, k, S=8, write_delay=5000,
                 uncert_theta=0.5, uncert_kappa=3.0,
                 contra_thresh=-0.1, contra_gain=6.0,
                 birth_gap=0.55, maturity_frac=0.85, seed=None,
                 cfg=None):
        super().__init__()
        self.cfg = cfg
        self.D = D
        self.k = k
        self.S = S
        self._write_delay = write_delay
        self._uncert_theta = uncert_theta
        self._uncert_kappa = uncert_kappa
        self._contra_thresh = contra_thresh
        self._contra_gain = contra_gain
        self._birth_gap = birth_gap
        self._maturity_frac = maturity_frac

        g = torch.Generator().manual_seed(seed) if seed is not None else None
        m_init = torch.randn(S, k, generator=g)
        self.register_buffer('M', F.normalize(m_init, dim=-1))
        self.register_buffer('U_s', torch.zeros(S))
        self.register_buffer('N_s', torch.zeros(S, dtype=torch.long))
        self.register_buffer('_step', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_resvar_ref', torch.zeros(1))
        self.register_buffer('_mature', torch.zeros(1))
        self.register_buffer('_gate_u', torch.zeros(1))
        self.register_buffer('_gate_c', torch.zeros(1))
        self._maturity_frac = maturity_frac
        self._maturity_warmup = 0

        # readout: S*k -> D  (design: W_o: R^{S*k}->R^D)
        self.W_o = nn.Linear(S * k, D, bias=False)
        nn.init.orthogonal_(self.W_o.weight)
        self._read_scale = nn.Parameter(torch.tensor(0.0))
        self._temp = nn.Parameter(torch.tensor(2.0))

    # ─── diagnostics ───
    def debug(self):
        occ = (self.U_s / (self.U_s.sum() + 1e-8)).tolist()
        counts = self.N_s.tolist()
        return {
            'mature': self._mature.item(),
            'u_gate': self._gate_u.item(),
            'c_gate': self._gate_c.item(),
            'read_scale': torch.sigmoid(self._read_scale).item(),
            'occupied': int((self.N_s > 0).sum().item()),
            'U_s': [round(o, 3) for o in occ],
            'N_s': counts,
        }

    @torch.no_grad()
    def _update_maturity(self, resvar):
        """Adaptive maturity: based on resvar stabilization, not fixed warmup.

        Maturity when coefficient of variation (CV = std/mean) of resvar
        drops below 1/λ_d for ceil(λ_d) consecutive steps.
        No fixed warmup — adapts to training dynamics."""
        if resvar is None:
            return
        lam = getattr(self.cfg, 'lambda_d', 3) if self.cfg else 3
        lam_inv = 1.0 / lam
        if not hasattr(self, '_resvar_ema'):
            self.register_buffer('_resvar_ema', torch.tensor(resvar))
            self.register_buffer('_resvar_var', torch.tensor(1.0))
            self.register_buffer('_mature_count', torch.zeros(1, dtype=torch.long))
        ema_rate = lam_inv
        delta = resvar - self._resvar_ema.item()
        self._resvar_ema.fill_(self._resvar_ema.item() + ema_rate * delta)
        self._resvar_var.mul_(1 - ema_rate).add_(delta * delta * ema_rate)
        cv = (self._resvar_var.item() ** 0.5) / (abs(self._resvar_ema.item()) + 1e-8)
        stable = cv < lam_inv
        if stable:
            self._mature_count += 1
        else:
            self._mature_count.zero_()
        self._mature.fill_(1.0 if self._mature_count.item() >= math.ceil(lam) else 0.0)

    @torch.no_grad()
    def _maybe_write(self, hp, pen, allow_write):
        """Mature-gated, confident+novel slot refinement and birth."""
        self._step += 1
        if self._step.item() < 3:
            print(f'DEBUG _maybe_write: step={self._step.item()} write_delay={self._write_delay} allow_write={allow_write} mature={self._mature.item():.1f}')
        if self._step.item() < self._write_delay:
            return
        if not allow_write:
            return
        if self._mature.item() < 0.5:
            return

        B, L, G, k = hp.shape
        shared = hp.mean(dim=-2)                      # (B,L,k)
        shared_n = F.normalize(shared, dim=-1)
        M_n = F.normalize(self.M, dim=-1)
        sim = shared_n @ M_n.T                        # (B,L,S)
        best = sim.argmax(dim=-1)
        best_sim = sim.max(dim=-1).values
        d_min = 1.0 - best_sim                        # (B,L)
        conf = torch.sigmoid(-pen)                    # (B,L) low pred_error -> confident
        conf_thresh = conf.median().clamp(min=0.01)
        if self._step.item() < 3:
            print(f'DEBUG collective: step={self._step.item()} pen_mean={pen.mean().item():.3f} conf_mean={conf.mean().item():.3f} conf_thresh={conf_thresh.item():.3f} best_unique={best.unique().tolist()} mask_any={(best == 0).any().item()}')

        # refine nearest slot with confident tokens
        for s in range(self.S):
            mask = (best == s) & (conf >= conf_thresh)
            if self._step.item() < 3:
                print(f'DEBUG L{s}: best={best[:5].tolist()} mask_any={mask.any().item()} conf_thresh={conf_thresh.item():.3f}')
            if mask.any():
                upd = F.normalize(shared[mask].mean(dim=0), dim=-1)
                if self.N_s[s].item() < 10:
                    self.M.data[s] = upd
                else:
                    alpha = 0.01
                    self.M.data[s] = F.normalize(
                        self.M[s] * (1 - alpha) + upd * alpha, dim=-1)
                self.N_s[s] += mask.sum().item()

        # birth: empty slot + confident novel tokens
        empty = torch.nonzero(self.N_s == 0)
        novel = (d_min > self._birth_gap * 0.2) & (conf >= conf_thresh)
        if empty.numel() > 0 and novel.any():
            idx = empty[0].item()
            self.M.data[idx] = F.normalize(shared[novel].mean(dim=0), dim=-1)
            self.N_s[idx] += 1
        elif empty.numel() == 0 and novel.any():
            # eviction: bank full -> least-used slot gets recycled for the novel concept
            evict = int(torch.argmin(self.U_s).item())
            self.M.data[evict] = F.normalize(shared[novel].mean(dim=0), dim=-1)
            self.N_s[evict] = 1
            self.U_s[evict] = 0.0

        # occupancy EMA
        occ = torch.zeros(self.S)
        for s in range(self.S):
            occ[s] = (best == s).float().mean().item()
        self.U_s.mul_(0.99).add_(occ.to(self.U_s), alpha=0.01)

    def forward(self, h, hp, pen, resvar=None, context_mem=None, allow_write=None, mature_override=None):
        if self._step.item() < 3:
            print(f'DEBUG collective.forward: step={self._step.item()} pen={pen.mean().item():.3f} if pen is not None else None')
        """
        h   (B,L,D)   block input state (pre-block RMSNorm output)
        hp  (B,L,G,k) mirror K-states
        pen (B,L)     mirror pred_error norm (model's own uncertainty)
        resvar float  mirror residual-var EMA (maturity signal from the model itself)
        """
        _write = allow_write is None or allow_write
        if mature_override is not None:
            self._mature.fill_(float(mature_override))
        else:
            self._update_maturity(resvar)
        self._maybe_write(hp, pen, _write)

        B, L, G, k = hp.shape
        shared = F.normalize(hp.mean(dim=-2), dim=-1)     # (B,L,k)
        M_n = F.normalize(self.M, dim=-1)
        sim = shared @ M_n.T                              # (B,L,S)
        temp = self._temp.clamp(min=0.5)
        a = torch.softmax(sim * temp, dim=-1)             # (B,L,S)
        occ_w = (self.U_s / (self.U_s.max() + 1e-8)).clamp(0, 1)
        # blend = relevance * occupancy * slot   ->  S*k read vector
        blend = (a.unsqueeze(-1) * occ_w.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                 * M_n.unsqueeze(0).unsqueeze(0))
        read = self.W_o(blend.reshape(B, L, -1))          # (B,L,D)

        with torch.no_grad():
            # uncertainty gate: open when the main-layer signal is weak
            u_gate = torch.sigmoid(self._uncert_kappa * (pen.unsqueeze(-1) - self._uncert_theta))
            # contradiction gate: concept must align with current context
            out_n = F.normalize(read, dim=-1)
            h_n = F.normalize(h.detach(), dim=-1)
            cos_c = (out_n * h_n).sum(dim=-1, keepdim=True)
            c_gate = torch.sigmoid(self._contra_gain * (cos_c - self._contra_thresh))
            self._gate_u.fill_(u_gate.mean().item())
            self._gate_c.fill_(c_gate.mean().item())

        scale = torch.sigmoid(self._read_scale)
        return read * u_gate * c_gate * scale
