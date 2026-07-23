import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─── Expert Sandbox ───────────────────────────────────────────────────────
# Each expert has a learnable behavior direction + magnitude.  The "sandbox"
# is just a container for these; the layer handles cross-step coherence.

class ExpertSandbox(nn.Module):
    """Holds per-expert learnable magnitude."""
    def __init__(self, G, cfg):
        super().__init__()
        self.G = G
        self.log_mag = nn.Parameter(torch.full((G,), math.log(0.3)))

    def compute(self, behavior):
        """Scale behavior into a correction."""
        return behavior * self.log_mag.exp().unsqueeze(-1)


# ─── Arbiter ──────────────────────────────────────────────────────────────
# Scores each expert on three axes and produces maturity ∈ [0, 1].

class Arbiter(nn.Module):
    def __init__(self, G, cfg):
        super().__init__()
        self.G = G
        self.log_thresh = nn.Parameter(torch.tensor([
            cfg.coh_threshold_init,
            cfg.div_threshold_init,
            cfg.act_threshold_init,
        ]))
        self.register_buffer('_maturity', torch.zeros(G))
        self._ema = cfg.maturity_ema

    def score(self, coherence, diversity, activity):
        thresh = torch.exp(self.log_thresh).clamp(min=1e-4, max=100.0)
        t_c, t_d, t_a = thresh[0], thresh[1], thresh[2]
        with torch.no_grad():
            coh_score = torch.sigmoid(-(coherence - t_c) / t_c.clamp(min=0.1))
            div_score = diversity / (diversity + t_d + 1e-10)
            act_score = activity / (activity + t_a + 1e-10)
        return coh_score * div_score * act_score

    def update_maturity(self, coherence, diversity, activity):
        new = self.score(coherence, diversity, activity)
        self._maturity = self._ema * self._maturity + (1 - self._ema) * new
        return self._maturity.clone()


# ─── Layer with Sandbox ──────────────────────────────────────────────────

class SandboxLayer(nn.Module):
    """One layer with G experts, input-dependent routing via matured gates.

    Each expert learns a behavior vector.  The expert's contribution to
    each token is gated by affinity(h, behavior), weighted by maturity
    (from the arbiter).  Mature experts with high coherence + diversity
    + activity get to contribute more.
    """
    def __init__(self, D, G, layer_idx, cfg):
        super().__init__()
        self.D = D
        self.G = G
        self.layer_idx = layer_idx

        self.sandbox = ExpertSandbox(G, cfg)
        self.arbiter = Arbiter(G, cfg)

        # Gate projection: hidden -> expert logits
        self.gate_proj = nn.Linear(D, G, bias=False)

        # Learnable behavior:  per-expert vectors  (G, D)
        self._behavior = nn.Parameter(torch.empty(G, D))
        self.register_buffer('_behavior_old', torch.empty(G, D))
        self.reset_behavior()

        # Smoothing
        self.register_buffer('_coh_ema', torch.zeros(G))
        self._ema = 0.99

        nn.init.orthogonal_(self.gate_proj.weight, gain=0.5)

    def reset_behavior(self, gain=0.5):
        with torch.no_grad():
            b = torch.empty_like(self._behavior)
            nn.init.orthogonal_(b, gain=gain)
            self._behavior.data.copy_(b)
            self._behavior_old.data.copy_(b)

    def forward(self, h):
        """h: (B, L, D)  →  (h_out, corrections, losses_dict)."""
        B, L, D = h.shape
        device = h.device

        old_bh = self._behavior_old.clone()                         # (G, D)
        cur_bh = self._behavior                                     # (G, D)
        mag = self.sandbox.log_mag.exp()                            # (G,)

        # ── Token-expert affinity → routed correction ───────────────
        # Gate projection: hidden -> expert activation logits
        aff = self.gate_proj(h) / math.sqrt(D)                     # (B, L, G)
        # Maturity-gated softmax
        maturity = self.arbiter._maturity                           # (G,)
        gate = F.softmax(aff * maturity.sigmoid().unsqueeze(0).unsqueeze(0), dim=-1)
        # Weighted sum of expert corrections
        corr = cur_bh * mag.unsqueeze(-1)                           # (G, D)
        h_out = h + torch.matmul(gate, corr)                        # (B, L, D)
        # Per-expert correction for diagnostic
        corr_all = corr.unsqueeze(0).unsqueeze(0).expand(B, L, self.G, D)

        # ── Cross-step coherence ─────────────────────────────────────
        cos = F.cosine_similarity(old_bh, cur_bh, dim=-1)
        coh_loss = 1.0 - cos

        # ── Diversity ───────────────────────────────────────────────
        div_loss = torch.tensor(0.0, device=device)
        if self.G > 1:
            normed = F.normalize(cur_bh, dim=-1)
            sim = torch.mm(normed, normed.T)
            triu = sim.triu(diagonal=1)
            n_pairs = self.G * (self.G - 1) / 2
            div_loss = -triu.sum() / n_pairs

        # ── Activity ────────────────────────────────────────────────
        action_mag = corr_all.norm(dim=-1).mean(dim=(0, 1))
        act_thr = 0.01
        act_loss = F.relu(act_thr - action_mag).mean()

        # ── Update arbiter maturity ──────────────────────────────────
        self.arbiter.update_maturity(coh_loss.detach(), -div_loss.detach(), action_mag.detach())

        # ── Store old behavior for next step ─────────────────────────
        with torch.no_grad():
            self._behavior_old.copy_(cur_bh.data)

        # ── Smooth coherence EMA ─────────────────────────────────────
        self._coh_ema = self._ema * self._coh_ema + (1 - self._ema) * coh_loss

        losses = {
            'coherence': coh_loss.detach(),
            'diversity': div_loss.unsqueeze(0).detach(),
            'activity': act_loss.unsqueeze(0).detach(),
        }
        return h_out, corr_all, losses

    def reset_state(self):
        self._behavior_old.zero_()
        self._coh_ema.zero_()
        self._behavior.data.zero_()


# ─── Full Stack ──────────────────────────────────────────────────────────

class SandboxMirrorStack(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.D = cfg.D

        self.embed = nn.Embedding(cfg.vocab, cfg.D, padding_idx=0)
        self.layers = nn.ModuleList([
            SandboxLayer(cfg.D, cfg.G, i, cfg)
            for i in range(cfg.n_layers)
        ])
        self.out_norm = nn.LayerNorm(cfg.D)
        self.lm_head = nn.Linear(cfg.D, cfg.vocab, bias=False)

        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, x):
        h = self.embed(x)
        all_losses = []
        total_correction = torch.zeros_like(h)

        for layer in self.layers:
            h, corr, losses = layer(h)
            all_losses.append(losses)
            total_correction = total_correction + corr.sum(dim=2)

        h = self.out_norm(h)
        logits = self.lm_head(h)
        return logits, all_losses, total_correction

    def compute_loss(self, logits, targets, losses):
        B, L, V = logits.shape
        ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction='mean')
        if losses is None or len(losses) == 0:
            return ce, 0., 0., 0.
        N = len(losses)
        coh_total = sum(ld['coherence'].mean() for ld in losses) / N
        div_total = sum(ld['diversity'].mean() for ld in losses) / N
        act_total = sum(ld['activity'].mean() for ld in losses) / N
        total = ce + 0.1 * coh_total + 0.01 * div_total + 0.01 * act_total
        return total, coh_total, div_total, act_total

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()
