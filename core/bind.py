"""BottleneckBind: bilinear D→K→D with golden/Fibonacci twist."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vsa_utils import fib_sigmoid_init


def migrate_bind_state_dict(sd, n_layers, mode="off", S=1):
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


def _golden_shifts(K, S):
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    shifts, used = [], set()
    s = 1
    while len(shifts) < S:
        sh = int(math.floor(s * K / phi)) % K
        while sh in used or sh == 0:
            sh = (sh + 1) % K
        shifts.append(sh); used.add(sh); s += 1
    return shifts


def _fibonacci_shifts(K, S):
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


class BottleneckBind(nn.Module):
    def __init__(self, D, K, cfg):
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
        if getattr(cfg, "bind_qk_norm", False):
            self.hp_norm = nn.RMSNorm(K)
        else:
            self.hp_norm = nn.Identity()

        shifts = _golden_shifts(K, self.S) if scheme == "golden" else _fibonacci_shifts(K, self.S)
        self.register_buffer("shifts", torch.tensor(shifts, dtype=torch.long), persistent=False)

        self.w_u = nn.Parameter(torch.empty(self.S, K))
        self.w_v = nn.Parameter(torch.empty(self.S, K))
        nn.init.normal_(self.w_u, 0.0, 1.0)
        nn.init.normal_(self.w_v, 0.0, 1.0)

        if self.mode == "shift" and tie_bind and self.S > 1:
            self.ocular = "multi"
        if self.mode != "off" and self.ocular == "multi" and self.S > 1:
            self.W_out = nn.Parameter(torch.empty(self.S, K, D))
            nn.init.xavier_uniform_(self.W_out, gain=0.5)
            self._tied = False
        else:
            self.W_out = nn.Parameter(torch.empty(K, D))
            nn.init.xavier_uniform_(self.W_out, gain=0.5)
            self._tied = tie_bind
            if self._tied:
                self.W_proj.register_forward_pre_hook(self._tie_hook)

        if self.gated:
            self.w_gate_proj = nn.Linear(K, self.S, bias=True)
            nn.init.xavier_uniform_(self.w_gate_proj.weight, gain=0.5)
            nn.init.zeros_(self.w_gate_proj.bias)

        if self.mode == "cascade":
            self.mix_logit = nn.Parameter(fib_sigmoid_init(self.S))

    def _tie_hook(self, module, inp):
        with torch.no_grad():
            self.W_out.data.copy_(self.W_proj.weight.data)

    def _cross(self, left, right, shift):
        return left * torch.roll(right, shifts=int(shift), dims=-1)

    def forward(self, h):
        hp = self.hp_norm(self.W_proj(h) + self.w_bind_bias)
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
                a[n] = F.normalize(crossed + 1e-10, dim=-1) * seed_norm
            mix = torch.sigmoid(self.mix_logit)
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
