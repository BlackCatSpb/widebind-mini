"""WideBind block: VSA memory + Bind + Conv + Spectral + Mirror + MLP."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import WideBindConfig
from .vsa_utils import dct_basis, fib_sigmoid_init
from .bind import BottleneckBind
from .mirror import GroupedCognitiveMirror
from .mlp import GroupedMLP


class WideBindBlock(nn.Module):
    def __init__(self, cfg: WideBindConfig, layer_idx: int):
        super().__init__()
        self.D = cfg.D
        self.K = cfg.bind_K
        self.layer_idx = layer_idx
        self.tie_bind = cfg.tie_bind
        self.register_buffer('pre_ln_w', torch.ones(cfg.D))
        self.total_layers = cfg.n_layers

        self.bind = BottleneckBind(cfg.D, cfg.bind_K, cfg)

        if getattr(cfg, 'mirror_k_staircase', False):
            n = cfg.n_layers
            l = layer_idx
            if l < n // 3:
                k = 8
            elif l < (2 * n) // 3:
                k = 16
            else:
                k = 32
        else:
            k = cfg.mirror_k
        self.mirror = GroupedCognitiveMirror(cfg.D, G=cfg.mlp_groups, k=k,
            w_pred_scale_init=cfg.w_pred_scale_init, log_scale_init_std=cfg.log_scale_init_std,
            delta_var_ema_min=cfg.delta_var_ema_min, delta_var_ema_max=cfg.delta_var_ema_max,
            tie_mirror_proj=cfg.tie_mirror_proj,
            layer_idx=layer_idx, n_layers=cfg.n_layers,
            has_private_mem=cfg.private_mem)

        self._n_scales = 4
        tau_s = torch.tensor([8, 32, 128, 512], dtype=torch.float32)
        self.register_buffer('_tau_s', tau_s)
        self.w_i = nn.Parameter(torch.randn(cfg.D))
        self.w_d = nn.Parameter(torch.randn(cfg.D) * cfg.w_d_init_std)
        self.w_q = nn.Parameter(torch.full((cfg.D,), 1.0 / math.sqrt(cfg.D)))
        self.w_q_leaf = nn.Parameter(torch.full((cfg.D,), 1.0 / math.sqrt(cfg.D)))
        self.w_q_ctx = nn.Parameter(torch.full((cfg.D,), 0.5 / math.sqrt(cfg.D)))
        self.w_mem2v = nn.Parameter(torch.randn(cfg.D))

        g = self.mirror.G
        d = self.mirror.d
        k = self.mirror.k
        self.w_q_dyn = nn.Parameter(torch.randn(g, k, d) * (1.0 / math.sqrt(k)))
        self.w_i_dyn = nn.Parameter(torch.randn(g, k, d) * (1.0 / math.sqrt(k)))
        self.w_d_pen = nn.Parameter(torch.zeros(g))
        self.w_bind_gate = nn.Parameter(torch.zeros(g))
        self.scale_w = nn.Parameter(fib_sigmoid_init(self._n_scales).unsqueeze(1).expand(-1, cfg.D).clone())

        layer_frac = layer_idx / max(cfg.n_layers - 1, 1)
        b_d_init = 2.0 + 3.0 * layer_frac
        self.b_i = nn.Parameter(torch.full((cfg.D,), -2.5))
        self.b_d = nn.Parameter(torch.full((cfg.D,), b_d_init))
        tau_l = math.exp(b_d_init)
        gamma_max = 0.5
        gamma_init = gamma_max * (1.0 / (1.0 + math.exp(-(math.log(tau_l) - math.log(32.0)))))
        self.gamma_surprisal = nn.Parameter(torch.full((), gamma_init))

        self.w_k_mu = nn.Parameter(torch.randn(cfg.D))
        self.w_q_mu = nn.Parameter(torch.randn(cfg.D))
        self.w_mu_mem = nn.Parameter(torch.randn(cfg.D))

        self.conv = nn.Conv1d(cfg.D, cfg.D, kernel_size=cfg.conv_kernel,
                              padding=cfg.conv_kernel - 1, groups=cfg.D, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='linear')

        self.register_buffer('V_dct', dct_basis(cfg.D))
        base = 0.5 + layer_idx / max(cfg.n_layers - 1, 1)
        freq_scale = torch.linspace(1.0, 0.5, cfg.D)
        per_dim = freq_scale * 0.2
        lam = torch.full((cfg.D,), base) + per_dim
        self.lambda_k = nn.Parameter(lam)

        self.mlp = GroupedMLP(cfg.D, expand=cfg.mlp_expand, groups=cfg.mlp_groups)

    def forward(self, h, state=None, global_state=None,
                mem2v_scale=1.0, diff=None, noise_scale=0.0,
                tanh_bias_mod=1.0, pred_scale_mod=None, spectral_mod=1.0,
                context_mem=None, allow_write=None, tau_s=None):
        mem_state = mu_state = conv_state = None
        if state is not None:
            mem_state, mu_state, conv_state = state
        B, L, D = h.shape
        NaN = float('nan')
        self._nan_at = None

        device = h.device
        if hasattr(self.mirror, '_cached_pred_error_norm') and self.mirror._cached_pred_error_norm is not None:
            pen = self.mirror._cached_pred_error_norm
            if pen.shape[-1] != L:
                self.mirror._cached_pred_error_norm = None
            else:
                self.mirror._cached_pred_error_norm = pen.detach().to(device)
        if hasattr(self.mirror, '_cached_hp') and self.mirror._cached_hp is not None:
            hp = self.mirror._cached_hp
            if hp.shape[1] != L:
                self.mirror._cached_hp = None
            else:
                self.mirror._cached_hp = hp.detach().to(device)

        def _chk(t, label):
            if t.is_floating_point() and (t.isnan().any() or t.isinf().any()):
                self._nan_at = f'L{self.layer_idx}.{label}[{t.min():.2f},{t.max():.2f}]'
                return True
            return False

        h = F.rms_norm(h, (D,), self.pre_ln_w)
        if _chk(h, 'rms_norm'): return h * NaN, (h[0,:1]*NaN, h[0,:1]*NaN, h[0,:1]*NaN)

        if conv_state is None:
            conv_state = torch.zeros(B, D, self.conv.padding[0], device=device, dtype=h.dtype)
        h_perm = h.transpose(1, 2)
        h_conv = self.conv(torch.cat([conv_state, h_perm], dim=-1))
        h_conv = h_conv[..., :L].transpose(1, 2)
        conv_state_out = h_perm[:, :, -(self.conv.padding[0]):]
        h = h + h_conv
        if _chk(h, 'conv'): return h * NaN, (h[0,:1]*NaN, h[0,:1]*NaN, h[0,:1]*NaN)
        self._cache_conv_out = h_conv

        bind_out = self.bind(h)
        if _chk(bind_out, 'bind'): return h * NaN, (h[0,:1]*NaN, h[0,:1]*NaN, h[0,:1]*NaN)

        S = self._n_scales
        tau_s = self._tau_s if tau_s is None else tau_s
        d_s = torch.exp(-1.0 / tau_s.to(device))

        igate_logit = h * self.w_i + self.b_i
        mir = self.mirror
        if hasattr(mir, '_cached_pred_error_norm') and mir._cached_pred_error_norm is not None:
            pen = mir._cached_pred_error_norm
            igate_logit = igate_logit + self.gamma_surprisal * pen.unsqueeze(-1)
        i_gate = F.softplus(igate_logit)
        d_mod = torch.sigmoid(h * self.w_d + self.b_d)

        if hasattr(mir, '_cached_pred_error_norm') and mir._cached_pred_error_norm is not None:
            pen = mir._cached_pred_error_norm
            d_pen_factor = 1.0 + 0.5 * torch.sigmoid(pen.unsqueeze(-1) + self.w_d_pen.unsqueeze(0).unsqueeze(0))
            d_mod = (d_mod.reshape(B, L, self.mirror.G, self.mirror.d) * d_pen_factor.unsqueeze(-1)).reshape(B, L, D)

        if noise_scale > 0 and self.training:
            noise = 1.0 + noise_scale * torch.randn_like(i_gate)
            i_gate = i_gate * noise

        d_s_vec = d_s.view(1, 1, S, 1).expand(B, L, S, D).reshape(B, L, S * D)
        d_mod_vec = d_mod.unsqueeze(2).expand(-1, -1, S, -1).reshape(B, L, S * D)
        decay = (d_s_vec * d_mod_vec).clamp(min=0.01)

        hp_cached = self.mirror._cached_hp
        if hp_cached is not None and self.training:
            write_mod = torch.einsum('blgk,gkd->blgd', hp_cached, self.w_i_dyn)
            write_mod = torch.sigmoid(write_mod / math.sqrt(self.mirror.k))
            mem_input = (h.reshape(B, L, self.mirror.G, self.mirror.d) * write_mod).reshape(B, L, D) * i_gate
        else:
            mem_input = h * i_gate
        input_vec = mem_input.unsqueeze(2).expand(-1, -1, S, -1).reshape(B, L, S * D)

        eps = 1e-6
        CHUNK = 32
        _dtype = decay.dtype
        decay_f32 = decay.float()
        input_vec_f32 = input_vec.float()
        if mem_state is not None:
            mem_state_f32 = mem_state.reshape(B, S * D).float()
        else:
            mem_state_f32 = None

        def _scan_chunk(b_chunk, d_chunk):
            log_a = torch.log(d_chunk.clamp(min=eps))
            log_cum = torch.cumsum(log_a, dim=1)
            cum_decay = torch.exp(log_cum)
            inv_cum = (1.0 / cum_decay.clamp(min=eps)).clamp(max=1e6)
            weighted = b_chunk * inv_cum
            cum_w = torch.cumsum(weighted, dim=1)
            intra = cum_decay * cum_w
            final = intra[:, -1:]
            return intra, final, cum_decay

        def _combine_chunks(chunk_data, initial_state):
            inter_decay = torch.cat([cd[:, -1:] for _, _, cd in chunk_data], dim=1)
            inter_input = torch.cat([f for _, f, _ in chunk_data], dim=1)
            s = initial_state.clone() if initial_state is not None else torch.zeros_like(inter_input[:, 0])
            cross_states = []
            for k in range(len(chunk_data)):
                cross_states.append(s.unsqueeze(1))
                s = inter_decay[:, k] * s + inter_input[:, k]
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

        chunks = []
        for start in range(0, L, CHUNK):
            end = min(start + CHUNK, L)
            intra, final, cum_decay = _scan_chunk(input_vec_f32[:, start:end], decay_f32[:, start:end])
            chunks.append((intra, final, cum_decay))

        mem_all_vec, mem_state_out_vec, mem_leaf_vec = _combine_chunks(chunks, mem_state_f32)
        mem_all_vec = mem_all_vec.to(_dtype)
        mem_state_out_vec = mem_state_out_vec.to(_dtype)
        mem_leaf_vec = mem_leaf_vec.to(_dtype)

        mem_all_vec = mem_all_vec.view(B, L, S, D)
        mem_leaf_vec = mem_leaf_vec.view(B, L, S, D)

        w = torch.sigmoid(self.scale_w)
        mem_all = (mem_all_vec * w.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        mem_leaf = (mem_leaf_vec * w.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        mem_read = mem_all * self.w_q + mem_leaf * self.w_q_leaf + mem_all * self.w_q_ctx
        mem_state_out = mem_state_out_vec.reshape(B, S * D)

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

        if _chk(mem_read, 'mem_read'): return h * NaN, (h[0,:1]*NaN, h[0,:1]*NaN, h[0,:1]*NaN)

        mirror, mlp_mod, mem_mod = self.mirror(
            h, mem_all, global_state=global_state, diff=diff,
            tanh_bias_mod=tanh_bias_mod, pred_scale_mod=pred_scale_mod,
            context_mem=context_mem, allow_write=allow_write)
        if _chk(mirror, 'mirror_out'): return h * NaN, (h[0,:1]*NaN, h[0,:1]*NaN, h[0,:1]*NaN)
        if _chk(mlp_mod, 'mlp_mod'): return h * NaN, (h[0,:1]*NaN, h[0,:1]*NaN, h[0,:1]*NaN)
        if _chk(mem_mod, 'mem_mod'): return h * NaN, (h[0,:1]*NaN, h[0,:1]*NaN, h[0,:1]*NaN)

        mm = mem_mod
        mm = mm.unsqueeze(-1)
        g = self.mirror.G
        d = self.mirror.d
        hp = self.mirror._cached_hp
        read_mod = torch.einsum('blgk,gkd->blgd', hp, self.w_q_dyn)
        read_mod = torch.sigmoid(read_mod / math.sqrt(self.mirror.k))
        mem_read_g = mem_read.reshape(B, L, g, d)
        mem_expert = mem_read_g * read_mod
        mem_modulated = (mem_expert * mm).reshape(B, L, D)
        bind_gate = torch.sigmoid(self.w_bind_gate).unsqueeze(0).unsqueeze(0)
        bind_gated = (bind_out.reshape(B, L, g, d) * mm * bind_gate.unsqueeze(-1)).reshape(B, L, D)
        enhanced_base = bind_gated + mem_modulated * self.w_mem2v * mem2v_scale
        enhanced = enhanced_base + mirror
        if _chk(enhanced, 'enhanced'): return h * NaN, (h[0,:1]*NaN, h[0,:1]*NaN, h[0,:1]*NaN)
        self._cache_bind_out = enhanced_base
        self._cache_mirror_out = mirror
        h = h + enhanced
        if _chk(h, 'post_enhanced'): return h * NaN, (h[0,:1]*NaN, h[0,:1]*NaN, h[0,:1]*NaN)

        h_dct = h @ self.V_dct.T
        h = h + (h_dct * self.lambda_k * spectral_mod) @ self.V_dct
        if _chk(h, 'spectral'): return h * NaN, (h[0,:1]*NaN, h[0,:1]*NaN, h[0,:1]*NaN)

        h_mlp = self.mlp(h)
        if _chk(h_mlp, 'mlp_out'): return h * NaN, (h[0,:1]*NaN, h[0,:1]*NaN, h[0,:1]*NaN)
        mm2 = mlp_mod.unsqueeze(-1)
        h_mlp = (h_mlp.reshape(B, L, g, d) * mm2).reshape(B, L, D)
        h = h + h_mlp
        if _chk(h, 'post_mlp'): return h * NaN, (h[0,:1]*NaN, h[0,:1]*NaN, h[0,:1]*NaN)

        return h, (mem_state_out, mu_state_out, conv_state_out)

    @property
    def base_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith('mirror.')]

    @property
    def mirror_parameters(self):
        return [p for n, p in self.named_parameters() if n.startswith('mirror.')]
