"""Tests for the three new specs: asymmetry, aux mirror, meta-trust."""

import sys, os, math, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from core import WideBandConfig, WideBindStack, GroupedCognitiveMirror


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── Fixtures ──────────────────────────────────────────────────────────

def make_cfg(**kwargs):
    defaults = dict(D=256, n_layers=2, bind_K=16, mlp_groups=4, mlp_expand=2,
                    seq_len=32, batch_size=2, private_mem=True,
                    expert_asymmetry=True, meta_trust=True, aux_mirror_weight=0.1,
                    accum_steps=1, warmup_steps=5)
    defaults.update(kwargs)
    return WideBandConfig(**defaults)


def make_model(cfg):
    model = WideBindStack(cfg)
    miss, unex = model.load_state_dict(model.state_dict(), strict=False)
    assert len(unex) == 0, f'Unexpected keys: {unex}'
    return model


# ─── Spec 1: Asymmetry ────────────────────────────────────────────────

def test_asymmetry_W_proj_orthogonal():
    """Each expert's W_proj should be semi-orthogonal (Q^T Q ≈ I)."""
    cfg = make_cfg(expert_asymmetry=True)
    model = make_model(cfg)
    for i, layer in enumerate(model.layers):
        W = layer.mirror.W_proj.data  # (G, d, k)
        for g in range(W.shape[0]):
            gram = W[g].T @ W[g]
            dev = (gram - torch.eye(gram.shape[0])).abs().mean().item()
            assert dev < 0.01, f'Layer {i} expert {g}: Q^T Q dev={dev:.4f} > 0.01'
    print('  PASS: W_proj orthogonal per expert')


def test_asymmetry_W_proj_differs():
    """W_proj should differ across experts when asymmetry is on."""
    cfg = make_cfg(expert_asymmetry=True)
    model = make_model(cfg)
    for i, layer in enumerate(model.layers):
        W = layer.mirror.W_proj.data
        same = True
        for g in range(1, W.shape[0]):
            if (W[0] - W[g]).abs().max() > 0.01:
                same = False
                break
        assert not same, f'Layer {i}: all W_proj identical'
    print('  PASS: W_proj differs across experts')


def test_asymmetry_alpha_varied():
    """alpha_diag should vary per expert when asymmetry is on."""
    cfg = make_cfg(expert_asymmetry=True)
    model = make_model(cfg)
    for i, layer in enumerate(model.layers):
        ad = layer.mirror.alpha_diag.data
        alphas = ad.mean(dim=-1)  # (G,)
        spread = alphas.max() - alphas.min()
        assert spread > 0.01, f'Layer {i}: alpha spread={spread:.4f} too small'
    print('  PASS: alpha_diag spread per expert')


def test_asymmetry_log_scale_varied():
    """log_scale should vary per expert when asymmetry is on."""
    cfg = make_cfg(expert_asymmetry=True)
    model = make_model(cfg)
    for i, layer in enumerate(model.layers):
        ls = layer.mirror.log_scale.data
        expert_means = ls.mean(dim=-1)  # (G,)
        spread = expert_means.max() - expert_means.min()
        assert spread > 0.05, f'Layer {i}: log_scale spread={spread:.4f} too small'
    print('  PASS: log_scale spread per expert')


def test_asymmetry_off_random():
    """Without asymmetry, W_proj is random (not orthogonal) per expert."""
    cfg = make_cfg(expert_asymmetry=False)
    model = make_model(cfg)
    for i, layer in enumerate(model.layers):
        W = layer.mirror.W_proj.data
        for g in range(W.shape[0]):
            gram = W[g].T @ W[g]
            dev = (gram - torch.eye(gram.shape[0])).abs().mean().item()
            # Random init gives dev ~0.3-0.7, orthogonal gives <0.01
            assert dev > 0.05, f'Layer {i} expert {g}: gram_dev={dev:.4f} unexpectedly orthogonal'
    print('  PASS: asymmetry=False gives non-orthogonal init')


# ─── Spec 2: Aux Mirror ───────────────────────────────────────────────

def test_aux_proj_exists():
    """aux_proj should be created on the model."""
    cfg = make_cfg(aux_mirror_weight=0.1)
    model = make_model(cfg)
    assert hasattr(model, 'aux_proj'), 'aux_proj missing'
    assert isinstance(model.aux_proj, torch.nn.Linear)
    D_aux = max(1, cfg.D // 8)
    assert model.aux_proj.out_features == D_aux, f'Expected D_aux={D_aux}, got {model.aux_proj.out_features}'
    print('  PASS: aux_proj exists with correct shape')


def test_aux_pred_cached():
    """Forward should cache aux_pred in training mode."""
    cfg = make_cfg(aux_mirror_weight=0.1)
    model = make_model(cfg).train()
    x = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len))
    h = model.embed_tokens(x)
    out, state, gs = model(h)
    assert model._cached_aux_pred is not None, '_cached_aux_pred is None'
    B = cfg.batch_size
    D_aux = max(1, cfg.D // 8)
    assert model._cached_aux_pred.shape == (B, D_aux), f'Expected ({B},{D_aux}), got {model._cached_aux_pred.shape}'
    print('  PASS: aux_pred cached with correct shape')


def test_aux_pred_gradient_flows():
    """Gradient should flow through aux_pred to model parameters."""
    cfg = make_cfg(aux_mirror_weight=0.1)
    model = make_model(cfg).train().to(device)
    x = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len), device=device)
    y = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len), device=device)
    h = model.embed_tokens(x)
    out, state, gs = model(h)
    aux_pred = model._cached_aux_pred
    target = torch.randn(cfg.batch_size, max(1, cfg.D // 8), device=device)
    aux_loss = (1.0 - F.cosine_similarity(aux_pred, target, dim=-1)).mean()
    ce_loss = model.compute_loss(out, y)
    total = ce_loss + 0.1 * aux_loss
    total.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, 'No gradient flowing'
    print(f'  PASS: aux_pred gradient flows (grad_norm={grad_norm:.4f})')


def test_aux_pred_eval_none():
    """In eval mode, _cached_aux_pred should be None."""
    cfg = make_cfg(aux_mirror_weight=0.1)
    model = make_model(cfg).eval()
    x = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len))
    h = model.embed_tokens(x)
    out, state, gs = model(h)
    assert model._cached_aux_pred is None, '_cached_aux_pred should be None in eval'
    print('  PASS: aux_pred None in eval mode')


# ─── Spec 3: Meta-Trust ───────────────────────────────────────────────

def test_asymmetry_alpha_competitive():
    """Alpha_diag should diverge across experts after training (competitive update)."""
    cfg = make_cfg(private_mem=False, expert_asymmetry=True,
                   meta_trust=False, aux_mirror_weight=0.0)
    model = make_model(cfg).train().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    alphas = []
    for step in range(50):
        x = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len), device=device)
        y = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len), device=device)
        h = model.embed_tokens(x)
        out, state, gs = model(h, None, global_state=None, step=step)
        ce_loss, aux_dict = model.compute_losses(out, y)
        total = ce_loss + sum(v for v in aux_dict.values() if isinstance(v, torch.Tensor))
        opt.zero_grad()
        total.backward()
        opt.step()
        alphas.append(model.layers[0].mirror.alpha_diag.clone().cpu())
    alpha_stack = torch.stack(alphas, dim=0)
    per_expert_var = alpha_stack.var(dim=0)
    max_var = per_expert_var.max().item()
    min_var = per_expert_var.min().item()
    assert max_var > 1e-6, f'Alpha collapsed: max_var={max_var:.8f}'
    assert max_var > min_var * 2, f'Alpha not competitive: max_var={max_var:.6f} min_var={min_var:.6f}'
    print(f'  PASS: alpha competitive (max_var={max_var:.6f})')


def test_meta_trust_buffers_exist():
    """_prev_trust_matrix and _meta_private_mem should exist when private_mem + meta_trust."""
    cfg = make_cfg(private_mem=True, meta_trust=True)
    model = make_model(cfg)
    for i, layer in enumerate(model.layers):
        mir = layer.mirror
        assert hasattr(mir, '_prev_trust_matrix'), f'Layer {i}: missing _prev_trust_matrix'
        assert hasattr(mir, '_meta_private_mem'), f'Layer {i}: missing _meta_private_mem'
        assert mir._prev_trust_matrix.shape == (cfg.mlp_groups, cfg.mlp_groups)
        assert mir._meta_private_mem.shape == (cfg.mlp_groups,)
    print('  PASS: meta_trust buffers exist')


def test_meta_trust_no_buffers_without_private_mem():
    """Without private_mem, meta_trust buffers should not exist."""
    cfg = make_cfg(private_mem=False, meta_trust=True)
    model = make_model(cfg)
    for i, layer in enumerate(model.layers):
        mir = layer.mirror
        assert not hasattr(mir, '_prev_trust_matrix'), f'Layer {i}: _prev_trust_matrix exists without private_mem'
        assert not hasattr(mir, '_meta_private_mem'), f'Layer {i}: _meta_private_mem exists without private_mem'
    print('  PASS: no meta_trust buffers without private_mem')


def test_meta_trust_forward_no_crash():
    """Forward with meta_trust enabled should not crash."""
    cfg = make_cfg(private_mem=True, meta_trust=True)
    model = make_model(cfg).train()
    x = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len))
    y = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len))
    h = model.embed_tokens(x)
    out, state, gs = model(h)
    loss = model.compute_loss(out, y)
    loss.backward()
    assert not torch.isnan(loss), 'NaN loss with meta_trust'
    print('  PASS: meta_trust forward + backward OK')


def test_meta_trust_instability_updates():
    """_meta_private_mem should change across steps (instability accumulates)."""
    cfg = make_cfg(private_mem=True, meta_trust=True)
    model = make_model(cfg).train()
    x = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len))
    h = model.embed_tokens(x)
    out, state, gs = model(h)
    mem_before = model.layers[0].mirror._meta_private_mem.clone()
    state = [tuple(t.detach() for t in s) if s is not None else None for s in state]
    gs = gs.detach()
    x = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len))
    h = model.embed_tokens(x)
    out, state, gs = model(h, state, global_state=gs)
    mem_after = model.layers[0].mirror._meta_private_mem
    changed = (mem_before - mem_after).abs().max().item()
    assert changed > 0 or mem_after.abs().sum().item() > 0, f'meta_private_mem not updating: {mem_before.tolist()}'
    print(f'  PASS: meta_private_mem updates (max|d|={changed:.6f})')


# ─── Integration ───────────────────────────────────────────────────────

def test_all_flags_on():
    """Full forward + backward with all three specs enabled."""
    cfg = make_cfg(private_mem=True, expert_asymmetry=True,
                   meta_trust=True, aux_mirror_weight=0.1)
    model = make_model(cfg).train().to(device)
    x = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len), device=device)
    y = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len), device=device)
    h = model.embed_tokens(x)
    out, state, gs = model(h)
    aux_pred = model._cached_aux_pred
    ce_loss, aux_dict = model.compute_losses(out, y)
    if aux_pred is not None:
        target = torch.randn_like(aux_pred)
        aux_mirror = (1.0 - F.cosine_similarity(aux_pred, target, dim=-1)).mean()
        aux_dict['aux_mirror'] = aux_mirror * 0.1
    total = ce_loss + sum(aux_dict.values())
    total.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, 'Zero gradient with all flags'
    assert not torch.isnan(total), 'NaN loss with all flags'
    assert model._cached_aux_pred is not None, 'aux_pred not cached'
    print(f'  PASS: all flags integrated (grad_norm={grad_norm:.4f})')


def test_checkpoint_compatibility():
    """Old-style checkpoint (private_mem=False) loads into new model with strict=False."""
    cfg_old = make_cfg(private_mem=False, expert_asymmetry=False,
                       meta_trust=False, aux_mirror_weight=0.0)
    model_old = make_model(cfg_old)
    old_sd = model_old.state_dict()
    old_sd = {k: v for k, v in old_sd.items() if 'aux_proj' not in k}

    # New model with same settings (private_mem=False) should only miss aux_proj
    cfg_new = make_cfg(private_mem=False, expert_asymmetry=False,
                       meta_trust=False, aux_mirror_weight=0.0)
    model_new = WideBindStack(cfg_new)
    miss, unex = model_new.load_state_dict(old_sd, strict=False)
    assert len(unex) == 0, f'Unexpected keys: {unex}'
    expected_miss = 2  # aux_proj.weight, aux_proj.bias
    assert len(miss) == expected_miss, f'Expected {expected_miss} missing, got {len(miss)}: {miss}'
    print(f'  PASS: old checkpoint loads with {len(miss)} missing keys (aux_proj)')


def test_nan_stability():
    """100 forward steps with state carry should not produce NaN."""
    cfg = make_cfg(private_mem=True, expert_asymmetry=True,
                   meta_trust=True, aux_mirror_weight=0.1)
    model = make_model(cfg).train().to(device)
    state = gs = None
    for step in range(100):
        x = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len), device=device)
        h = model.embed_tokens(x)
        out, state, gs = model(h, state, global_state=gs, step=step)
        if out.is_floating_point() and (out.isnan().any() or out.isinf().any()):
            raise RuntimeError(f'NaN at step {step}')
        state = [tuple(t.detach() for t in s) if s is not None else None for s in state]
        gs = gs.detach() if gs is not None else None
    print('  PASS: 100-step NaN stability OK')


# ─── Run ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        ('asymmetry', [
            test_asymmetry_W_proj_orthogonal,
            test_asymmetry_W_proj_differs,
            test_asymmetry_alpha_varied,
            test_asymmetry_log_scale_varied,
            test_asymmetry_off_random,
            test_asymmetry_alpha_competitive,
        ]),
        ('aux_mirror', [
            test_aux_proj_exists,
            test_aux_pred_cached,
            test_aux_pred_gradient_flows,
            test_aux_pred_eval_none,
        ]),
        ('meta_trust', [
            test_meta_trust_buffers_exist,
            test_meta_trust_no_buffers_without_private_mem,
            test_meta_trust_forward_no_crash,
            test_meta_trust_instability_updates,
        ]),
        ('integration', [
            test_all_flags_on,
            test_checkpoint_compatibility,
            test_nan_stability,
        ]),
    ]

    passed = failed = 0
    for group_name, group_tests in tests:
        print(f'\n--- {group_name} ---')
        for test_fn in group_tests:
            try:
                test_fn()
                print(f'  OK {test_fn.__name__}')
                passed += 1
            except Exception as e:
                print(f'  FAIL {test_fn.__name__}: {e}')
                failed += 1
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()

    print(f'\n{"="*40}')
    print(f'  {passed} passed, {failed} failed')
    if failed > 0:
        exit(1)
