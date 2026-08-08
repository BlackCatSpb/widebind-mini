"""Comprehensive checkpoint analyzer for WideBind Mini.
Usage:
  python scripts/analyze_checkpoint.py <path/to/checkpoint.pt>
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from torch.serialization import add_safe_globals
from core import WideBandConfig, WideBindStack

add_safe_globals([WideBandConfig])


def generate_report(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    sd = ckpt['model']
    model = WideBindStack(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    print('=' * 72)
    print('EVA MINI CHECKPOINT ANALYSIS')
    print('=' * 72)
    print(f'File:       {ckpt_path}')
    print(f'Step:       {ckpt.get("step", "?")}')
    print(f'Val loss:   {ckpt.get("best_val_loss", float("inf")):.4f}')
    print(f'Params:     {model.param_count() / 1e6:.2f}M')
    print(f'Tensors:    {len(sd)}')
    print(f'Missing:    {len(missing)} keys')
    print(f'Unexpected: {len(unexpected)} keys')
    if missing:
        for k in missing[:5]:
            print(f'  MISS: {k}')
    if unexpected:
        for k in unexpected[:5]:
            print(f'  UNEXP: {k}')

    print()
    print('CONFIG')
    print('-' * 72)
    important = ['D', 'n_layers', 'bind_K', 'vocab', 'seq_len', 'batch_size',
                 'mlp_groups', 'mlp_expand', 'head_mode', 'variable_precision',
                 'precision_threshold', 'explicit_reasoning', 'reasoning_max_steps',
                 'private_mem', 'meta_trust', 'collective_layer', 'use_amp',
                 'surprisal_weight', 'branch_balance_weight', 'grad_clip',
                 'warmup_steps', 'max_steps', 'lr']
    for attr in important:
        val = getattr(cfg, attr, 'N/A')
        print(f'  {attr:30s} = {val}')

    print()
    print('LAYER STATS')
    print('-' * 72)
    header = f'{"L":>3s} {"alpha":>7s} {"|1-a|":>7s} {"ls_mean":>9s} {"ls_std":>7s} {"gate_ema":>8s} {"col_m":>6s} {"col_w":>6s}'
    print(header)
    print('-' * len(header))
    for i, layer in enumerate(model.layers):
        mir = layer.mirror
        a = mir.alpha_diag.data
        ls = mir.log_scale.data
        ge = mir._gate_ema

        col_m, col_w = 0, 0
        if layer.collective:
            col_m = int(layer.collective._mature.item())
            col_w = int(layer.collective.N_s.sum().item())

        print(f'{i:3d} {a.mean().item():7.4f} {(1-a.mean()).item():7.4f} {ls.mean().item():9.4f} {ls.std().item():7.4f} {ge.mean().item():8.3f} {col_m:>6} {col_w:>6}')

    print()
    print('VSA TIMESCALES')
    print('-' * 72)
    vsa_log = model._vsa_log_param
    vsa_tau = torch.exp(torch.cumsum(torch.nn.functional.softplus(vsa_log), dim=0)) + 1.0
    print(f'  tau[0]    = {vsa_tau[0].item():.2f}')
    print(f'  tau[-1]   = {vsa_tau[-1].item():.2f}')
    print(f'  tau ratio = {vsa_tau[-1].item() / vsa_tau[0].item():.1f}x')

    print()
    print('PARAMETER GROUPS')
    print('-' * 72)
    groups = {
        'Embedding': list(model.embed.parameters()),
        'LM Head': list(model.lm_head.parameters()),
        'Mirror': [],
        'Bind': [],
        'VSA': [],
        'Conv': [],
        'MLP': [],
        'Collective': [],
        'Other': [],
    }
    # Collect layer-level params
    for layer in model.layers:
        # Mirror params (inside layer.mirror)
        groups['Mirror'].extend(layer.mirror.parameters())
        # Bind params (inside layer.bind)
        if hasattr(layer.bind, 'parameters'):
            groups['Bind'].extend(layer.bind.parameters())
        # Conv
        groups['Conv'].extend(layer.conv.parameters())
        # MLP
        groups['MLP'].extend(layer.mlp.parameters())
        # Collective
        if layer.collective:
            groups['Collective'].extend(layer.collective.parameters())
        # Spectral
        if hasattr(layer, 'lambda_k'):
            groups['Other'].append(layer.lambda_k)
        # Precision gate + exact memory
        if hasattr(layer, 'precision_gate'):
            groups['Other'].extend(layer.precision_gate.parameters())
        if hasattr(layer, 'exact_memory'):
            groups['Other'].extend(layer.exact_memory.parameters())

    # VSA params (top-level + per-layer)
    for p in model.parameters():
        if any(x in str(p) for x in ['vsa', 'w_i', 'w_d', 'w_q', 'b_i', 'b_d', 'scale_w', 'gamma_surprisal', 'w_mem2v', 'w_k_mu', 'w_q_mu', 'w_mu_mem']):
            if p not in [x for grp in groups.values() for x in grp]:
                groups['VSA'].append(p)

    # Reclassify: put anything not yet captured into 'Other'
    captured = set(id(p) for grp in groups.values() for p in grp)
    for p in model.parameters():
        if id(p) not in captured:
            groups['Other'].append(p)

    total = 0
    for name in ['Embedding', 'LM Head', 'Mirror', 'Bind', 'VSA', 'Conv', 'MLP', 'Collective', 'Other']:
        params = groups[name]
        n = sum(p.numel() for p in params)
        total += n
        if n > 0:
            print(f'  {name:15s}: {n:>10,} ({n/1e6:.2f}M)')

    actual_total = sum(p.numel() for p in model.parameters())
    print(f'  {"":-<32s}')
    print(f'  {"GROUPS SUM":15s}: {total:>10,} ({total/1e6:.2f}M)')
    print(f'  {"ACTUAL TOTAL":15s}: {actual_total:>10,} ({actual_total/1e6:.2f}M)')
    if total != actual_total:
        diff = actual_total - total
        print(f'  {"MISSING":15s}: {diff:>10,} ({diff/1e6:.2f}M)')

    print()
    print('VRAM ESTIMATE')
    print('-' * 72)
    param_bytes = total * 4  # fp32
    # Activations ~ params * 3-5x for transformer-like
    act_mult = 3.0 + (2.0 if cfg.variable_precision else 0) + (1.0 if cfg.explicit_reasoning else 0)
    vram_train = param_bytes * 3  # params + grads + optimizer states
    vram_infer = param_bytes * 1.2  # params + activations
    print(f'  Parameters:     {param_bytes/1e9:.2f} GB')
    print(f'  Training est:   {vram_train/1e9:.2f} GB (with optimizer)')
    print(f'  Inference est:  {vram_infer/1e9:.2f} GB')

    print()
    print('TENSOR STATS')
    print('-' * 72)
    all_vals = torch.cat([p.data.flatten() for p in model.parameters()])
    print(f'  Total scalars: {all_vals.numel()}')
    print(f'  Global mean:   {all_vals.mean().item():.6f}')
    print(f'  Global std:    {all_vals.std().item():.6f}')
    print(f'  NaN:           {torch.isnan(all_vals).any().item()}')
    print(f'  Inf:           {torch.isinf(all_vals).any().item()}')

    # Random baseline comparison
    random_loss = torch.log(torch.tensor(float(cfg.vocab)))
    val_loss = ckpt.get('best_val_loss', float('inf'))
    print()
    print('QUALITY ASSESSMENT')
    print('-' * 72)
    print(f'  Random baseline:  {random_loss.item():.4f}')
    print(f'  Current val_loss: {val_loss:.4f}')
    if val_loss > random_loss.item() + 2:
        stage = "Early training (above random)"
    elif val_loss > random_loss.item():
        stage = "Approaching random baseline"
    elif val_loss > random_loss.item() - 1:
        stage = "At random baseline"
    elif val_loss > 9:
        stage = "Learning structure"
    elif val_loss > 8:
        stage = "Words emerging"
    else:
        stage = "Coherent text"
    print(f'  Stage: {stage}')

    # Estimated progress
    if val_loss < 20:
        est_steps_to_9 = int((val_loss - 9) * 250)  # ~0.05 loss/250 steps
        est_hours = est_steps_to_9 / 160 / 3600  # ~160 tok/s
        print(f'  Est. to val<9:   ~{est_steps_to_9:,} steps (~{est_hours:.1f}h)')

    print()
    print('=' * 72)
    print('DONE')


if __name__ == '__main__':
    generate_report(sys.argv[1] if len(sys.argv) > 1
                    else r'C:\Users\black\OneDrive\Desktop\WideBind Mini\checkpoints\best.pt')
