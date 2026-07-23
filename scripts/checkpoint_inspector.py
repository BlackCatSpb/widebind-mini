"""
Checkpoint inspector: load, diagnose, and print key metrics from any WideBind .pt file.

Usage:
  python scripts/checkpoint_inspector.py <path/to/checkpoint.pt> [--codebase PATH]

  If --codebase is omitted, tries the parent of scripts/ as the sys.path root.
  For Mini checkpoints (separate project), pass --codebase <Mini project root>.
"""

import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch


def load_cfg_and_model(ckpt_path, codebase=None):
    if codebase is not None:
        sys.path.insert(0, codebase)
    from core import model as _mod, config as _cfg
    CfgCls = getattr(_cfg, 'WideBindConfig', getattr(_cfg, 'WideBandConfig', None))
    ModelCls = getattr(_mod, 'WideBindStack', getattr(_mod, 'WideBandStack', None))
    if CfgCls is None or ModelCls is None:
        raise ImportError(f'Cannot find Config/Model class in {codebase or "."}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    model = ModelCls(cfg)
    model.load_state_dict(ckpt['model'], strict=False)
    model.train()
    return cfg, model, ckpt


def inspect(cfg, model, ckpt):
    info = {}
    info['step'] = ckpt.get('step', 0)
    info['best_val'] = ckpt.get('best_val_loss', float('inf'))
    info['params'] = sum(p.numel() for p in model.parameters())
    info['D'] = cfg.D
    info['G'] = cfg.mlp_groups
    info['L'] = cfg.n_layers
    info['K'] = cfg.bind_K
    info['has_pm'] = cfg.private_mem
    info['accum'] = getattr(cfg, 'accum_steps', 1)

    x = torch.randint(0, cfg.vocab, (2, min(64, cfg.seq_len)))
    h = model.embed_tokens(x)
    out, _, _ = model(h)
    info['out_std'] = round(out.std().item(), 4)
    ls = model.compute_loss(out[:, :-1], x[:, 1:])
    info['loss'] = round(ls.item(), 4)

    info['mirror'] = []
    for i, layer in enumerate(model.layers):
        m = layer.mirror
        d = {
            'layer': i,
            '|1-a|': round((1.0 - m.alpha_diag.data).abs().mean().item(), 4),
            'ls_var': round(m.log_scale.data.var().item(), 4),
            'ls_exp_max': round(m.log_scale.data.exp().max().item(), 2),
            'gate_var': round(m._last_gates.var().item(), 6),
        }
        if hasattr(m, '_private_mem') and m._private_mem is not None:
            d['pm_norm'] = round(m._private_mem.norm(dim=-1).mean().item(), 4)
        info['mirror'].append(d)

    if cfg.private_mem:
        m0 = model.layers[0].mirror
        w = torch.softmax(m0._signal_log_weights, dim=0)
        info['signals'] = {n: round(w[i].item(), 3)
                           for i, n in enumerate(['temp', 'pred', 'smooth', 'sym', 'help'])}
        info['w_help'] = round(torch.sigmoid(m0.w_help).mean().item(), 3)
        info['w_contra'] = round(m0.w_contra.mean().item(), 3)

        cs = m0._concept_sim_ema
        info['concept_sim'] = {
            'mean': round(cs.mean().item(), 4),
            'std': round(cs.std().item(), 4),
            'diag': round(cs.diag().mean().item(), 4),
        }
        info['behavior_div'] = round(m0._behavior_div_ema.mean().item(), 4)
        tr = m0._trust_matrix
        info['trust'] = {
            'mean': round(tr.mean().item(), 4),
            'diag': round(tr.diag().mean().item(), 4),
        }
        if m0._cached_dominance is not None:
            info['dominance'] = [round(x, 3) for x in m0._cached_dominance.tolist()]
        if m0._cached_isolation is not None:
            info['isolation'] = [round(x, 3) for x in m0._cached_isolation.tolist()]
        if m0._cached_contra_expert is not None:
            info['contra_expert'] = [round(x, 3) for x in m0._cached_contra_expert.tolist()]
        if hasattr(m0, '_pm_step'):
            info['pm_step'] = (int(m0._pm_step.item()), m0._pm_write_delay)

    return info


def print_report(info, file=None):
    out = []
    out.append(f"Step {info['step']}  |  best_val={info.get('best_val', 'inf')}  |  "
               f"params={info['params']:,}  |  D={info['D']} G={info['G']} L={info['L']} K={info['K']}")
    out.append(f"out.std={info['out_std']}  loss={info['loss']}  "
               f"private_mem={info['has_pm']}  accum={info['accum']}")
    out.append("")
    out.append(f"{'Layer':>5}  {'|1-a|':>8}  {'ls_var':>8}  {'ls_exp_max':>10}  "
               f"{'gate_var':>10}  {'pm_norm':>8}")
    for d in info['mirror']:
        pm = d.get('pm_norm', '—')
        out.append(f"L{d['layer']:>02d}  {d['|1-a|']:>8.4f}  {d['ls_var']:>8.4f}  "
                   f"{d['ls_exp_max']:>10.2f}  {d['gate_var']:>10.6f}  {pm:>8}")
    out.append("")

    if 'signals' in info:
        s = info['signals']
        out.append(f"Signals:  temp={s['temp']}  pred={s['pred']}  smooth={s['smooth']}  "
                   f"sym={s['sym']}  help={s['help']}")
        out.append(f"w_help(sigmoid)={info['w_help']}  w_contra={info['w_contra']}")
        cs = info['concept_sim']
        out.append(f"concept_sim: mean={cs['mean']}  std={cs['std']}  diag={cs['diag']}")
        out.append(f"behavior_div: {info['behavior_div']}")
        t = info['trust']
        out.append(f"trust: mean={t['mean']}  diag={t['diag']}")
        if 'dominance' in info:
            out.append(f"dominance: {info['dominance']}")
        if 'isolation' in info:
            out.append(f"isolation: {info['isolation']}")
        if 'contra_expert' in info:
            out.append(f"contra_expert: {info['contra_expert']}")
        if 'pm_step' in info:
            cur, max_ = info['pm_step']
            out.append(f"pm_step: {cur}/{max_}")
        if 'gate_ema' in info:
            ge = info['gate_ema']
            out.append(f"gate_ema: mean={ge.mean():.3f} min={ge.min():.3f} max={ge.max():.3f}")

    text = '\n'.join(out)
    if file is None:
        print(text)
    else:
        file.write(text)
    return text


def main():
    p = argparse.ArgumentParser(description='Inspect WideBind checkpoint')
    p.add_argument('checkpoint', type=str, help='Path to .pt checkpoint')
    p.add_argument('--codebase', type=str, default=None,
                   help='Project root with core/ (for Mini checkpoints)')
    args = p.parse_args()

    cfg, model, ckpt = load_cfg_and_model(args.checkpoint, args.codebase)
    info = inspect(cfg, model, ckpt)
    print_report(info)


if __name__ == '__main__':
    main()
