"""
WideBand Mini — compact local training (FP32, no AMP).
Defaults: D=896, L=12, G=8, ~40M params, fits 6+ GB VRAM.

Usage:
    python train.py --data-dir ./data
    python train.py --data-dir ./data --D 1024 --n-layers 16
    python train.py --data-dir ./data --accum 8  (effective batch = 1024*8 = 8192)
"""

import os, sys, math, time, glob, argparse, gc
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from core import WideBandConfig, WideBindStack, MirrorLRScheduler, AdaptiveController


# ─── Data ────────────────────────────────────────────────────────────────

class TokenStream:
    def __init__(self, path):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.len = len(self.data)

    def get_batch(self, seq_len, batch_size, offset):
        need = batch_size * seq_len + 1
        if offset + need > self.len:
            offset = 0
        chunk = self.data[offset:offset + need]
        x = torch.from_numpy(chunk[:batch_size * seq_len].copy()).long().view(batch_size, seq_len)
        y = torch.from_numpy(chunk[1:batch_size * seq_len + 1].copy()).long().view(batch_size, seq_len)
        return x, y, offset + batch_size * seq_len


def load_streams(data_dir):
    pattern = os.path.join(data_dir, 'token_stream_*_clean.bin')
    files = sorted(glob.glob(pattern))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, 'token_stream_*.bin')))
    if not files:
        raise FileNotFoundError(f'No token_stream_*.bin in {data_dir}')
    streams = [TokenStream(f) for f in files]
    total = sum(s.len for s in streams)
    print(f'  Data: {len(streams)} files, {total:,} tokens')
    return streams, total


# ─── Eval ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, streams, cfg, device):
    model.eval()
    total_loss = 0.0
    steps = 0
    n = max(1, sum(s.len for s in streams) // (cfg.batch_size * cfg.seq_len) // len(streams))
    n = min(100, n)
    for s in streams:
        off = max(s.len // 4, cfg.batch_size * cfg.seq_len + 1)
        state = gs = None
        for _ in range(n):
            x, y, off = s.get_batch(cfg.seq_len, cfg.batch_size, off)
            if off == 0:
                break
            x, y = x.to(device), y.to(device)
            h = model.embed_tokens(x)
            out, state, gs = model(h, state, global_state=gs, adaptive=False)
            loss = model.compute_loss(out, y)
            total_loss += loss.item()
            steps += 1
            if _ % 25 == 24:
                state = gs = None
        del state, gs
        gc.collect()
    model.train()
    return total_loss / max(steps, 1)


# ─── Train ───────────────────────────────────────────────────────────────

def train(cfg, data_dir, device):
    print(f'Device: {device} ({torch.cuda.get_device_name(0) if device=="cuda" else "cpu"})')
    if device == 'cuda':
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'VRAM: {mem:.1f} GB')

    # Model
    model = WideBindStack(cfg).to(device)
    n = model.param_count()
    print(f'Model: {n:,} params ({n/1e6:.2f}M)')
    if getattr(cfg, 'compile', False):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print('  torch.compile: ON')
        except Exception:
            print('  torch.compile: SKIP')
    if device == 'cuda':
        print(f'  VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB')

    # Phase tracking state (EMA-based adaptive threshold)
    model._phase_ratio_ema = [0.0] * cfg.n_layers
    model._phase_ratio_std = [1.0] * cfg.n_layers

    # Data
    streams, total_tokens = load_streams(data_dir)

    # Optimizer
    groups = model.param_groups()
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    scheduler = MirrorLRScheduler(model, optimizer, cfg=cfg)

    # Resume
    start_step = 0
    best_val = float('inf')
    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(cfg.save_dir, 'step_*.pt')),
                   key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]))
    if ckpts:
        ckpt = torch.load(ckpts[-1], map_location=device, weights_only=False)
        miss, _ = model.load_state_dict(ckpt['model'], strict=False)
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except Exception:
            pass
        try:
            scheduler.load_state_dict(ckpt['scheduler'])
        except Exception:
            pass
        start_step = ckpt.get('step', 0)
        best_val = ckpt.get('best_val_loss', float('inf'))
        print(f'Resumed step {start_step} (miss={len(miss)})')
        # NaN sanity
        with torch.no_grad():
            xt = torch.randint(0, 50000, (cfg.batch_size, cfg.seq_len), device=device)
            ot, _, _ = model(model.embed_tokens(xt))
            if torch.isnan(ot).any():
                raise RuntimeError('NaN after resume — weights corrupted')
        del ckpt; gc.collect()
    else:
        print('Fresh start')

    # Training loop
    accum = getattr(cfg, 'accum_steps', 1)
    state = gs = None
    stream_idx = 0
    offset = 0
    tokens = 0
    t0 = time.time()
    rng = torch.Generator().manual_seed(42)

    def detach(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach()
        if isinstance(x, (tuple, list)):
            return type(x)(detach(v) for v in x)
        return x

    def _soft_reset(state, factor=0.3):
        if state is None:
            return None
        if isinstance(state, torch.Tensor):
            return state * factor
        if isinstance(state, (list, tuple)):
            return type(state)(_soft_reset(s, factor) for s in state)
        return state

    print(f'Training: {start_step} -> {cfg.max_steps}')
    print(f'  Tokens/step: {cfg.batch_size * cfg.seq_len}')
    print(f'  Accum: {accum}  (effective batch: {cfg.batch_size * cfg.seq_len * accum})')

    try:
        for step in range(start_step, cfg.max_steps):
            model.train()

            # Sample batch
            s = streams[stream_idx]
            x, y, offset = s.get_batch(cfg.seq_len, cfg.batch_size, offset)
            if offset == 0:
                stream_idx = (stream_idx + 1) % len(streams)
                state = gs = None

            x, y = x.to(device), y.to(device)

            # Soft EOS reset: decay state instead of dropping it
            if (y[:, -1] == 2).any() and state is not None:
                state = _soft_reset(state, factor=0.3)

            # Forward (pure FP32, no autocast)
            h = model.embed_tokens(x)
            out, state, gs = model(h, state, global_state=gs, step=step)

            # Compute losses (raw, unweighted)
            ce_loss, aux_dict = model.compute_losses(out, y)

            # NaN guard
            if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                raise RuntimeError(f'NaN/Inf CE loss at step {step}')

            state = detach(state)
            gs = detach(gs)

            # CE gradients (retain graph for aux backward)
            ce_grads = torch.autograd.grad(ce_loss / accum, model.parameters(),
                                           retain_graph=bool(aux_dict), allow_unused=True)

            # Spectral gradient alignment: scale aux by cos(g_CE, g_aux)
            if aux_dict:
                aux_total = sum(
                    v for v in aux_dict.values()
                    if isinstance(v, torch.Tensor) and v.requires_grad
                ) / accum
                aux_grads = torch.autograd.grad(aux_total, model.parameters(), allow_unused=True)
                # Flatten shared subspace (params where BOTH grads exist)
                ce_list, aux_list = [], []
                for gce, gaux in zip(ce_grads, aux_grads):
                    if gce is not None and gaux is not None:
                        ce_list.append(gce.flatten())
                        aux_list.append(gaux.flatten())
                if ce_list:
                    ce_flat = torch.cat(ce_list)
                    aux_flat = torch.cat(aux_list)
                    cos_sim = F.cosine_similarity(ce_flat.unsqueeze(0), aux_flat.unsqueeze(0))
                    scale = max(0, min(10.0, cos_sim.item())) * ce_flat.norm() / (aux_flat.norm() + 1e-8)
                else:
                    scale = 0.0
            else:
                scale = 0.0

            # Combine: g = g_CE + scale * g_aux (scale > 0 only when aux aligns with CE)
            with torch.no_grad():
                for p, cg in zip(model.parameters(), ce_grads):
                    if cg is not None:
                        p.grad = cg.clone()
                    else:
                        p.grad = None
                if aux_dict and scale > 0:
                    for p, ag in zip(model.parameters(), aux_grads):
                        if p.grad is not None and ag is not None:
                            p.grad.add_(ag, alpha=scale)
                        elif ag is not None:
                            p.grad = ag * scale
            
            # Adaptive phase scaling: EMA-based mirror/base gradient balance
            # mirror_scale = sigmoid((ratio - ratio_ema) / (ratio_std + 1e-8))
            # When ratio exceeds EMA by >1 std, mirror is growing → sigmoid > 0.5 → boost
            phase_scales = []
            for i, layer in enumerate(model.layers):
                mirror_norm = 0.0
                base_norm = 0.0
                for p in layer.mirror_parameters:
                    if p.grad is not None:
                        mirror_norm += p.grad.norm().item() ** 2
                for p in layer.base_parameters:
                    if p.grad is not None:
                        base_norm += p.grad.norm().item() ** 2
                mirror_norm = mirror_norm ** 0.5
                base_norm = base_norm ** 0.5
                ratio = mirror_norm / (base_norm + 1e-8)
                ema = model._phase_ratio_ema[i]
                std = model._phase_ratio_std[i]
                model._phase_ratio_ema[i] = 0.99 * ema + 0.01 * ratio
                model._phase_ratio_std[i] = 0.99 * std + 0.01 * abs(ratio - ema)
                mir_s = max(0.2, min(2.0, 1.0 / (1.0 + math.exp(-(ratio - ema) / (std + 1e-8)))))
                phase_scales.append((mir_s, ratio))
                for p in layer.mirror_parameters:
                    if p.grad is not None:
                        p.grad *= mir_s
            mean_mirror_scale = sum(s[0] for s in phase_scales) / len(phase_scales)
            mean_ratio = sum(s[1] for s in phase_scales) / len(phase_scales)
            tokens += cfg.batch_size * cfg.seq_len

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if step % cfg.log_interval == 0:
                dt = time.time() - t0
                try:
                    idiff = torch.stack([(1.0 - l.mirror.alpha_diag.data).abs().mean() for l in model.layers]).mean().item()
                    gvar = torch.stack([l.mirror._last_gates.var() for l in model.layers]).mean().item()
                    ls_var = torch.stack([l.mirror.log_scale.data.var() for l in model.layers]).mean().item()
                except Exception:
                    idiff = gvar = ls_var = 0.0
                lr = scheduler.get_last_lr()[0]
                mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0
                # Individual aux losses from compute_loss cache
                lc = getattr(model, '_cached_losses', {})
                aux_str = ' '.join(f'{k}={v:.4f}' for k, v in lc.items())
                print(f'step={step:>6} loss={ce_loss.item():.4f} |1-a|={idiff:.4f} '
                      f'g_var={gvar:.4f} ls_var={ls_var:.4f} lr={lr:.2e} tok/s={tokens/dt:.0f} '
                      f'ms={mean_mirror_scale:.3f} mr={mean_ratio:.4f} '
                      f'mem={mem_gb:.2f}GB | {aux_str}')
                if device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()

            if step > 0 and step % cfg.eval_interval == 0:
                vl = evaluate(model, streams, cfg, device)
                print(f'  EVAL step={step}: val_loss={vl:.4f} ppl={math.exp(vl):.2f}')
                scheduler.report_val_loss(vl)
                torch.cuda.empty_cache(); gc.collect()
                if vl < best_val:
                    best_val = vl
                    torch.save({
                        'step': step, 'model': model.state_dict(),
                        'best_val_loss': best_val, 'cfg': cfg,
                    }, os.path.join(cfg.save_dir, 'best.pt'))
                    print(f'  New best!')
                torch.save({
                    'step': step, 'model': model.state_dict(),
                    'best_val_loss': best_val, 'cfg': cfg,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, os.path.join(cfg.save_dir, f'eval_{step}.pt'))
                print(f'  Saved eval_{step}.pt')

            if step > 0 and step % cfg.save_interval == 0:
                ckpt = {
                    'step': step, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_loss': best_val, 'cfg': cfg,
                }
                torch.save(ckpt, os.path.join(cfg.save_dir, f'step_{step}.pt'))
                print(f'  Saved step_{step}.pt ({len(ckpt)} keys)')
                del ckpt; gc.collect()

    except KeyboardInterrupt:
        ckpt = {
            'step': step, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val, 'cfg': cfg,
        }
        path = os.path.join(cfg.save_dir, f'step_{step}.pt')
        torch.save(ckpt, path)
        print(f'\nInterrupted. Saved {path}')

    print('Done.')


# ─── CLI ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WideBand Mini — local FP32 training')
    parser.add_argument('--data-dir', required=True, help='Directory with token_stream_*.bin files')
    parser.add_argument('--save-dir', default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--D', type=int, default=896, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--mlp-groups', type=int, default=8, help='MLP groups')
    parser.add_argument('--mlp-expand', type=int, default=4, help='MLP expansion factor')
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max-steps', type=int, default=300000, help='Training steps')
    parser.add_argument('--eval-interval', type=int, default=500, help='Eval every N steps')
    parser.add_argument('--save-interval', type=int, default=2000, help='Save every N steps')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile (~30% tok/s)')
    parser.add_argument('--div-weight', type=float, default=0.087, help='Expert diversity loss weight (pushes var(log_scale) up)')
    parser.add_argument('--private-mem', action='store_true', help='Enable cross-expert private memory bank')
    parser.add_argument('--no-lambda', action='store_true', help='Disable lambda_d hierarchy')
    parser.add_argument('--accum', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--bind-twist-mode', default='shift', help='BottleneckBind twist mode (off/shift/cascade)')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()

    cfg = WideBandConfig(
        D=args.D,
        n_layers=args.n_layers,
        mlp_groups=args.mlp_groups,
        mlp_expand=args.mlp_expand,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        lambda_d_enabled=not args.no_lambda,
        bind_twist_mode=args.bind_twist_mode,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        grad_clip=0.5,
        conv_kernel=48,
        accum_steps=args.accum,
        compile=args.compile,
        div_weight=args.div_weight,
        private_mem=args.private_mem,
    )

    device = args.device if torch.cuda.is_available() else 'cpu'
    train(cfg, args.data_dir, device)