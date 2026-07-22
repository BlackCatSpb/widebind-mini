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
            out, state, gs = model(h, state, global_state=gs)
            loss = model.compute_loss(out, y) / accum

            # NaN guard
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f'NaN/Inf loss at step {step}')

            state = detach(state)
            gs = detach(gs)
            loss.backward()
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
                print(f'step={step:>6} loss={loss.item()*accum:.4f} |1-a|={idiff:.4f} '
                      f'g_var={gvar:.4f} ls_var={ls_var:.4f} lr={lr:.2e} tok/s={tokens/dt:.0f} mem={mem_gb:.2f}GB')
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
    )

    device = args.device if torch.cuda.is_available() else 'cpu'
    train(cfg, args.data_dir, device)