"""
Dry-run test for sandbox-mirror architecture.

Tests: forward, loss shapes, backward, 100-step training, memory coherence.
"""
import sys, os, math, torch
sys.path.insert(0, os.path.dirname(__file__))
from core import SandboxConfig, SandboxMirrorStack

torch.manual_seed(42)
cfg = SandboxConfig(D=256, n_layers=2, G=4, d_mem=32, seq_len=64, batch_size=2)
model = SandboxMirrorStack(cfg)
print(f'Params: {model.param_count():,} ({model.param_count()/1e6:.2f}M)')

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

# ─── Warmup forward (first call always loses first-step coherence) ────
x = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len))
y = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len))
logits, losses, corr = model(x)
total, coh, div, act = model.compute_loss(logits, y, losses)
print(f'[Warmup] total={total.item():.4f} coh={coh.item():.4f} div={div.item():.4f} act={act.item():.4f}')

total.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
opt.step()
opt.zero_grad()

# ─── Train 100 steps ──────────────────────────────────────────────────
print('\nTraining 100 steps...')
first_coh = None
for step in range(100):
    x = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len))
    y = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len))

    logits, losses, corr = model(x)
    total, coh, div, act = model.compute_loss(logits, y, losses)

    opt.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    opt.step()

    if step == 1:
        first_coh = coh.item()
    if step % 20 == 0 or step < 3:
        coh_val = sum(l['coherence'].mean().item() for l in losses) / max(len(losses), 1)
        div_val = sum(l['diversity'].item() for l in losses) / max(len(losses), 1)
        act_val = sum(l['activity'].item() for l in losses) / max(len(losses), 1)
        print(f'  step {step:>3}: total={total.item():.4f}  coh={coh_val:.6f}  '
              f'div={div_val:.6f}  act={act_val:.6f}  |corr|={corr.norm():.4f}')

last_coh = sum(l['coherence'].mean().item() for l in losses) / max(len(losses), 1)
print(f'\nCoherence: {first_coh:.6f} - {last_coh:.6f}')
if last_coh < first_coh:
    print('  => Coherence is improving (expert becoming more predictable).')
else:
    print('  => Coherence NOT improving (expert not learning to be predictable).')

# ─── Determinism check (pos_id fix) ──────────────────────────────────
print('\nDeterminism check (pos_id):')
model.eval()
x = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len))
with torch.no_grad():
    out1, _, _ = model(x)
    out2, _, _ = model(x)
diff = (out1 - out2).abs().max().item()
print(f'  two evals same input: max|out1-out2|={diff:.2e}')
if diff == 0:
    print('  PASS: deterministic inference')
else:
    print(f'  WARN: non-deterministic (diff={diff:.2e})')

# ─── Reset ──────────────────────────────────────────────────────────────
model.reset_state()
print('  reset_state: OK')

# ─── Expert divergence ────────────────────────────────────────────────
print('\nExpert divergence:')
_, losses, _ = model(torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len)))
for li, ld in enumerate(losses):
    print(f'  layer {li}: coh={ld["coherence"].tolist()}')

print('\nAll tests done.')
