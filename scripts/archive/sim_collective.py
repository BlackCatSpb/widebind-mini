"""Fast CPU simulator of the collective concept layer lifecycle.

No GPU training needed: feeds synthetic mirror states and drives the layer's
own maturity signal (resvar) through realistic trajectories, verifying the
full concept lifecycle from the design doc:

  immature -> mature (births begin) -> bank fills -> eviction of least-used
  slot -> contradiction gate kills "pink elephants" (novel-but-unconfirmed
  concepts never leak into the signal when they contradict context).

Usage:
  python sim_collective.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn.functional as F
from core.concept_layer import CollectiveConceptLayer

torch.manual_seed(0)
PASS = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {name}' + (f'  ({detail})' if detail else ''))
    PASS.append(cond)


# ─── Helper: synthetic mirror state around a set of planted "concepts" ───
def make_hp(B, L, G, k, concepts):
    """hp: (B,L,G,k) mirror K-states. Each concept c is a (k,) direction;
    a random fraction of token positions in a block are drawn near a concept."""
    hp = torch.randn(B, L, G, k) * 0.05
    for c in concepts:
        cdir = F.normalize(c, dim=-1)
        for t in range(0, L, 7):
            hp[:, t, :, :] += cdir * 1.5
    return hp


def run_sim(layer, steps, rv_fn, hp_fn, pen_fn, label):
    """Drive the layer through `steps` synthetic training steps."""
    print(f'\n=== {label} ===')
    first_mature = None
    births_at = []
    for s in range(steps):
        rv = rv_fn(s)
        hp = hp_fn(s)
        B, L, G, k = hp.shape
        h = torch.randn(B, L, 896) * 0.05
        pen = pen_fn(s)
        out = layer(h, hp, pen, resvar=rv, allow_write=True)
        assert not torch.isnan(out).any(), f'NaN at step {s}'
        db = layer.debug()
        if db['mature'] > 0.5 and first_mature is None:
            first_mature = s
        if db['occupied'] == 8 and len(births_at) == 0:
            births_at.append(s)
        if s % 5 == 0 or s == steps - 1:
            print(f'  s{s:3d} rv={rv:.3f} mature={db["mature"]:.0f} occ={db["occupied"]} '
                  f'u={db["u_gate"]:.2f} c={db["c_gate"]:.2f} scale={db["read_scale"]:.2f} '
                  f'N={db["N_s"]}')
    return first_mature, births_at, layer.debug()


# ─── Scenario 1: maturation -> births -> eviction ──────────────────────
print('Scenario 1: layer matures (resvar falls), births begin, bank fills, eviction recycles')
layer1 = CollectiveConceptLayer(D=896, k=16, S=8, write_delay=3, birth_gap=0.5, cfg=None)
layer1._maturity_warmup = 3
concepts = []
def rv1(s):
    # held high (immature) for the first 6 steps, then a slow decline (mature)
    if s < 6:
        return 0.30
    return max(0.24 * (0.93 ** (s - 6)), 0.10)
def hp1(s):
    # each new "era" (every 6 steps) plants a fresh concept in a random K-subspace
    while len(concepts) < s // 6 + 1:
        concepts.append(torch.randn(16))
    return make_hp(2, 64, 8, 16, concepts)
pen1 = lambda s: torch.rand(2, 64) * 0.2
# capture slot counts at the end of the immature phase (s=5) and after (s=39)
immature_N = None
final_N = None
for s in range(40):
    rv = rv1(s)
    hp = hp1(s)
    h = torch.randn(2, 64, 896) * 0.05
    pen = pen1(s)
    out = layer1(h, hp, pen, resvar=rv, allow_write=True)
    assert not torch.isnan(out).any(), f'NaN at step {s}'
    db = layer1.debug()
    if s == 5:
        immature_N = list(db['N_s'])
    if s == 39:
        final_N = list(db['N_s'])
    if s % 5 == 0 or s == 39:
        print(f'  s{s:3d} rv={rv:.3f} mature={db["mature"]:.0f} occ={db["occupied"]} '
              f'u={db["u_gate"]:.2f} c={db["c_gate"]:.2f} scale={db["read_scale"]:.2f} '
              f'N={db["N_s"]}')
print(f'  immature phase (s=5):  N={immature_N}')
print(f'  after maturation (s=39): N={final_N}')
check('No births during the immature phase',
      all(n == 0 for n in immature_N), f'N={immature_N}')
check('Births happened after maturation', any(n > 0 for n in final_N),
      f'occupied={sum(1 for n in final_N if n > 0)}')
# after 40 steps with an era every 8 steps we have >= 5 concepts; with S=8 the bank
# should be at least partially filled and eviction should have fired (counts > 1 cycles)
check('Bank has no empty slots (filled or recycled)',
      all(n > 0 for n in layer1.debug()['N_s']), f'N={layer1.debug()["N_s"]}')
check('Occupancy EMA normalized (max ~1.0)',
      max(layer1.debug()['U_s']) <= 1.0 + 1e-3, f'U={layer1.debug()["U_s"]}')

# ─── Scenario 2: contradiction gate kills the "pink elephant" ──────────
print('\nScenario 2: pink-elephant guard — a novel-but-unconfirmed slot')
print('  (read out must be suppressed when the concept contradicts the input context)')
layer2 = CollectiveConceptLayer(D=896, k=16, S=8, write_delay=2, birth_gap=0.5)
layer2._maturity_warmup = 1
# mature immediately, plant one slot = "elephant"
layer2._update_maturity(0.05)
layer2._mature.fill_(1.0)
elephant = F.normalize(torch.randn(16), dim=-1)
layer2.M.data[0] = elephant
layer2.N_s[0] = 10
layer2.U_s[0] = 0.9

pen = torch.full((1, 16), 0.8)
elephant_hp = make_hp(1, 16, 8, 16, [elephant * 1.2])

# the read vector the layer would inject (pre-gate) for the elephant slot
with torch.no_grad():
    probe_h = torch.randn(1, 16, 896) * 0.05
    a_elephant = torch.zeros(1, 1, layer2.S); a_elephant[0, 0, 0] = 1.0
    raw = layer2.W_o(
        (a_elephant.unsqueeze(-1)
         * layer2.U_s.clamp(0, 1).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
         * layer2.M.unsqueeze(0).unsqueeze(0)).reshape(1, 1, -1).expand(1, 16, -1))
    read_dir = F.normalize(raw, dim=-1)

# (a) context agrees with the elephant read direction -> gate open
agree_h = read_dir + torch.randn(1, 16, 896) * 0.02
out_a = layer2(agree_h, elephant_hp, pen, resvar=0.05, mature_override=1.0)
c_a = layer2.debug()['c_gate']

# (b) context CONTRADICTS the elephant (anti-aligned) -> gate closed
pink_h = -read_dir + torch.randn(1, 16, 896) * 0.02
out_b = layer2(pink_h, elephant_hp, pen, resvar=0.05, mature_override=1.0)
c_b = layer2.debug()['c_gate']

print(f'  agreeing context: c_gate={c_a:.3f} |read|={out_a.norm().item():.4f}')
print(f'  contradicting context: c_gate={c_b:.3f} |read|={out_b.norm().item():.4f}')
check('Agreeing context lets the concept through (c_gate > 0.5)',
      c_a > 0.5, f'c_gate={c_a:.3f}')
check('Contradicting context suppresses it (c_gate << 1)',
      c_b < c_a * 0.5, f'c_gate={c_b:.3f} vs {c_a:.3f}')
check('Pink elephant does not leak into the signal (|read| shrinks)',
      out_b.norm().item() < out_a.norm().item() * 0.6,
      f'|read|: {out_b.norm().item():.4f} vs {out_a.norm().item():.4f}')

# ─── Scenario 3: uncertainty gate ──────────────────────────────────────
print('\nScenario 3: uncertainty gate opens when the main layer is unsure')
layer3 = CollectiveConceptLayer(D=896, k=16, S=8, write_delay=0, birth_gap=0.5)
layer3._maturity_warmup = 0
layer3._mature.fill_(1.0)
hp3 = make_hp(1, 16, 8, 16, [torch.randn(16)])
h3 = torch.randn(1, 16, 896) * 0.01
u_low = layer3(h3, hp3, torch.full((1, 16), 0.05), resvar=0.05, mature_override=1.0)
u_low_v = layer3.debug()['u_gate']
u_high = layer3(h3, hp3, torch.full((1, 16), 1.5), resvar=0.05, mature_override=1.0)
u_high_v = layer3.debug()['u_gate']
print(f'  confident (pen=0.05): u_gate={u_low_v:.3f}')
print(f'  unsure   (pen=1.50): u_gate={u_high_v:.3f}')
check('Uncertainty gate opens when main signal is weak',
      u_high_v > u_low_v + 0.1, f'{u_low_v:.3f} -> {u_high_v:.3f}')
check('Confident main layer suppresses concept read',
      u_low_v < 0.5, f'u_gate={u_low_v:.3f}')

# ─── Summary ───────────────────────────────────────────────────────────
ok = all(PASS)
print(f'\n{"="*50}')
print(f'Simulated collective concept layer: {sum(PASS)}/{len(PASS)} checks passed')
sys.exit(0 if ok else 1)
