import sys, torch
sys.path.insert(0, '.')
from sensitivity_sweep import Simulator

s = Simulator({'div_weight': 50.0, 'alpha_novelty_weight': 0.05, 'gate_repulse_weight': 0.3,
    'ranking_weight': 0.01, 'gate_bias_scale': 1.0, 'balance_weight': 0.026,
    'reinforce_weight': 0.001, 'gate_l1_weight': 0.0001, 'log_scale_l2_weight': 0.01})

checkpoints = {}
for t in range(150000):
    s.step(t)
    if t % 30000 == 0 or t == 0:
        sig_ls = torch.sigmoid(s.ls)
        gate_sig = torch.sigmoid(s.gate_logit)
        checkpoints[t] = {
            'ls_mean': s.ls.mean(dim=-1).tolist(),
            'sig_ls': sig_ls.mean(dim=-1).tolist(),
            'alpha': s.alpha.mean(dim=-1).tolist(),
            'gate_prob': gate_sig.tolist(),
        }

header = 'step  '
for g in range(8):
    header += f'  exp{g}: ls   sig  alp gate  '
print(header)

for step in sorted(checkpoints.keys()):
    cp = checkpoints[step]
    line = f'{step:>5d}  '
    for g in range(8):
        line += f'{cp["ls_mean"][g]:>5.2f} {cp["sig_ls"][g]:>5.3f} {cp["alpha"][g]:>5.3f} {cp["gate_prob"][g]:>5.3f}  '
    print(line)
