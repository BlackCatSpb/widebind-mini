import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from core.concept_layer import CollectiveConceptLayer

torch.manual_seed(0)
cl = CollectiveConceptLayer(D=896, k=16, S=8, write_delay=5, birth_gap=0.4)
cl._maturity_warmup = 3

resvar_series = [0.20, 0.19, 0.17, 0.15, 0.12, 0.10, 0.09, 0.085, 0.08, 0.078, 0.075, 0.072, 0.07, 0.068, 0.065]
for b, rv in enumerate(resvar_series):
    h = torch.randn(2, 64, 896) * 0.1
    hp = torch.randn(2, 64, 8, 16) * 0.1
    if b < 5:
        hp[:, :, :, :4] += 2.0
    elif b < 10:
        hp[:, :, :, 4:8] += 2.0
    pen = torch.rand(2, 64) * 0.3
    out = cl(h, hp, pen, resvar=rv, allow_write=True)
    assert not torch.isnan(out).any(), f'NaN at {b}'
    db = cl.debug()
    print(f'b{b:02d} rv={rv:.3f} mature={db["mature"]:.0f} occ={db["occupied"]} N={db["N_s"]} U={db["U_s"]}')

ok = db['occupied'] >= 2
print('RESULT:', 'OK births after maturity' if ok else 'FAIL no births')
sys.exit(0 if ok else 1)
