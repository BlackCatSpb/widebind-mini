"""
Quick 10k-step test with new architecture defaults.
Run: python test_train.py  (or click the .bat created alongside)
Logs: every 100 steps, diagnostics to stdout, saves at 5k/10k
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

# Override sys.argv before train.py starts
sys.argv = [
    'train.py',
    '--data-dir', './wb',
    '--accum', '4',
    '--private-mem',
    '--max-steps', '10000',
    '--log-interval', '100',
    '--save-interval', '5000',
    '--device', 'cuda',
]

# Run train.py's main block
exec(open('train.py', encoding='utf-8').read())
