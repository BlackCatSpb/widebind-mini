from dataclasses import dataclass, field


@dataclass
class SandboxConfig:
    D: int = 896
    n_layers: int = 4
    G: int = 4
    d_mem: int = 64
    seq_len: int = 128
    batch_size: int = 2
    lr: float = 3e-4
    grad_clip: float = 0.5
    accum_steps: int = 1
    vocab: int = 50000
    
    # sandbox
    mem_decay_init: float = 0.5
    act_threshold: float = 0.01
    
    # arbiter
    coh_threshold_init: float = 1.0
    div_threshold_init: float = 0.05
    act_threshold_init: float = 0.1
    maturity_ema: float = 0.999
    
    # sanity
    arch_check: bool = True
