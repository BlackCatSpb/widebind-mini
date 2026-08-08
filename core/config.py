"""WideBand Mini config — compact defaults for local GPU training."""

from dataclasses import dataclass
from .lambda_utils import LambdaConfig


@dataclass
class WideBandConfig:
    D: int = 896
    n_layers: int = 12
    bind_K: int = 32
    vocab: int = 65536
    seq_len: int = 512
    batch_size: int = 2
    lr: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 0.5
    dtype: str = 'float32'

    lambda_d: int = 3
    lambda_d_enabled: bool = True

    tie_bind: bool = True
    tie_mirror_proj: bool = True

    head_mode: str = "sigmoid_coded"
    head_normalize: bool = True
    code_dim: int = 32
    code_sparsity: int = 6

    mirror_k: int = 32
    mirror_k_staircase: bool = True
    w_pred_scale_init: float = 3.0
    log_scale_init_std: float = 0.05
    mlp_groups: int = 8
    mlp_expand: int = 4
    private_mem: bool = True

    expert_asymmetry: bool = True
    meta_trust: bool = True

    collective_layer: bool = True
    collective_layer_idx: int = 6
    collective_read_out: bool = True
    collective_S: int = 8
    collective_uncert_theta: float = 0.5
    collective_uncert_kappa: float = 3.0
    collective_contra_thresh: float = -0.1
    collective_contra_gain: float = 6.0
    collective_birth_gap: float = 0.55
    collective_maturity_thresh: float = 0.12

    log_scale_l2_weight: float = 0.01
    div_weight: float = 50.0
    ranking_weight: float = 0.01
    gate_repulse_weight: float = 0.3
    alpha_novelty_weight: float = 0.05
    gate_bias_scale: float = 2.0
    gate_bias_scale_per_layer: bool = True

    scheduler: str = 'mirror'
    target_var: float = 0.1
    mag_threshold: float = 0.3
    lr_min_ratio: float = 0.05
    max_decay_steps: int = 50000
    var_min_for_lr_decay: float = 0.005

    exploration_threshold: float = 0.25
    differentiation_threshold: float = 0.08
    w_mem2v_scale_min: float = 0.5
    w_mem2v_scale_max: float = 1.0
    ema_alpha_min: float = 0.90
    ema_alpha_max: float = 0.99
    noise_scale_min: float = 0.001
    noise_scale_max: float = 0.05
    delta_var_ema_min: float = 0.80
    delta_var_ema_max: float = 0.99

    gate_lr_mult: float = 5.0
    lambda_lr_hierarchy: bool = True

    w_m2v_hierarchy_target: float = 1.0
    w_m2v_hierarchy_weight: float = 0.001

    w_d_init_std: float = 0.1
    conv_init_std: float = 0.01
    conv_kernel: int = 48

    bind_twist_mode: str = "trajectory_spiral"
    bind_twist_S: int = 4
    bind_traj_dims: int = 3
    hybrid_alpha_max: float = 0.7
    hybrid_alpha_min: float = 0.3
    bind_twist_ocular: str = "tied"
    bind_twist_scheme: str = "golden"
    bind_twist_gate: bool = False

    bind_qk_norm: bool = True
    rope_theta: float = 1000000.0
    rope_scaling: float = 1.0
    mlp_swiglu: bool = True

    # Variable Precision Memory
    variable_precision: bool = False
    precision_threshold: float = 0.3

    # Explicit Reasoning
    explicit_reasoning: bool = False
    reasoning_max_steps: int = 8

    surprisal_weight: float = 0.0
    branch_balance_weight: float = 0.0

    accum_steps: int = 1
    compile: bool = False

    gate_l1_weight: float = 0.0001
    reinforce_weight: float = 0.001
    balance_weight: float = 0.026
    diversity_weight: float = 0.001
    nuclear_weight: float = 1e-5
    orth_weight: float = 1e-4

    cov_multi_timescale: bool = True
    cov_tau_lo: int = 3
    cov_tau_hi: int = 200

    vsa_b_d_max: float = 12.0
    vsa_b_d_smooth: float = 0.999
    vsa_b_lr_mult: float = 0.1

    max_steps: int = 300000
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 2000
    patience: int = 999999
    resume: str = ''

    data_dir: str = ''
    save_dir: str = 'checkpoints'
    log_dir: str = 'logs'

    def __post_init__(self):
        if self.lambda_d_enabled:
            self._apply_lambda_d()

    def _apply_lambda_d(self):
        lc = LambdaConfig(self.lambda_d)
        self.warmup_steps = lc.warmup_steps
        self.target_var = lc.target_var
        self.mag_threshold = lc.mag_threshold
        self.lr_min_ratio = lc.lr_min_ratio
        self.max_decay_steps = lc.max_decay_steps
        self.var_min_for_lr_decay = lc.var_min_for_lr_decay
        self.exploration_threshold = lc.exploration_threshold
        self.differentiation_threshold = lc.differentiation_threshold
        self.w_mem2v_scale_min = lc.mem2v_scale_min
        self.w_mem2v_scale_max = lc.mem2v_scale_max
        self.ema_alpha_min = lc.ema_alpha_min
        self.ema_alpha_max = lc.ema_alpha_max
        self.noise_scale_min = lc.noise_scale_min
        self.noise_scale_max = lc.noise_scale_max
        self.delta_var_ema_min = lc.delta_var_ema_min
        self.delta_var_ema_max = lc.delta_var_ema_max
        self.gate_lr_mult = lc.gate_lr_mult
        self.log_scale_init_std = lc.log_scale_init_std
        self.conv_init_std = lc.conv_init_std
        self.w_d_init_std = lc.w_d_init_std
        self.log_interval = lc.log_interval
        self.eval_interval = lc.eval_interval
        self.save_interval = lc.save_interval
        self.patience = lc.patience


WideBindConfig = WideBandConfig
