"""WideBand Mini config — compact defaults for local GPU training."""

from dataclasses import dataclass
from .lambda_utils import LambdaConfig


@dataclass
class WideBandConfig:
    D: int = 896
    n_layers: int = 12
    bind_K: int = 32
    vocab: int = 50000
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
    zeckendorf_readout: bool = False

    # Head selector (overrides zeckendorf_readout when non-default)
    #   "partitioned"   → PartitionedHead (linear V-logits + softmax-CE)
    #   "sigmoid_coded" → SigmoidCodedHead (factored Bernoulli, no softmax, O(K) train)
    #   "codec"         → SignedAmpCodec (порт из основного репо): SignedAmpEmbedding +
    #                     SignedAmpHead с CE-целью, W_pred, эхо-каналом (подтверждённый рецепт)
    head_mode: str = "partitioned"

    code_dim: int = 32
    code_sparsity: int = 6

    # ─── SignedAmpCodec (порт из основного репо, рецепт docs/AMPLITUDE_CODEC.md) ───
    amp_codec: bool = False       # True → SignedAmpEmbedding + SignedAmpHead
    amp_pred: bool = True         # механизм A: оператор перехода W_pred (K×K)
    amp_obj: str = 'ce'           # 'ce' — одна CE-цель (подтверждён); 'mh' — margin+hinge
    amp_scale: float = 1.0        # масштаб записи кода в residual stream
    amp_sigma_min: float = 0.2    # нижняя граница σ (коробка [σ_min, 1] на forward)
    amp_gain_init: float = 0.5    # стартовый gain чтения (против насыщения tanh)
    amp_proto_init: float = 0.2   # разброс прототипов амплитуд α_vk
    amp_seed: int = 0             # детерминизм базиса/прототипов/кодов

    # CognitiveCodedHead: normalize token scores over the code space (the only
    # honest fix for the C(K,S)>V calibration leak). False keeps the strict
    # unnormalized factored Bernoulli (miscalibrated by construction).
    head_normalize: bool = True

    mirror_k: int = 32
    mirror_k_staircase: bool = True
    w_pred_scale_init: float = 3.0
    log_scale_init_std: float = 0.05
    mlp_groups: int = 8
    mlp_expand: int = 4

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

    spec_lo: float = 0.5
    spec_hi: float = 1.5
    lambda_sliding: bool = True

    cov_multi_timescale: bool = True
    cov_tau_lo: int = 3
    cov_tau_hi: int = 200

    gate_l1_weight: float = 0.0001
    reinforce_weight: float = 0.001
    balance_weight: float = 0.026  # λ⁻⁶ → HHI-based load balancing (adaptive from here)
    diversity_weight: float = 0.001
    nuclear_weight: float = 1e-5
    orth_weight: float = 1e-4
    pred_w_weight: float = 0.01

    vsa_b_d_max: float = 12.0
    vsa_b_d_smooth: float = 0.999
    vsa_b_lr_mult: float = 0.1

    # BottleneckBind twist: inter-channel bilinear mixing via golden-angle shifts
    bind_twist_mode: str = "shift"        # "off" | "shift" | "cascade"
    bind_twist_S: int = 4                # number of shifts (overridden to 1 when mode=off)
    bind_twist_ocular: str = "tied"      # "tied" | "multi" — per-shift W_out
    bind_twist_scheme: str = "golden"    # "golden" | "fibonacci"
    bind_twist_gate: bool = False        # per-token adaptive aperture via hp

    # ─── Qwen3-inspired upgrades ───
    bind_qk_norm: bool = True            # RMSNorm on hp before bottleneck cross (≈QK-Norm)
    rope_theta: float = 1000000.0        # RoPE base frequency (Qwen3: 1e6)
    rope_scaling: float = 1.0            # RoPE scaling factor (linear)
    mlp_swiglu: bool = True              # SwiGLU gate_proj parallel to up_proj (Qwen3-style)

    accum_steps: int = 1
    compile: bool = False
    div_weight: float = 50.0  # push log_scale variance via -var(sigmoid(ls)) (bounded, self-limiting)
    ranking_weight: float = 0.01  # pairwise order ls_mean by gate_usage (bypasses spectral alignment)
    gate_repulse_weight: float = 0.3  # push gate variance up (inverse of balance, bypasses spectral)
    alpha_novelty_weight: float = 0.05  # push per-expert alpha apart (heuristic, no spectral)
    gate_bias_scale: float = 2.0  # linspace init for gate bias per expert [-scale, scale]
    gate_bias_scale_per_layer: bool = True  # 0.5 (first layer) -> 2.0 (last)
    private_mem: bool = False  # cross-expert private memory bank (meta-cognitive layer)

    # ─── Spec 1: Asymmetric expert init ───
    expert_asymmetry: bool = True  # break symmetry: different alpha, log_scale, W_proj per expert

    # ─── Spec 2: External mirror (auxiliary world model) ───
    aux_mirror_weight: float = 0.0  # 0=off, 0.1=aux cosine loss weight

    # ─── Spec 3: Recursive meta-trust ───
    meta_trust: bool = True  # track trust dynamics, penalize unstable experts (requires private_mem)

    # ─── Spec 4: Collective Concept Layer (experimental) ───
    collective_layer: bool = False
    collective_layer_idx: int = 6      # Mini: L6 (Main: L11)
    collective_S: int = 8              # shared slots
    collective_write_delay: int = 5000 # like private_mem
    collective_uncert_theta: float = 0.5
    collective_uncert_kappa: float = 3.0
    collective_contra_thresh: float = -0.1
    collective_contra_gain: float = 6.0
    collective_birth_gap: float = 0.55
    collective_maturity_thresh: float = 0.12  # mirror resvar EMA below -> layer mature

    log_scale_l2_weight: float = 0.01  # L2 on exp(log_scale) > 10 to prevent gradient explosion

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


# Backward-compat alias for model.py
WideBindConfig = WideBandConfig