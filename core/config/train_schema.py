from dataclasses import dataclass, field, fields as _fields, is_dataclass as _is_dataclass
import sys
import typing
from typing import Tuple, Optional

@dataclass
class DataloaderConfig:
    """DataLoader settings."""

    seed: int = 42
    """Random seed for data shuffling and sampling."""
    
    batch_size: int = 128
    """Per-device batch size. (Global batch size = batch_size * world_size)."""

    shuffle: bool = True
    """Whether to shuffle the dataset at the start of every epoch."""

    drop_last: bool = True
    """Whether to drop the last incomplete batch."""

    infinite: bool = True
    """Whether to use an infinite data sampler."""

    num_workers: int = 8
    """Number of subprocesses for data loading."""

    pin_memory: bool = True
    """Whether to pin memory for faster host-to-device transfer."""

    prefetch_factor: int = 2
    """Number of batches to prefetch per DataLoader worker."""

@dataclass
class DataManagerConfig:
    """ A data manager to handle data pre-encoding and prefetching. Neccessary for Context parallelism. """

    pre_encode_batches: int = 1
    """ Number of batches to pre-encode """

    pre_encode_log_every: int = 0
    """Log progress every N batches during pre-encode. 0 disables logging."""


@dataclass
class OnlineFrameCheckConfig:
    """Online first-frame colorfulness gate applied in the dataset at train time.

    The sampled window's first frame is scored by core.data.frame_quality.colorfulness;
    a sample failing the gate is rejected and re-drawn (see SharedMetadataDataset).
    """

    enabled: bool = False
    """Master switch. When False the dataset behaves exactly as before."""

    colorfulness_th: float = 0.2
    """Reject if colorfulness < this. Only catches truly achromatic frames
    (solid black, pure white, greyscale film). At 0.2, normal snow/office
    scenes (cf≈7-10) pass; only near-zero monochrome is rejected."""

    max_resample: int = 10
    """Max index re-draws on rejection before giving up on this __getitem__ call."""



@dataclass
class FreshParamGroupConfig:
    """Optimizer settings for parameters the model initializes from zero."""

    lr: float
    """Absolute learning rate for the from-scratch parameters."""

    weight_decay: float = 0.0
    """Weight decay for the from-scratch parameters."""


@dataclass
class OptimizerConfig:
    """Optimizer algorithm parameters."""

    learning_rate: float = 2e-4
    """Base/Peak learning rate."""

    weight_decay: float = 1e-2
    """L2 regularization coefficient."""

    beta1: float = 0.9
    """Exponential decay rate for the 1st moment estimates."""

    beta2: float = 0.999
    """Exponential decay rate for the 2nd moment estimates."""

    clip_grad: bool = True
    """Whether to clip gradients during training."""

    max_grad_norm: float = 1.0
    """Maximum norm for gradient clipping."""

    skip_outlier_grad: bool = False
    """If True, skip the optimizer step when pre-clip grad_norm exceeds an
    EMA-based adaptive threshold (Open-Sora-Plan style). The skipped step
    zeros gradients without updating params or stepping the LR scheduler,
    preventing momentum-buffer pollution from anomalous batches."""

    skip_outlier_grad_factor: float = 3.0
    """Base σ-multiplier for the skip-outlier threshold:
    threshold = ema_mean + factor * sqrt(world_size) * sqrt(ema_var).
    The sqrt(world_size) correction compensates for FSDP averaging gradients
    across all data-parallel ranks, which reduces the observed std by ~sqrt(N)
    relative to the per-rank norms used in Open-Sora-Plan (where factor=3)."""

    skip_outlier_grad_ema_alpha: float = 0.99
    """EMA decay for ema_mean and ema_var. ~1/(1-α) effective window."""

    skip_outlier_grad_observe_steps: int = 200
    """Pure-observation window after LR warmup ends: EMA samples but does not skip."""

    skip_outlier_grad_reset_steps: int = 3
    """If this many optimizer steps are consecutively skipped, the EMA statistics are
    reset (mean=None, var=0) so the detector re-calibrates from the current gradient
    level. This breaks the deadlock where a corrupted threshold permanently blocks all
    steps. Set to 0 to disable."""

    gradient_accumulation_steps: int = 1
    """Number of micro-steps to accumulate before each optimizer step."""

    gradient_checkpoint_modules: list[str] = field(default_factory=lambda: [])
    """List of module types to apply gradient checkpointing to."""

    fresh_params: Optional[FreshParamGroupConfig] = None
    """Optimizer settings for the parameters the pretrained weights do not
    cover, i.e. the ones the model initializes from zero. None keeps the
    single-group behavior. Which parameters those are is recorded in the
    checkpoint, so a resume restores the same grouping instead of recomputing
    it from a load that no longer happens."""

@dataclass
class LRSchedulerConfig:
    """Learning Rate Scheduler parameters."""

    max_steps : int = 0
    """Total number of training steps."""

    warmup_steps: int = 0
    """Number of steps to linearly warmup the learning rate."""

    decay_steps: int = 0
    """Total steps for the decay schedule."""

    min_lr_factor: float = 0.1
    """Minimum LR as a fraction of the peak LR."""

    decay_type: str = "cosine"
    """Strategy for decay: 'linear', 'sqrt', 'cosine'."""


@dataclass
class LoRAConfig:
    """LoRA adapter settings (using peft).

    Notes on fields:
    - enabled: gate to turn LoRA on/off.
    - pretrained_lora_path: if provided, load adapter weights from this path; otherwise create a fresh adapter.
    - target_modules: modules to apply LoRA to. Use "all-linear" to target every nn.Linear in the model, or a list of module name substrings.
    - task_type: kept as a string to avoid importing peft at config-parse time; default is "FEATURE_EXTRACTION" for generic PyTorch models.
    """

    enabled: bool = False
    """Whether to enable LoRA during training."""

    pretrained_lora_path: Optional[str] = None
    """Path to a pretrained LoRA adapter to load (directory or model file)."""

    lora_r: int = 8
    """LoRA rank (r)."""

    lora_alpha: int = 16
    """LoRA alpha scaling factor."""

    lora_dropout: float = 0.0
    """LoRA dropout probability."""

    lora_target_modules: list[str] | str = "all-linear"
    """Target modules; e.g., ["q_proj", "v_proj"] or "all-linear" for all Linear layers."""

    lora_bias: str = "none"
    """Bias handling for LoRA layers: one of {"none", "all", "lora_only"}."""

    modules_to_save: list[str] | None = None
    """Extra submodules to fully fine-tune (not LoRA-adapt) alongside the LoRA layers,
    e.g. a newly-added compressor that must train from scratch. Passed straight to
    peft.LoraConfig; None keeps existing callers unaffected."""

@dataclass
class EMAConfig:
    """Exponential Moving Average (EMA) configuration."""
    
    enabled: bool = True
    """Whether to maintain an EMA copy of the model weights."""

    decay: float = 0.999
    """Decay rate for the moving average."""

    start_step: int = 0
    """Global step at which EMA tracking should start."""

    update_freq: int = 1
    """Frequency (in steps) to update the EMA weights."""


@dataclass
class ParallelConfig:
    """Distributed strategy configuration (DDP/FSDP/Mesh)."""

    hsdp: bool = False
    """Whether to use Hybrid Sharded Data Parallel (HSDP)."""

    shard_size : int = 8
    """FSDP shard size in bytes. 0 means full sharding. Only used if hsdp is True."""

    zero3: bool = False
    """Whether to use Zero3-like behavior (full parameter sharding)."""

    reshard_after_forward: bool = None
    """(FSDP) Release parameters after forward pass to save memory."""

    wrap_submodules: Optional[list[str]] = field(default_factory=lambda: [])
    """Submodule names to wrap for FSDP/sharding."""

    use_amp: bool = True
    """Whether to use mixed precision training."""

    param_dtype: str = "bfloat16"
    """Data type for model gather type (e.g., 'bfloat16', 'float16')."""

    reduce_dtype: str = "float32"
    """Data type for gradient communication/reduction."""

    cp_size: int = 1
    """Context/Sequence parallelism size (CP)."""

    def __post_init__(self):
        self.reshard_after_forward = self.zero3


@dataclass
class LoggingConfig:
    """Logging settings."""
    
    log_dir: str = ""
    """Directory to save log files and tensorboard summaries."""

    log_file: str = "train.log"
    """Filename for the main log file."""
    
    use_tensorboard: bool = True
    """Whether to use TensorBoard for logging."""

    log_interval: int = 100
    """Number of steps between logging events."""

@dataclass
class CheckpointConfig:
    """Checkpointing settings."""

    dir: str = ""
    """Directory to save model checkpoints."""

    pretrained_path: str = ""
    """Path to a pretrained checkpoint (can be a .pt, .safetensors file or a directory with .pth, .safetensors, .distcp files)."""

    resume_from: str = ""
    """Checkpoint id or path to resume from. Empty keeps the legacy behavior of
    resuming from latest when it exists in checkpoint.dir."""

    force_overwrite_optimizer: bool = False
    """After loading a training checkpoint, overwrite optimizer hyperparameters
    and scheduler base LR from the current config while preserving optimizer
    state such as Adam moments."""

    init_on_meta_device: bool = False
    """Whether to initialize model on 'meta' device before FSDP wrapping."""

    ignore_frozen_params: bool = True
    """Whether to exclude frozen (requires_grad=False) parameters from save/load.
    Set to False to include frozen params in checkpoints (larger files but no need to
    reload pretrained weights on every restart)."""

    consolidated: bool = False
    """Save model/optimizer/ema as a single consolidated .pt on rank0 (full_state_dict
    + cpu_offload) instead of sharded DCP .distcp dirs. Trades a rank0 gather at save
    time for a plain torch.load at load/infer/eval time — no DCP reshard. Default False
    keeps the sharded DCP path for memory-frugal large-scale resume."""

    save_latest_interval: int = 5000
    """Interval (in steps) to save the latest checkpoint."""

    save_checkpoint_interval: int = 5000
    """Interval (in steps) to save regular checkpoints."""


@dataclass
class EvalConfig:
    """Evaluation settings."""

    online_eval: bool = True
    """Whether to perform online evaluation during training."""

    online_eval_interval: int = 500
    """Interval (in steps) to run online evaluation."""

    eval_dir: str = ""
    """Directory to save evaluation outputs (e.g., generated samples)."""

    val_data_path: str = ""
    """Path to val json (caption_units format). If empty, val is skipped."""

    val_num_samples: int = 128
    """Total number of val samples to evaluate each run. Bucketed across n_ref values."""

    val_n_ref_buckets: tuple = (0, 2, 4)
    """n_ref values to bucket val samples by. Each bucket gets val_num_samples / len(buckets)
    samples, yielding eval/loss_n_ref{k} per bucket."""

    val_seed: int = 12345
    """Fixed seed for val noise / timestep sampling — reproducible across steps."""

    val_fixed_timesteps: tuple[float, ...] = ()
    """If non-empty, eval overrides timestep sampling: for each t in list,
    set t_latent=t, t_context=0.5*t. Output keys gain _t{int(t)} suffix."""


@dataclass
class MiscConfig:
    """Miscellaneous settings."""
    
    runs_dir: str = "./runs"
    """Base directory to store all experiment runs."""

    seed: int = 42
    """Random seed for reproducibility."""

    torch_compile: bool = True
    """Whether to use torch.compile for model compilation."""

    eval_only: bool = False
    """If True, run one eval pass and exit before any training step."""

@dataclass
class BaseExpConfig:
    """
    Root Configuration Object.
    Composes all granular configurations into a complete experiment definition.
    """

    exp_id: str = ""
    """Unique experiment identifier string."""

    num_steps: int = 200000
    """Total number of training steps."""

    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    
    data_manager: DataManagerConfig = field(default_factory=DataManagerConfig)

    ema: EMAConfig = field(default_factory=EMAConfig)
    
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    lora: LoRAConfig = field(default_factory=LoRAConfig)
    
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    eval: EvalConfig = field(default_factory=EvalConfig)

    misc: MiscConfig = field(default_factory=MiscConfig)

    def __post_init__(self):
        if self.exp_id:
            if not self.logging.log_dir:
                self.logging.log_dir = f"{self.misc.runs_dir}/{self.exp_id}/logs"
            if not self.checkpoint.dir:
                self.checkpoint.dir = f"{self.misc.runs_dir}/{self.exp_id}/checkpoints"
            if not self.eval.eval_dir:
                self.eval.eval_dir = f"{self.misc.runs_dir}/{self.exp_id}/eval"
        if not self.lr_scheduler.max_steps:
            self.lr_scheduler.max_steps = self.num_steps



def _config_from_dict(cls, data: dict):
    """Recursively construct a nested dataclass from a plain dict (e.g. loaded from YAML)."""
    type_hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in _fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        hint = type_hints.get(f.name, f.type)
        origin = getattr(hint, "__origin__", None)
        if origin is typing.Union:
            args = [a for a in hint.__args__ if a is not type(None)]
            if len(args) == 1:
                hint = args[0]
                origin = getattr(hint, "__origin__", None)
        if _is_dataclass(hint) and isinstance(val, dict):
            val = _config_from_dict(hint, val)
        elif (origin is tuple or hint is tuple) and isinstance(val, list):
            val = tuple(val)
        kwargs[f.name] = val
    return cls(**kwargs)


def build_config_from_cli(config_class):
    """Drop-in replacement for tyro.cli() that also supports --config-file <path>.

    If --config-file is present in sys.argv, the YAML at that path is loaded as
    default values before tyro parses the remaining CLI flags.  Any additional
    CLI flags still override individual fields as usual.

    Example
    -------
    torchrun ... Trainer.py --config-file runs/exp1/logs/config.yaml --exp-id exp2
    """
    import tyro
    import yaml

    argv = sys.argv[1:]
    if "--config-file" in argv:
        idx = argv.index("--config-file")
        yaml_path = argv[idx + 1]
        sys.argv = sys.argv[: idx + 1] + sys.argv[idx + 3 :]
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        default = _config_from_dict(config_class, data)
        stale = (default.logging.log_dir, default.checkpoint.dir, default.eval.eval_dir)
        cfg = tyro.cli(config_class, default=default)
        if cfg.exp_id:
            base = f"{cfg.misc.runs_dir}/{cfg.exp_id}"
            if cfg.logging.log_dir == stale[0]:
                cfg.logging.log_dir = f"{base}/logs"
            if cfg.checkpoint.dir == stale[1]:
                cfg.checkpoint.dir = f"{base}/checkpoints"
            if cfg.eval.eval_dir == stale[2]:
                cfg.eval.eval_dir = f"{base}/eval"
        return cfg
    return tyro.cli(config_class)
