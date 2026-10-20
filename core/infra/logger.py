import os
import sys
import logging
import time
import subprocess
import shutil
from pathlib import Path
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, asdict
from collections import defaultdict

import yaml

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from colorlog import ColoredFormatter

def _inside_git_repo(path: Path) -> bool:
    """Return True if `path` is inside a Git work tree, else False."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        return proc.returncode == 0 and proc.stdout.strip().lower() == "true"
    except Exception:
        return False


def snapshot_experiment(snapshot_dir: Path | None = None, max_size_mb: float = 5.0) -> Path | None:
    """Create a tar snapshot of the current Git work tree, excluding large files.

    If the current working directory is not inside a Git repository, do nothing
    and return None. This avoids failures in environments without Git.
    """
    if snapshot_dir is None:
        snapshot_dir = Path("./code_snapshots")

    if not _inside_git_repo(Path.cwd()):
        return None

    os.makedirs(snapshot_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    tar_name = snapshot_dir / f"code_snapshot_{timestamp}.tar"

    try:
        git_output = subprocess.check_output(
            ["git", "ls-files", "-c", "-m", "-o", "--exclude-standard"],
            text=True,
            stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    max_size_bytes = max_size_mb * 1024 * 1024
    files_to_archive = []

    unique_file_paths = set(git_output.splitlines())
    
    for file_path_str in unique_file_paths:
        if not file_path_str:
            continue
            
        p = Path(file_path_str)
        try:
            if p.is_file() and p.stat().st_size <= max_size_bytes:
                files_to_archive.append(file_path_str)
        except OSError:
            continue

    if not files_to_archive:
        return None

    try:
        with subprocess.Popen(
            ["tar", "-cf", str(tar_name), "-T", "-"], 
            stdin=subprocess.PIPE, 
            text=True
        ) as tar_proc:
            tar_proc.communicate(input="\n".join(files_to_archive) + "\n")
            
        if tar_proc.returncode != 0:
            raise RuntimeError(f"tar command failed with return code {tar_proc.returncode}")
            
    except Exception as e:
        if tar_name.exists():
            tar_name.unlink()
        raise e

    return tar_name
                
class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float | None) -> None:
        if val is None:
            return
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


class ExpLogger:
    def __init__(self, 
                 log_dir: str = "./logs",
                 log_file: str = "train.log",
                 use_tensorboard: bool = True,
                 log_interval: int = 10,
                 device: torch.device = torch.device("cpu") if not torch.cuda.is_available() else torch.cuda.current_device(),
                 rank: int = 0):
        """
        Initialize the experiment logger.
        
        Args:
            log_dir: Directory to save log files and tensorboard events
            log_file: Name of the log file
            use_tensorboard: Whether to use tensorboard for logging
            log_interval: Interval for logging metrics
            device: Device used for training
            rank: Process rank for distributed training
        """
        global GLOBAL_LOGGER
        if GLOBAL_LOGGER is not None:
            raise RuntimeError("ExpLogger is already initialized, use get_logger() to access the global logger.")
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / log_file
        self.use_tensorboard = use_tensorboard
        self.log_interval = log_interval
        self.device = device
        self.rank = rank
        self.is_master = (rank == 0)
        self.separator_width = int(2/3*self._resolve_separator_width())

        os.makedirs(log_dir, exist_ok=True)
        
        self._setup_logging()
        
        if self.use_tensorboard and self.is_master:
            self.tb_writer = SummaryWriter(log_dir=log_dir)
        
        self.meters: dict[str, AverageMeter] = defaultdict(AverageMeter)
        self.step = 0

        self.log_separator()
        self.log_separator()
        self.log_separator()
        self.log_separator()
        self.info(f"Logger initialized with log directory: {self.log_dir}")
        if self.is_master:
            tar_name = snapshot_experiment(self.log_dir / "code_snapshots")
            if tar_name is not None:
                self.info(f"Code snapshot saved at: {tar_name}")
            else:
                self.info("Not a Git repository; skipping code snapshot.")
        
        GLOBAL_LOGGER = self
        self.log_separator()
        
    
    def _setup_logging(self) -> None:
        """
        Setup logging configuration with colorful console output and file logging.
        """
        self.logger = logging.getLogger("experiment_logger")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        console_formatter = ColoredFormatter(
            fmt=f'[%(cyan)s%(asctime)s%(reset)s] %(log_color)s[%(levelname)s] [%(filename)s:%(lineno)d] [r{self.rank}] - %(message)s%(reset)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'purple',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red'
            }
        )
        
        file_formatter = logging.Formatter(
            fmt=f'[%(asctime)s][%(levelname)s] [%(filename)s:%(lineno)d] [r{self.rank}] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)

    def _resolve_separator_width(self) -> int:
        return shutil.get_terminal_size(fallback=(120, 24)).columns

    def _format_section_banner(self, title: str) -> str:
        normalized_title = " ".join(title.strip().split()).upper()
        return f" {normalized_title} ".center(self.separator_width, "=")

    def log_separator(self) -> None:
        self.info("=" * self.separator_width)

    def log_section_start(self, title: str) -> None:
        self.info("")
        self.info(self._format_section_banner(title))

    @contextmanager
    def section(self, title: str):
        self.log_section_start(title)
        try:
            yield
        finally:
            self.log_separator()
    
    def log_config(self, config) -> None:
        """
        Log hyperparameters from a dataclass config.

        Args:
            config: Dataclass containing configuration parameters
        """
        if not self.is_master:
            return

        self.info("Hyperparameters:")

        if is_dataclass(config):
            for field in fields(config):
                value = getattr(config, field.name)
                self.info(f"{field.name}: {value}")
            yaml_path = self.log_dir / "config.yaml"
            with open(yaml_path, "w") as f:
                yaml.safe_dump(asdict(config), f, default_flow_style=False, sort_keys=False)
            self.info(f"Config saved to {yaml_path}")
        else:
            self.info(str(config))
    
    def log_model_info(self, model: torch.nn.Module) -> None:
        """
        Log model and optimizer information.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
        """ 
        self.info("Model Architecture:")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info(f"Model: {model.__class__.__name__}")
        self.info(f"Total parameters: {total_params:,}")
        self.info(f"Trainable parameters: {trainable_params:,}")
    
    def log_optimizer_info(self, optimizer: torch.optim.Optimizer) -> None:
        self.info(f"Optimizer: {optimizer.__class__.__name__}")
        self.info(f"Optimizer parameters: {optimizer.defaults}")

    def log_data_info(self, dataset: torch.utils.data.Dataset, dataloader: torch.utils.data.DataLoader) -> None:
        """
        Log dataset information.
        
        Args:
            dataset: PyTorch dataset
            dataloader: PyTorch dataloader
        """
        if not self.is_master:
            return
        
        self.info(f"Dataset: {dataset.__class__.__name__}")
        if hasattr(dataset, "__len__"):
            self.info(f"Number of samples: {len(dataset)}")
        
        self.info(f"Number of batches: {len(dataloader.batch_sampler)}")
    
    def log_metrics(self, metrics: dict[str, float]) -> None:
        """
        Update metrics to AverageMeter without logging.
        
        Args:
            metrics: Dictionary of metrics to log (e.g. {'loss': 0.5})
        """
        for k, v in metrics.items():
            self.meters[k].update(v)
        

    def log_step(self, step: int) -> None:
        """
        Update metrics to AverageMeter and log at specified intervals.
        
        Args:
            step: Current training step
        """
        self.step = step

        if step % self.log_interval == 0:
            if dist.is_initialized():
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, set(self.meters.keys()))
                all_keys = sorted(set().union(*gathered))

                for key in all_keys:
                    meter = self.meters[key]
                    sum_tensor = torch.tensor(meter.sum, device=self.device, dtype=torch.float32)
                    count_tensor = torch.tensor(meter.count, device=self.device, dtype=torch.int32)
                    dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
                    meter.sum = sum_tensor.item()
                    meter.count = count_tensor.item()
                    meter.avg = meter.sum / meter.count if meter.count > 0 else 0

            if self.is_master:
                if self.use_tensorboard:
                    for key, meter in self.meters.items():
                        if meter.count > 0:
                            self.tb_writer.add_scalar(key, meter.avg, step)
                if torch.cuda.is_available():
                    allocated_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                    reserved_gb = torch.cuda.max_memory_reserved(self.device) / (1024 ** 3)
                    torch.cuda.reset_peak_memory_stats(self.device)
                metrics_str = " | ".join([f"{k}: {m.avg:.8f}" for k, m in sorted(self.meters.items()) if m.count > 0])
                
                memory_str = f" | Mem Alloc: {allocated_gb:.2f} GB | Mem Resv: {reserved_gb:.2f} GB" if torch.cuda.is_available() else ""
                self.info(f"Step {step:08d} | {metrics_str}{memory_str}")
            self.reset_metrics()

    def log_eval(self, metrics: dict[str, float], step: int) -> None:
        """
        Log evaluation metrics.
        
        Args:
            metrics: Dictionary of evaluation metrics
            step: Current training step
        """
        if not self.is_master:
            return
        
        if self.use_tensorboard:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"eval/{key}", value, step)
        
        metrics_str = " | ".join([f"{k}: {v:.8f}" for k, v in metrics.items()])
        self.info(f"Eval at Step {step:08d} | {metrics_str}")
        
    @contextmanager
    def timer(self, tag: str, only_once: bool = False, sync: bool = False):
        """
        Context manager to start a timer with given tag.
        
        Args:
            tag: Timer identifier
            only_once: If True, timer will only be recorded once
            sync: If True, synchronize CUDA before and after timing
        """
        if torch.cuda.is_available() and sync:
            torch.cuda.synchronize(self.device)
        start = time.time()
        yield
        if torch.cuda.is_available() and sync:
            torch.cuda.synchronize(self.device)
        elapsed = time.time() - start
        if only_once:
            self.info(f"Step [{self.step}] timer {tag} finished: {elapsed:.4f} sec")
        else:
            self.meters[f"timer/{tag}"].update(elapsed)
            
    def info(self, msg: str, rank_0_only: bool = True) -> None:
        """
        Log info message.
        
        Args:
            msg: Message to log
            rank_0_only: Whether to log only on rank 0
        """
        if not rank_0_only or self.is_master:
            self.logger.info(msg, stacklevel=2)
    
    def warning(self, msg: str, rank_0_only: bool = True) -> None:
        """
        Log warning message.
        
        Args:
            msg: Message to log
            rank_0_only: Whether to log only on rank 0
        """
        if not rank_0_only or self.is_master:
            self.logger.warning(msg, stacklevel=2)

    def error(self, msg: str, rank_0_only: bool = True) -> None:
        """
        Log error message with stack trace.
        
        Args:
            msg: Message to log
            rank_0_only: Whether to log only on rank 0
        """
        if not rank_0_only or self.is_master:
            self.logger.error(msg, exc_info=True, stacklevel=2)

    def reset_metrics(self) -> None:
        """
        Reset all metrics. Usually called at the beginning of an epoch.
        """
        for meter in self.meters.values():
            meter.reset()

    def purge_tensorboard(self, step: int) -> None:
        """Write a SessionLog.START marker into the existing tfevents file.

        TensorBoard reads events sequentially; upon encountering
        SessionLog.START at step N it discards all previously accumulated
        events with global_step >= N from that file, producing a clean curve.

        The marker MUST go into the same (already-open) writer file — closing
        and reopening creates a second tfevents file which TensorBoard reads
        independently, leaving the stale events in the old file visible and
        producing a broken curve.
        """
        if not (self.use_tensorboard and self.is_master):
            return
        if not hasattr(self, "tb_writer"):
            return
        try:
            import time
            from tensorboard.compat.proto.event_pb2 import Event, SessionLog
            event = Event(
                step=step,
                wall_time=time.time(),
                session_log=SessionLog(status=SessionLog.START),
            )
            self.tb_writer._get_file_writer().add_event(event)
            self.tb_writer.flush()
        except Exception as e:
            self.warning(f"[purge_tensorboard] could not write purge marker: {e}", rank_0_only=True)

    def close(self) -> None:
        """
        Close the logger and tensorboard writer.
        """
        if hasattr(self, 'tb_writer') and self.use_tensorboard:
            self.tb_writer.close()

GLOBAL_LOGGER: ExpLogger = None

def get_logger() -> ExpLogger:
    """
    Get the global experiment logger.
    
    Returns:
        The global ExpLogger instance.
    """
    global GLOBAL_LOGGER
    if GLOBAL_LOGGER is None:
        assert False, "Global logger is not initialized. Please create an ExpLogger instance first."
    return GLOBAL_LOGGER
