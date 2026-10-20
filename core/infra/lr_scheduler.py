import math
from torch.optim.lr_scheduler import LambdaLR

class WarmupStableDecayLR(LambdaLR):
    """
    A unified learning rate scheduler that combines:
    1. Linear warmup from 0 to 1 over `warmup_steps`.
    2. A stable phase maintaining the peak learning rate for `stable_steps`.
    3. A configurable decay phase (linear, sqrt, cosine) over `decay_steps`.

    The actual learning rate is determined by multiplying the base learning rate
    by a factor computed by the `linear_warmup_stable_decay` function.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        lr_decay_type: str = "linear",
        min_lr_factor: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Initializes the scheduler.

        Args:
            optimizer: The optimizer to schedule.
            warmup_steps: Number of steps for the linear warmup phase.
            stable_steps: Number of steps to maintain the peak learning rate.
            decay_steps: Number of steps for the decay phase.
            lr_decay_type: The type of decay ("linear", "sqrt", "cosine").
            min_lr_factor: The minimum learning rate factor (e.g., 0.1 for 10% of peak).
            last_epoch: The index of the last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.lr_decay_type = lr_decay_type
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer, self._get_lr_factor, last_epoch)

    def _get_lr_factor(self, current_step: int):
        """
        Computes the multiplicative factor for the learning rate based on the current step.
        This function implements the logic described in the reference.
        """
        warmup_stable_steps = self.warmup_steps + self.stable_steps
        total_steps = warmup_stable_steps + self.decay_steps
        if current_step < self.warmup_steps:
            current_step_calc = current_step + 1
            assert (
                self.warmup_steps != 0
            ), "warmup_steps must not be zero to reach this branch"
            curr_adjustment = float(current_step_calc / self.warmup_steps)
        elif current_step < warmup_stable_steps:
            curr_adjustment = 1.0
        elif current_step < total_steps:
            current_step_calc = current_step + 1
            assert self.decay_steps != 0, "decay_steps must not be zero to reach this branch"
            progress = float(current_step_calc - warmup_stable_steps) / self.decay_steps

            if self.lr_decay_type == "linear":
                curr_adjustment = 1 - progress
            elif self.lr_decay_type == "sqrt":
                curr_adjustment = 1 - math.sqrt(progress)
            elif self.lr_decay_type == "cosine":
                 curr_adjustment = 0.5 * (1.0 + math.cos(math.pi * progress))
            else:
                raise ValueError(f"Unsupported lr_decay_type: {self.lr_decay_type}. "
                                 "Choose from 'linear', 'sqrt', 'cosine'.")

            curr_adjustment = self.min_lr_factor + (1 - self.min_lr_factor) * curr_adjustment
        else:
            curr_adjustment = self.min_lr_factor
        return curr_adjustment

def build_lr_scheduler(
    optimizer,
    max_steps: int,
    warmup_steps: int,
    decay_steps: int,
    min_lr_factor: float,
    lr_decay_type: str = "cosine"
):
    """
    Builds a WarmupStableDecayLR scheduler instance.

    This function calculates the stable phase duration based on the provided
    total max_steps, warmup_steps, and decay_steps. The stable phase is the
    time between the end of warmup and the start of decay, which occurs in the
    final decay_steps of the training run.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of steps for the linear warmup phase.
        decay_steps: Number of steps for the decay phase at the end.
        max_steps: Total number of training steps.
        min_lr_factor: The minimum learning rate factor (e.g., 0.1 for 10% of peak).
        lr_decay_type: The type of decay ("linear", "sqrt", "cosine"). Default: "cosine".
        verbose: If True, prints a message to stdout for each update. Default: False.

    Returns:
        An instance of WarmupStableDecayLR.
    """
    stable_steps = max_steps - warmup_steps - decay_steps

    if stable_steps < 0:
        raise ValueError(
            f"Invalid schedule: warmup_steps ({warmup_steps}) + decay_steps ({decay_steps}) "
            f"must be less than or equal to max_steps ({max_steps}). "
            f"Calculated stable_steps would be {stable_steps}."
        )

    return WarmupStableDecayLR(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        lr_decay_type=lr_decay_type,
        min_lr_factor=min_lr_factor
    )
