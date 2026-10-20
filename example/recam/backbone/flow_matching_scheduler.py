import torch
import math
from enum import Enum
from typing import Optional, Union, Tuple

class SNRType(Enum):
    UNIFORM = "uniform"
    LOGNORM = "lognorm"
    MODE = "mode"

class FlowMatchScheduler:
    """
    A scheduler implementing Flow Matching noise schedule and related methods.
    """
    TRAIN_EPS = 1e-5

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        snr_type: Union[str, SNRType] = SNRType.LOGNORM,
        min_timestep: float = 0.0,
        max_timestep: Optional[float] = None,
        shift: float = 3.0,
        lognorm_mu: float = 0.0,
        device: Optional[torch.device] = None
    ):
        self.num_train_timesteps = num_train_timesteps
        self.lognorm_mu = lognorm_mu
        self.snr_type = SNRType(snr_type) if isinstance(snr_type, str) else snr_type
        self.shift = shift
        self.device = device
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep if max_timestep is not None else 1.0

    def sample_train_timesteps(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        lognorm_mu: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Sample training timesteps based on the precomputed probability table
        """
        mu = self.lognorm_mu if lognorm_mu is None else lognorm_mu
        t0 = self.TRAIN_EPS
        t1 = 1.0 - self.TRAIN_EPS
        accepted_t = []
        num_accepted = 0

        while num_accepted < batch_size:
            num_to_sample = batch_size * 4
            if self.snr_type == SNRType.UNIFORM:
                t = torch.rand((num_to_sample,), device=self.device, generator=generator) * (t1 - t0) + t0

            elif self.snr_type == SNRType.LOGNORM:
                u = torch.normal(mean=mu, std=1.0, size=(num_to_sample,), device=self.device, generator=generator)
                t = 1.0 / (1.0 + torch.exp(-u)) * (t1 - t0) + t0
            
            elif self.snr_type == SNRType.MODE:
                mode_scale = 1.29
                u = torch.rand(size=(num_to_sample,), device=self.device, generator=generator)
                t = 1.0 - u - mode_scale * (torch.cos(math.pi * u / 2.0) ** 2 - 1.0 + u)
                t = t * (t1 - t0) + t0
        
            timesteps = self.shift * t / (1 + (self.shift - 1) * t)
            mask = (timesteps >= self.min_timestep) & (timesteps <= self.max_timestep)
            valid_t = timesteps[mask]
            accepted_t.append(valid_t)
            num_accepted += valid_t.shape[0]

        accepted_t = torch.cat(accepted_t)[:batch_size]
        return accepted_t * self.num_train_timesteps
    
    def training_target(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return x1 - x0
    
    def add_noise(self, x0: torch.Tensor, x1: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t = timesteps / self.num_train_timesteps
        t = t.view(-1, *([1] * (len(x0.shape) - 1)))
        return (1 - t) * x0 + t * x1

    def set_timesteps(self, num_steps: int, shift: Optional[float] = 1.0) -> torch.Tensor:
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        if shift != 1.0: 
            self.shift = shift
            timesteps = self.shift * timesteps / (1 + (self.shift - 1) * timesteps)
        self.timesteps = timesteps * self.num_train_timesteps
        return self.timesteps

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, next_timestep: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        dt = (next_timestep - timestep) / self.num_train_timesteps
        return sample + model_output * dt
