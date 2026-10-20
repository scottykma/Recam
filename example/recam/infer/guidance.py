"""Flow-matching guidance utilities.

FlowMatchingGuidance supports standard CFG (use_apg=False) and
Adaptive Projected Guidance (APG, use_apg=True).

Inputs/outputs are in velocity-prediction space (v = z1 - z0),
consistent with FlowMatchScheduler. t must be in [0, 1] (sigma).
"""
import torch
import torch.nn.functional as F


class FlowMatchingGuidance:
    def __init__(
        self,
        guidance_scale: float = 5.0,
        use_apg: bool = False,
        eta: float = 0.0,
        norm_threshold: float = 27.0,
        momentum: float = -0.5,
    ):
        self.guidance_scale = guidance_scale
        self.use_apg = use_apg
        self.eta = eta
        self.norm_threshold = norm_threshold
        self.momentum = momentum

        self.momentum_buffer = None
        if momentum != 0:
            self.momentum_buffer = MomentumBuffer(momentum)

    def reset(self):
        """Reset APG momentum before starting an independent denoising run."""
        if self.momentum_buffer is not None:
            self.momentum_buffer.reset()

    def __call__(self, noise_cond, noise_uncond, z_t, t):
        """
        noise_cond/uncond: velocity predictions [B, C, T, H, W]
        z_t:               current latent       [B, C, T, H, W]
        t:                 sigma scalar in [0, 1]
        Returns guided velocity [B, C, T, H, W].
        """
        D_cond   = z_t - t * noise_cond
        D_uncond = z_t - t * noise_uncond

        guided_D = self._apply_apg(D_cond, D_uncond) if self.use_apg else (
            D_uncond + self.guidance_scale * (D_cond - D_uncond)
        )

        guided_velocity = (z_t - guided_D) / (t + 1e-8)
        return guided_velocity.to(dtype=z_t.dtype)

    def _apply_apg(self, D_cond, D_uncond):
        delta_D = D_cond - D_uncond

        if self.momentum_buffer is not None:
            self.momentum_buffer.update(delta_D)
            delta_D = self.momentum_buffer.running_average

        if self.norm_threshold > 0:
            delta_D = self._rescale(delta_D)

        delta_D_parallel, delta_D_orthogonal = self._project(delta_D, D_cond)
        delta_D_adjusted = delta_D_orthogonal + self.eta * delta_D_parallel

        return D_cond + (self.guidance_scale - 1) * delta_D_adjusted

    def _project(self, v0, v1):
        original_dtype = v0.dtype
        v0 = v0.double()
        v1 = v1.double()

        v1_normalized = F.normalize(v1.flatten(1), dim=1).view_as(v1)
        v0_flat = v0.flatten(1)
        v1_flat = v1_normalized.flatten(1)
        dot = (v0_flat * v1_flat).sum(dim=1, keepdim=True)
        v0_parallel    = (dot * v1_flat).view_as(v0)
        v0_orthogonal  = v0 - v0_parallel

        return v0_parallel.to(original_dtype), v0_orthogonal.to(original_dtype)

    def _rescale(self, delta_D):
        original_dtype = delta_D.dtype
        norm = delta_D.norm(p=2, dim=(1, 3, 4), keepdim=True)
        scale = torch.minimum(torch.ones_like(norm), self.norm_threshold / (norm + 1e-8))
        return (delta_D * scale).to(dtype=original_dtype)


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = None

    def update(self, update_value: torch.Tensor):
        if self.running_average is None:
            self.running_average = update_value.clone()
        else:
            self.running_average = update_value + self.momentum * self.running_average

    def reset(self):
        self.running_average = None
