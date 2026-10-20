import torch
import torch.nn as nn

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Exponential Moving Average (EMA) of model parameters.
        Args:
            model (nn.Module): the model containing parameters to track
            decay (float): decay rate for the moving average
        """
        self.decay = decay
        self.ema_params = {}
        
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.ema_params[name] = p.detach().clone()

    @torch.no_grad()
    def copy_from(self, model: nn.Module) -> None:
        """
        Overwrite EMA parameters with the current model parameters.
        Args:
            model (nn.Module): the model containing parameters to copy
        """
        for name, p in model.named_parameters():
            if name in self.ema_params:
                self.ema_params[name].copy_(p.detach())

    @torch.no_grad()
    def step(self, model: nn.Module) -> None:
        """
        update EMA parameters
        Args:
            model (nn.Module): the model containing parameters to update EMA
        """
        for name, p in model.named_parameters():
            if name in self.ema_params:
                ema_p = self.ema_params[name]
                ema_p.lerp_(p, weight=1.0 - self.decay)

    def state_dict(self) -> dict:
        """
        Get the state dictionary of EMA parameters
        Returns:
            dict: state dictionary of EMA parameters
        """
        return {k.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", ""): v.detach() for k, v in self.ema_params.items()}
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load EMA parameters from state dictionary
        Args:
            state_dict (dict): state dictionary of EMA parameters
        """
        for k, v in self.ema_params.items():
            if k.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "") in state_dict:
                v.copy_(state_dict[k.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")])

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        """
        Store current model parameters
        Args:
            model (nn.Module): the model containing parameters to store
        """
        self.temp_params = {
            k: v.detach().clone() 
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """
        Restore stored model parameters
        Args:
            model (nn.Module): the model to restore parameters to
        """
        if getattr(self, 'temp_params', None) is not None:
            model.load_state_dict(self.temp_params, strict=True)
            self.temp_params = None
        
    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """
        Copy EMA parameters to the model
        Args:
            model (nn.Module): the model to copy EMA parameters to
        """
        model.load_state_dict(self.ema_params, strict=False)