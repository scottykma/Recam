import torch.nn as nn

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

def apply_gradient_checkpointing(
    model: nn.Module, 
    checkpoint_module_type: list[str],
) -> None:
    """
    Apply activation checkpointing to the transformer model.
    Args:
        checkpoint_module_type: Module type to include for checkpointing.
    """
    def non_reentrant_wrapper(module):
        return checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
    
    def selective_checkpointing(submodule):
        return any(
            submodule.__class__.__name__ == module_type for module_type in checkpoint_module_type
        )
    
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=selective_checkpointing,
    )