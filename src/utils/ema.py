"""
Exponential Moving Average (EMA) for model parameters.

Based on implementation from: https://github.com/Alibaba-MIIL/ASL/blob/main/train.py
"""
from copy import deepcopy
import torch
import torch.nn as nn


class ModelEma(nn.Module):
    """Model Exponential Moving Average.
    
    Maintain a moving average of model parameters. This helps stabilize training
    and often leads to better final performance.
    
    Args:
        model: The model to create EMA for.
        decay: Decay factor for EMA (default: 0.9997).
        device: Device to perform EMA on. If None, uses same device as model.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9997, device=None):
        super(ModelEma, self).__init__()
        # Make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        self.decay = decay
        self.device = device  # Perform EMA on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model: nn.Module, update_fn):
        """Internal method to update EMA parameters.
        
        Args:
            model: The source model to update from.
            update_fn: Function to compute new EMA value: (ema_val, model_val) -> new_val
        """
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model: nn.Module):
        """Update EMA parameters using exponential moving average.
        
        Args:
            model: The source model to update from.
        """
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model: nn.Module):
        """Directly set EMA parameters to match the model (for initialization).
        
        Args:
            model: The source model to copy from.
        """
        self._update(model, update_fn=lambda e, m: m)

