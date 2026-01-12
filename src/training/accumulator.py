"""
Gradient Accumulation and Mixed Precision Wrapper.

This module defines the `AmpGrad` utility class. It abstracts away the complexity of:
- Gradient Accumulation: Stepping the optimizer only after N batches.
- Automatic Mixed Precision (AMP): Managing the GradScaler for fp16 training on CUDA devices.
- Providing unified methods for `backward()`, `step()`, and `zero_grad()`.

Classes:
    AmpGrad: Wrapper around optimizer and GradScaler.

Usage Example:
    ```python
    optimizer = torch.optim.AdamW(model.parameters())
    amp_grad = AmpGrad(optimizer, accum=4, amp=True)
    
    # In training loop:
    loss = model(input)
    amp_grad.backward(loss)
    
    if amp_grad.should_step():
        amp_grad.step()
        amp_grad.zero_grad()
    ```
"""
import torch

class AmpGrad:
    def __init__(
        self, 
        optimizer, 
        accum: int = 1,
        amp: bool = True,
        ):
        
        assert torch.cuda.is_available(), "AMP Training only works with NVIDIA GPUs, use them or disable it."
        
        self.optim = optimizer
        self.accum = max(1, accum)
        self.amp = amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self._n = 0
        
    def backward(self, loss: torch.Tensor):
        
        loss = loss / self.accum
        
        if self.amp:
            self.scaler.scale(loss).backward()
            
        else:
            loss.backward()
            
        self._n += 1
        
    def should_step(self):
        return (self._n % self.accum) == 0
    
    def unscale_(self): 
        if self.amp:
            self.scaler.unscale_(self.optim)
    
    def step(self):
        if self.amp:
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.optim.step()
            
    def zero_grad(self):
        self.optim.zero_grad(set_to_none=True)
        