import torch
from torch.optim import Optimizer
import torch.distributed as dist


class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) optimizer.
    
    Implements LARS optimization with support for layerwise adaptation and trust coefficient
    as described in "Large Batch Training of Convolutional Networks":
    https://arxiv.org/abs/1708.03888
    """
    def __init__(self, 
                 params,
                 lr=1e-3,
                 momentum=0.9,
                 weight_decay=0.0,
                 trust_coefficient=0.001,
                 eps=1e-8,
                 nesterov=False):
        """
        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
            lr (float): Base learning rate
            momentum (float): Momentum factor
            weight_decay (float): Weight decay (L2 penalty)
            trust_coefficient (float): Trust coefficient for computing adaptive lr
            eps (float): Small constant for numerical stability
            nesterov (bool): Enables Nesterov momentum
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if trust_coefficient < 0.0:
            raise ValueError(f"Invalid trust_coefficient value: {trust_coefficient}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            trust_coefficient=trust_coefficient,
            eps=eps,
            nesterov=nesterov,
            lars_exclude=False  # For excluding certain layers from LARS adaptation
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            trust_coefficient = group['trust_coefficient']
            eps = group['eps']
            nesterov = group['nesterov']
            lars_exclude = group['lars_exclude']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad
                
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if not lars_exclude:
                    # Compute local learning rate
                    w_norm = torch.norm(p.data)
                    g_norm = torch.norm(d_p)
                    
                    # Safeguard for numerical stability
                    if w_norm > eps and g_norm > eps:
                        local_lr = trust_coefficient * w_norm / (g_norm + weight_decay * w_norm + eps)
                    else:
                        local_lr = 1.0
                    
                    d_p = d_p.mul(local_lr)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss
