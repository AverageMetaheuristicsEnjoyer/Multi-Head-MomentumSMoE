import torch
from torch.optim import Optimizer

class Signum(Optimizer):
    r"""Implements Signum optimizer that takes the sign of gradient or momentum.

    See details in the original paper at:https://arxiv.org/abs/1711.05101

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0.9)
        weight_decay (float, optional): weight decay (default: 0)

    Example:
        >>> optimizer = signum.Signum(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    .. note::
        The optimizer updates the weight by:
            buf = momentum * buf + (1-momentum)*rescaled_grad
            weight = (1 - lr * weight_decay) * weight - lr * sign(buf)

        Considering the specific case of Momentum, the update Signum can be written as

        .. math::
                \begin{split}g_t = \nabla J(W_{t-1})\\
			    m_t = \beta m_{t-1} + (1 - \beta) g_t\\
				W_t = W_{t-1} - \eta_t \text{sign}(m_t)}\end{split}

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        If do not consider Momentum, the update Sigsgd can be written as

        .. math::
            	g_t = \nabla J(W_{t-1})\\
				W_t = W_{t-1} - \eta_t \text{sign}(g_t)}

    """
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0, **kwargs):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                    else:
                        buf = param_state["momentum_buffer"]

                    buf.mul_(momentum).add_((1 - momentum), d_p)
                    d_p = torch.sign(buf)
                else:
                    d_p = torch.sign(d_p)

                p.data.add_(-group["lr"], d_p)

        return loss