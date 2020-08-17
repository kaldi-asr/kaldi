import torch
from torch.optim.optimizer import Optimizer, required


class SgdMaxChange(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum and max 
    change).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        max_change_per_layer (float, optional): change in parameters allowed of
            any given layer, on any given batch, measured in l2 norm
        max_change (float, optional): change in parameters allowed of the whole
            model, after applying the per-layer constraint
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}
        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the 
        parameters, gradient, velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, max_change_per_layer=0.75, max_change=1.5):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if max_change_per_layer < 0.01:
            raise ValueError("Invalid max_change_per_layer value: {}".format(max_change_per_layer))
        if max_change < 0.01:
            raise ValueError("Invalid max_change value: {}".format(max_change))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, 
                        max_change_per_layer=max_change_per_layer, 
                        max_change=max_change)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SgdMaxChange, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SgdMaxChange, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        change = 0

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            max_change_per_layer = group['max_change_per_layer']
            max_change = group['max_change']
            
            delta = []
            total_norm = 0

            for i in range(len(group['params'])):
                p = group['params'][i]
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                norm = d_p.norm(2).item()
                if norm * group['lr'] > max_change_per_layer:
                    d_p.mul_(max_change_per_layer / (norm * group['lr']))
                delta.append(d_p)
                total_norm += d_p.norm(2).item() ** 2.

            total_norm = total_norm ** 0.5

            for i in range(len(group['params'])):
                p = group['params'][i]
                if p.grad is None:
                    continue
                if total_norm * group['lr'] > max_change:
                    p.add_(delta[i], alpha=-group['lr'] * max_change / (total_norm * group['lr']))
                else:
                    p.add_(delta[i], alpha=-group['lr'])

            change += total_norm * group['lr']
        
        return loss, change
