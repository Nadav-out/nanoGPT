import torch
class PAdam(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, lambda_p=1e-2, p_norm=1, *args, **kwargs):
        super(PAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=2e-3, *args, **kwargs)
        self.p_norm = p_norm
        self.lambda_p = lambda_p
    

    @torch.no_grad()
    def step(self, closure=None):
        # Store the old params
        old_params = []
        for group in self.param_groups:
            old_params.append({param: param.data.clone() for param in group['params'] if param.grad is not None})

        # Perform the standard AdamW step
        loss = super(PAdam, self).step(closure)

        # Perform the PAdam step
        for group, old_group in zip(self.param_groups, old_params):
            lambda_p_group = group.get('lambda_p', self.lambda_p)  # Use group-specific lambda_p or default to global lambda_p
            if lambda_p_group > 0:  # Apply the regularization only if lambda_p is greater than 0
                for param in group['params']:
                    if param.grad is None:
                        continue

                    # Use old parameters in the decay factor
                    param_old = old_group[param]
                    X = (param_old.abs()+1e-14)**(2 - self.p_norm)
                    update_term = X / (X + self.p_norm * group['lr'] * lambda_p_group)

                    # Update the parameters
                    param.data.mul_(update_term)

        return loss

class Adam_L1(torch.optim.AdamW):
    def __init__(self, params, l1_lambda=0.01,weight_decay=0, *args, **kwargs):
        super(Adam_L1, self).__init__(params,weight_decay=0, *args, **kwargs)
        self.l1_lambda = l1_lambda
        self.WD=weight_decay

    
    @torch.no_grad()
    def step(self, closure=None):
        # Standard AdamW optimization step
        loss = super(Adam_L1, self).step(closure)

        # Apply L1 regularization per group
        for group in self.param_groups:
            lr = group['lr']
            l1_lambda = group.get('l1_lambda', 0)  # Use group-specific l1_lambda or default to 0
            for p in group['params']:
                if p.grad is not None:
                    # Apply soft thresholding for L1 regularization
                    p.data = torch.sign(p.data) * torch.clamp(torch.abs(p.data) - l1_lambda * lr, min=0)
                    # Apply weight decay (L2 regularization)
                    p.data /= (1 + lr * self.WD)

        return loss

class AdamP(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 lambda_p=0, amsgrad=False, p_norm=2):
        # Initialize the parent class (Adam) with weight_decay set to 0
        super(AdamP, self).__init__(params, lr=lr, betas=betas, eps=eps,
                                     weight_decay=0, amsgrad=amsgrad)
        self.lambda_p = lambda_p
        self.p_norm = p_norm

    def step(self, closure=None):
        # Compute the loss using the parent class's step method if a closure is provided
        loss = None
        if closure is not None:
            loss = closure()

        # Apply custom Lp^p regularization to the gradients
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                # Apply the general Lp^p regularization
                if self.lambda_p != 0:
                    lp_grad = (param.data.abs()**(self.p_norm - 2)) * param.data
                    param.grad.data.add_(lp_grad, alpha=self.p_norm * self.lambda_p)

        # Perform the optimization step using the parent class's logic
        super(AdamP, self).step(closure)

        return loss


class CustomAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(CustomAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1

                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
