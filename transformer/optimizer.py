import numpy as np


class ScheduledOptimizer(object):
    "A simple wrapper class for learning rate scheduling"
    def __init__(self, optimizer, d_model, n_layers, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0

    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def update_lr(self):
        "Learning rate scheduling per step"
        self.n_current_steps += 1
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])
        new_lr_weighted = np.power(self.d_model / self.n_layers, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps / 10, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            # set a separate lr for the weighted model
            param_group['lr'] = new_lr if param_group['type'] == 'base' else new_lr_weighted
