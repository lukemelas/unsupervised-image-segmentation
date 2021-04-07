import torch
import torch.nn as nn


class DirectionModel(nn.Module):
    """ Base class for direction models. Given a point in latent space, returns 
        a shift direction d. """
    
    def forward(self, z=None):
        raise NotImplementedError()


class FixedDirection(DirectionModel):
    """ Every point in latent space is shifted in the same (fixed, learned) 
        direction. """

    def __init__(self, shape):
        super().__init__()
        self.d = nn.Parameter(torch.randn(shape) * 0.1)
    
    def forward(self, z=None):
        d = self.d / self.d.norm(dim=1, keepdim=True)
        if z is not None:
            assert z.shape[1:] == d.shape[1:], 'mismatched dimensions'
            d = d.expand(z.shape)
        return d

    def after_train_iter(self):
        """Normalize after training"""
        self.d.data = self.d.data / self.d.data.norm(dim=1, keepdim=True)


class RandomDirection(DirectionModel):
    """ Returns a random direction. """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, z=None, same_across_batch=True):
        if z is None:
            d = torch.randn(1, self.dim)
        elif same_across_batch:
            d = torch.randn(1, self.dim).expand(z.shape)
        else:
            d = torch.randn(z.shape[0], self.dim)
        return d


class LinearModel(DirectionModel):
    """ The shift direction is a linear transformation of the latent z """

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
    
    def forward(self, z):
        d = self.linear(z)
        d = d / d.norm(dim=1, keepdim=True)
        return d


class MLP(DirectionModel):
    """ The shift direction is a nonlinear transformation of the latent z """

    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SELU(), 
            nn.Linear(dim, dim), nn.SELU(), 
            nn.Linear(dim, dim))
    
    def forward(self, z):
        d = self.mlp(z)
        d = d / d.norm(dim=1, keepdim=True)
        return d


class MetaLearningModel(DirectionModel):
    """ Like the optimization-based model, but we backpropagate through the 
        gradient steps during training to find a good starting direction. """
    
    def __init__(self, dim):
        super().__init__()
        raise NotImplementedError()


class OptimizationModel(DirectionModel):
    """ The shift direction is optimized separately for every input latent point z.
        This module has no parameters; it does not require training, as it performs
        optimization on-the-fly. Of course, this means inference is a lot slower. """
    
    def __init__(self, dim, loss_function, num_steps, lr, init_direction=None):
        super().__init__()
        
        # Optimization starts from an initial point
        # If initial direction is not given as input, init near 0
        if init_direction is None:  
            init_direction = torch.randn(1, dim) * 0.001
        self.init_direction = init_direction
        self.init_direction.requires_grad = False

    def forward(self, z):
        super().__init__()
        raise NotImplementedError()

        # # Expand input direction to batch size
        # # (because we the directions will be different)
        # d = self.init_direction.repeat(z.shape[0], 1).clone()

        # # Create optimizer
        # optimizer = torch.optim.SGD(lr=lr)


MODELS = {
    'fixed': FixedDirection,
    'linear': LinearModel,
    'mlp': MLP,
    'metalearning': MetaLearningModel,
    'optimization': OptimizationModel,
    'random': RandomDirection,
}