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


MODELS = {
    'fixed': FixedDirection,
    'random': RandomDirection,
}
