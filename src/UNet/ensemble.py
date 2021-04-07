""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class Ensemble(nn.Module):
    """An ensemble class"""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        return sum(m(x) for m in self.models) / len(self.models)


if __name__ == '__main__':
    from torchvision.models import resnet18
    model = Ensemble([resnet18(), resnet18()])
    output = model(torch.randn(5, 3, 224, 224))
    print(output.shape)
