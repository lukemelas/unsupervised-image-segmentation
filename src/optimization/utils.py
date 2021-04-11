from pathlib import Path
import random
import numpy as np
import torch
import torchvision
import kornia as K


def rgb2gray(img):
    return 0.2989 * img[:, 0] + 0.5870 * img[:, 1] + 0.1140 * img[:, 2]


@torch.no_grad()
def create_grid(G, model, zs, ys, n_imgs=8, rs=[0, 2, 4, 8]):
    """Visualization helper function"""
    img_grid = []
    for i in range(n_imgs):
        z = zs[i]
        y = ys[i] if ys is not None else None
        d = model(z)
        img_row = []
        for r in rs:
            img = G(z + d * r, y=y) if ys is not None else G(z + d * r)
            img = img.detach().squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
            img = np.clip(img, 0, 1)
            img_row.append(img)
        img_row = np.hstack(img_row)
        img_grid.append(img_row)
    img_grid = np.vstack(img_grid)
    return img_grid


def save_checkpoint(model, checkpoint_dir='.', name='latest.pth', **kwargs):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(dict(
        state_dict=model.state_dict(),
        **kwargs
    ), checkpoint_dir / name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def tensor_to_np(t: torch.Tensor):
    t = t.cpu().detach()
    if len(t.shape) == 4:
        t = t[0]
    t = torch.clamp(t * 0.5 + 0.5, min=0, max=1)  # de-normalize
    return np.array(torchvision.transforms.ToPILImage()(t))


def tensor2d_to_np(t: torch.Tensor):
    t = t.cpu().detach()
    t = t.reshape(1, t.shape[-2], t.shape[-1]).repeat(3, 1, 1)
    t = t.permute(1, 2, 0) * 255
    return t.numpy().astype(np.uint8)


def set_requires_grad(module, requires_grad=True):
    for p in module.parameters():
        p.requires_grad = requires_grad
