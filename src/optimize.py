"""
Example usage:
>>> python optimize.py losses.outside_circle_loss=2.0 losses.sobel_shift_loss=10.0
>>> streamlit run optimize.py losses.outside_circle_loss=2.0 losses.sobel_shift_loss=10.0
"""
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from PIL import Image
import kornia as K
from omegaconf import OmegaConf

from models import MODELS
import optimization_utils as utils
from optimization_utils import rgb2gray
from inversion.gans import make_gan

import streamlit as st


class UnsupervisedSegmentationLoss(torch.nn.Module):

    def __init__(self, loss_weights={}, image_size=None):
        super().__init__()
        self.loss_weights = {n: w for n, w in loss_weights.items() if abs(w) > 1e-5}
        self.image_size = 128 if image_size is None else image_size

        # Precompute outside masks to save computation
        if 'outside_square_loss' in self.loss_weights:
            border = {128: 20, 256: 38, 512: 75}[self.image_size]
            square = torch.ones(1, 1, self.image_size, self.image_size)
            square[:, :, border:-border, border:-border].zero_()
            square = 2 * square - 1
            self.register_buffer('outside_square', square)
        if 'outside_circle_loss' in self.loss_weights:
            x_axis = torch.linspace(-1, 1, self.image_size).unsqueeze(1)
            y_axis = torch.linspace(-1, 1, self.image_size).unsqueeze(0)
            circle = torch.sqrt(x_axis ** 2 + y_axis ** 2)
            circle = (circle - circle.mean()) * 4  # magic number
            self.register_buffer('outside_circle', circle)

    @staticmethod
    def denormalize(img):
        """Helper function to turn images from [-1,1] to [0,1]"""
        return (img * 0.5) + 0.5

    def lightness_loss(self, img, *cfg):
        """Lightness variance loss"""
        batch_size = img.shape[0]
        img_gray = rgb2gray(img).reshape(batch_size, -1)  # RGB --> Grayscale
        img_gray = img_gray - img_gray.mean(dim=1, keepdim=True)  # subtract mean
        return - torch.mean(img_gray ** 2)  # encourage large values

    def outside_square_loss(self, img, *cfg):
        return torch.mean(rgb2gray(img) * self.outside_square)

    def outside_circle_loss(self, img, *cfg):
        return torch.mean(rgb2gray(img) * self.outside_circle)

    def sobel_shift_loss(self, img, img_shifted):
        """Edge pairwise loss"""
        edge = K.sobel(rgb2gray(img).unsqueeze(0))  # extract edges
        edge_shifted = K.sobel(rgb2gray(img_shifted).unsqueeze(0))  # extract edges
        return torch.mean((edge - edge_shifted) ** 2)  # encourage edges to be the same

    def sobel_difference_loss(self, img, img_shifted):
        edge = K.sobel(rgb2gray(img).unsqueeze(0))  # extract edges
        edge_diff = K.sobel(rgb2gray(img - img_shifted).unsqueeze(0))
        return torch.mean((edge - edge_diff) ** 2)  # encourage edges to be the same

    def forward(self, img, img_shifted):
        losses = {}
        for name, weight in self.loss_weights.items():
            losses[name] = getattr(self, name)(img, img_shifted) * weight
        losses['loss'] = sum(losses.values())
        return losses


def forward_generate(*, G, z, r, d):
    """ Helper function because conditional and unconditional models need 
        to be dealt with slightly differently"""
    if hasattr(G, 'conditional') and G.conditional:
        with torch.no_grad():
            img_original, y = G(z, return_y=True)
        img_shifted = G(z + r * d, y=y)
    else:
        with torch.no_grad():
            img_original, y = G(z), None
        img_shifted = G(z + r * d)
    return img_original, img_shifted


def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    # Device
    device = torch.device('cuda')

    # Load GAN
    G = make_gan(gan_type=cfg.generator.gan_type, **cfg.generator.kwargs)
    G.eval().to(device)
    utils.set_requires_grad(G, False)

    # Seed
    utils.set_seed(cfg.seed)

    # Model
    init_shape = G.sample_latent(batch_size=1, device=device).shape
    model = MODELS[cfg.model_type](shape=init_shape)
    model = model.to(device)

    # Create optimizer
    Optimizer = getattr(torch.optim, cfg.optimizer.cls)
    optimizer = Optimizer(model.parameters(), **cfg.optimizer.kwargs)

    # Create scheduler
    Scheduler = getattr(torch.optim.lr_scheduler, cfg.scheduler.cls)
    scheduler = Scheduler(optimizer, **cfg.scheduler.kwargs)

    # Loss function
    criterion = UnsupervisedSegmentationLoss(cfg.losses, image_size=cfg.generator.image_size)
    criterion.to(device)

    # Logging, checkpointing, tensorboard
    log_dir = Path(f'logs/{cfg.model_type}/{cfg.generator.gan_type.lower()}/{cfg.name}')
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cfg, log_dir / 'cfg.pth')
    with open(log_dir / 'config.yaml', 'w') as f:
        print(OmegaConf.to_yaml(cfg), file=f)
    writer = SummaryWriter(log_dir=log_dir / 'tensorboard')

    # Fixed vectors for visualization
    z_vis_fixed = G.sample_latent(batch_size=8, device=device).unsqueeze(1).requires_grad_(False)  # 8 vis images
    y_vis_fixed = G.sample_class(batch_size=8, device=device).unsqueeze(1).requires_grad_(
        False) if hasattr(G, 'sample_class') else None

    # Optimize
    start_time = time.time()
    progress_bar = trange(1, cfg.scheduler.iterations + 1)
    for i in progress_bar:

        # Sample random latent variable
        z = G.sample_latent(cfg.batch_size, device=device).requires_grad_(False)

        # Sample shift amount
        r = torch.randn_like(z)
        r = torch.clip(r * cfg.r.var + cfg.r.mean, min=cfg.r.min)

        # Forward: latent to direction
        d = model(z)

        # Generate
        img_original, img_shifted = forward_generate(G=G, z=z, r=r, d=d)

        # Loss
        losses = criterion(img_shifted, img_original)
        losses['loss'].backward()

        # Step
        optimizer.step()
        optimizer.zero_grad()

        # Projection to unit norm
        if hasattr(model, 'after_train_iter'):
            model.after_train_iter()

        # Learning rate scheduler
        if scheduler:
            scheduler.step()

        # Log
        log_message = f"Iteration {i:5d} | Time {time.time() - start_time:.1f}"
        for k, v in sorted(losses.items()):
            writer.add_scalar(f'train/{k}', v.item(), global_step=i)
            log_message += f' | {k}: {v.item():.3f}'
        if i % cfg.log_every == 0:
            progress_bar.write(log_message)

        # Visualize with Tensorboard, Streamlit and save to file
        if i % cfg.vis_every == 0:
            img_grid = utils.create_grid(G=G, model=model, zs=z_vis_fixed, ys=y_vis_fixed, n_imgs=8)
            writer.add_image(f'vis', img_grid, global_step=i, dataformats='HWC')
            st.image(img_grid, caption=i)
            img_file = log_dir / "imgs" / f"{i}.png"
            img_file.parent.mkdir(parents=True, exist_ok=True)
            img_grid = (img_grid * 255).astype(np.uint8)
            Image.fromarray(img_grid).save(str(img_file))

    # Save checkpoint after iterations are complete
    kwargs = dict(log_dir=log_dir, iteration=i, cfg=cfg)
    utils.save_checkpoint(model, checkpoint_dir, name='latest.pth', **kwargs)

    # Return final model
    return model


if __name__ == '__main__':
    cfg = OmegaConf.merge(OmegaConf.load('config/optimize.yaml'), OmegaConf.from_cli())
    main(cfg)
