import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import kornia as K
from tqdm import trange, tqdm
import hydra
from omegaconf import OmegaConf, DictConfig

from models.latent_shift_model import MODELS
from optimization import utils

from pytorch_pretrained_gans import make_gan


class UnsupervisedSegmentationLoss(torch.nn.Module):

    def __init__(self, loss_weights={}, image_size=None):
        super().__init__()
        self.loss_weights = {n: w for n, w in loss_weights.items() if abs(w) > 1e-5}
        self.image_size = 128 if image_size is None else image_size

        if 'outside_circle_loss' in self.loss_weights:
            x_axis = torch.linspace(-1, 1, self.image_size).unsqueeze(1)
            y_axis = torch.linspace(-1, 1, self.image_size).unsqueeze(0)
            circle = torch.sqrt(x_axis ** 2 + y_axis ** 2)
            circle = (circle - circle.mean()) * 4  # magic number
            self.register_buffer('outside_circle', circle)

    def outside_circle_loss(self, img, img_shifted):
        return torch.mean(utils.rgb2gray(img) * self.outside_circle)

    def sobel_shift_loss(self, img, img_shifted):
        """Edge pairwise loss"""
        edge = K.sobel(utils.rgb2gray(img).unsqueeze(0))  # extract edges
        edge_shifted = K.sobel(utils.rgb2gray(img_shifted).unsqueeze(0))  # extract edges
        return torch.mean((edge - edge_shifted) ** 2)  # encourage edges to be the same

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


def run(cfg: DictConfig):

    # Device
    device = torch.device('cuda')

    # Load GAN
    G = make_gan(gan_type=cfg.data_gen.generator.gan_type, **cfg.data_gen.generator.kwargs)
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
    criterion = UnsupervisedSegmentationLoss(cfg.losses, image_size=cfg.data_gen.generator.image_size)
    criterion.to(device)

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
            log_message += f' | {k}: {v.item():.3f}'
        if i % cfg.log_every == 0:
            progress_bar.write(log_message)

        # Visualize with Tensorboard and save to file
        if i % cfg.vis_every == 0:
            img_grid = utils.create_grid(G=G, model=model, zs=z_vis_fixed, ys=y_vis_fixed, n_imgs=8)
            img_file = f"visualization-{i:05d}.png"
            img_grid = (img_grid * 255).astype(np.uint8)
            Image.fromarray(img_grid).save(str(img_file))

            # # Uncomment these lines to visualize the data with Streamlit
            # import streamlit as st
            # st.image(img_grid, caption=i)

    # Save checkpoint after iterations are complete
    utils.save_checkpoint(model=model, name='latest.pth', iteration=i, cfg=cfg)
    print('Done')


@hydra.main(config_path='../config', config_name='optimize')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == '__main__':
    main()
