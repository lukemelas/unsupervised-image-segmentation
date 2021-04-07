"""
Example usage:
    PYTHONPATH=. python helpers/generate-dataset.py data_gen=from-generation data_gen.save_size: 1000000
"""

from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra
import pytorch_lightning as pl
import numpy as np
import torch
import torchvision
from tqdm import tqdm, trange
from PIL import Image

from datasets import create_gan_dataset

import logging
log = logging.getLogger(__name__)


def tensor_to_image(t: torch.Tensor):
    t = t.cpu().detach()
    t = torch.clamp(t * 0.5 + 0.5, min=0, max=1)
    return torchvision.transforms.ToPILImage()(t)


def tensor_to_mask(t: torch.Tensor):
    t = t.cpu().detach()
    t = t * 255
    return Image.fromarray(t.numpy().astype(np.uint8))


@hydra.main(config_path='../config', config_name='segment')
def main(cfg: DictConfig):

    # Setup
    torch.set_grad_enabled(False)  # disable gradient
    print = log.info  # logging
    print(OmegaConf.to_yaml(cfg))  # print config
    pl.seed_everything(cfg.seed)  # set random seed

    # Create GAN dataset
    gan_train_dataset = create_gan_dataset(cfg.data_gen)

    # Create directory to save files
    assert cfg.data_gen.save_dir is not None
    save_dir = Path(cfg.data_gen.save_dir)
    (save_dir / 'images').mkdir(exist_ok=True, parents=True)
    (save_dir / 'extra').mkdir(exist_ok=True, parents=True)
    print(f'Saving images and extra information to {save_dir}')

    # Generate dataset and save to disk as images
    i = 0
    num_imgs = cfg.data_gen.save_size
    total = num_imgs if num_imgs > 0 else None
    with tqdm(desc="generated images", total=total) as pbar:
        while (i < num_imgs):
            output_batch = gan_train_dataset.__getitem__(i, return_save_format=True)
            for output_dict in output_batch:
                img = tensor_to_image(output_dict['img'])
                mask = tensor_to_mask(output_dict['mask'])
                y = int(output_dict['y'])
                stem = f'{i:08d}-seed_{cfg.seed}-class_{y:03d}'
                img.save(save_dir / 'images' / f'{stem}.jpg')
                mask.save(save_dir / 'images' / f'{stem}.png')
                pbar.update(1)
                i += 1

                # # OPTIONAL -- also save extra info (e.g. the original zs used to generate the data)
                # extra = {k: v for k, v in output_dict.items() if 'img' not in k and 'mask' not in k}
                # torch.save(extra, save_dir / 'extra' / f'{i:08d}.pth')

    print('Done.')


if __name__ == '__main__':
    main()
