from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra
import pytorch_lightning as pl
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from PIL import Image

from datasets import create_gan_dataset


def tensor_to_image(t: torch.Tensor):
    t = t.cpu().detach()
    t = torch.clamp(t * 0.5 + 0.5, min=0, max=1)
    return torchvision.transforms.ToPILImage()(t)


def tensor_to_mask(t: torch.Tensor):
    t = t.cpu().detach()
    t = t * 255
    return Image.fromarray(t.numpy().astype(np.uint8))


def run(cfg: DictConfig):
    # Setup
    torch.set_grad_enabled(False)  # disable gradient

    pl.seed_everything(cfg.seed)  # set random seed

    # Create GAN dataset
    gan_train_dataset = create_gan_dataset(cfg.data_gen)

    # Create directory to save files
    save_dir = Path(cfg.data_gen.save_dir) / 'images'
    save_dir.mkdir(exist_ok=True, parents=True)

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
                y = int(output_dict['y']) if 'y' in output_dict else 0
                stem = f'{i:08d}-seed_{cfg.seed}-class_{y:03d}'
                img.save(save_dir / f'{stem}.jpg')
                mask.save(save_dir / f'{stem}.png')
                pbar.update(1)
                i += 1
    print('Done.')


@hydra.main(config_path='../config', config_name='segment')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # print config
    assert cfg.data_gen.save_dir is not None, 'specify save_dir'
    run(cfg)


if __name__ == '__main__':
    main()
