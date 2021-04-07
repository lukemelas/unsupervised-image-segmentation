from collections import defaultdict
import os
import time
from pathlib import Path
from functools import partial
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf, DictConfig
import logging
import hydra

from UNet import UNet, Ensemble
from datasets import SegmentationDataset, create_gan_dataset, create_train_and_val_dataloaders
from metrics import compute_metrics, aggregate_metrics, list_of_dict_of_lists_to_dict_of_lists
from utils import get_subset_of_dataset, save_overlayed_images, get_metrics_as_table


def compute_entropy_loss(v):
    r"""Entropy loss for input shape BxCxHxW which has already been passed through softmax"""
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))


class SementationModule(pl.LightningModule):
    def __init__(self, net, cfg):
        super().__init__()
        self.net = net
        self.cfg = cfg
        self.save_hyperparameters(cfg)

    def configure_optimizers(self):
        if self.cfg.optimizer.kind == 'SGD':
            Opt = partial(torch.optim.SGD, momentum=0.9)
        else:
            Opt = getattr(torch.optim, self.cfg.optimizer.kind)
        opt = Opt(self.net.parameters(), lr=self.cfg.optimizer.lr)
        sch = torch.optim.lr_scheduler.StepLR(
            opt, gamma=self.cfg.optimizer.lr_decay_gamma,
            step_size=self.cfg.optimizer.lr_decay_step_size)
        return [opt], [sch]

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        if hasattr(self.cfg, 'entropy_lambda') and self.cfg.entropy_lambda:
            (img, mask), (imagenet_img, _) = batch
            out = self(img)
            seg_loss = F.cross_entropy(out, mask)
            imagenet_out = self(imagenet_img)
            entropy_loss = compute_entropy_loss(torch.softmax(imagenet_out, dim=1))
            loss = seg_loss + self.cfg.entropy_lambda * entropy_loss
            self.log('seg_loss', seg_loss)
            self.log('entropy_loss', entropy_loss)
            loss_dict = {'loss': loss, 'seg_loss': seg_loss, 'entropy_loss': entropy_loss}
            self.log('train_loss', loss)
        else:
            img, mask = batch
            if len(img.shape) > 4:
                img, mask = map(lambda x: x.squeeze(0), batch)
            out = self(img)
            loss = F.cross_entropy(out, mask)
            loss_dict = {'loss': loss}
            self.log('train_loss', loss)
        return loss_dict

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        img, mask = batch
        out = self(img)

        # Validation metrics
        loss_val = F.cross_entropy(out, mask)
        results = compute_metrics(
            preds=out, targets=mask, metrics=['acc', 'iou'], threshold=self.cfg.data_seg.binary_threshold)
        results['loss'] = [loss_val.item()]
        return results

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        img, mask = batch
        out = self(img)
        if isinstance(out, tuple):
            out = out[0]

        # Resize and test metrics
        out = F.interpolate(out, mask.shape[-2:], mode='bilinear', align_corners=False)
        results = compute_metrics(
            preds=out, targets=mask, metrics=['acc', 'iou', 'f_beta', 'f_max'],
            threshold=self.cfg.data_seg.binary_threshold, swap_dims=(hasattr(self.cfg, 'deepusps') and self.cfg.deepusps))
        return results

    def validation_epoch_end(self, outputs):
        dataset_names = ['GAN', *[x.name for x in self.cfg.data_seg.data]]
        self.epoch_end(outputs, dataset_names)

    def test_epoch_end(self, outputs):
        dataset_names = [x.name for x in self.cfg.data_seg.data]
        self.epoch_end(outputs, dataset_names, prefix='test')

    def epoch_end(self, outputs, dataset_names, prefix='val'):
        outputs = [outputs] if len(dataset_names) == 1 else outputs
        for outputs, dataset_name in zip(outputs, dataset_names):
            outputs = list_of_dict_of_lists_to_dict_of_lists(outputs)
            results = aggregate_metrics(outputs)
            for name, value in results.items():
                self.log(f"{prefix}-{dataset_name}-{name}", value)


def run(cfg: DictConfig):

    # [VSCODE] Segmentation datasets
    if True:

        # Setup
        print = logging.getLogger(__name__).info
        print(OmegaConf.to_yaml(cfg))
        pl.seed_everything(cfg.seed)

        # Create validation and test segmentation datasets
        # NOTE: The batch size must be 1 for test because the masks are different sizes,
        # and evaluation should be done using the mask at the original resolution. I
        # could makethis work with batched inputs, but there's no need.
        val_dataloaders = []
        test_dataloaders = []
        for _cfg in cfg.data_seg.data:
            kwargs = dict(images_dir=_cfg.images_dir, labels_dir=_cfg.labels_dir,
                          image_size=cfg.data_seg.image_size)

            # NEW
            if hasattr(cfg, 'deepusps') and cfg.deepusps:
                print(f'Using DeepUSPS mean and std!')
                print(f'Using seg size: {cfg.data_seg.image_size}')
                kwargs['mean'] = [0.1829540508368939, 0.18656561047509476, 0.18447508988480435]
                kwargs['std'] = [0.29010095242892997, 0.32808144844279574, 0.28696394422942517]

            val_dataset = SegmentationDataset(**kwargs, crop=True)
            test_dataset = SegmentationDataset(**kwargs, crop=_cfg.crop, resize_mask=False)
            val_dataloaders.append(DataLoader(val_dataset, **cfg.dataloader))
            # DataLoader(test_dataset, **cfg.dataloader, collate_fn=collate_to_list))
            test_dataloaders.append(DataLoader(test_dataset, **{**cfg.dataloader, 'batch_size': 1}))

    # Evaluate only
    if not cfg.train:

        # Print dataset info
        for i, dataloader in enumerate(test_dataloaders):
            dataset = dataloader.dataset
            print(f'Test dataset / dataloader size [{i}]: {len(dataset)} / {len(dataset)}')

        # Create trainer
        trainer = pl.Trainer(**cfg.trainer)

        # DeepUSPS
        if hasattr(cfg, 'deepusps') and cfg.deepusps:
            from utils_copy import DRNSeg
            net = DRNSeg('drn_d_105', 2, None, pretrained=False)
            checkpoint = torch.load('/home/luke/projects/experiments/DeepUSPS/pretrained/DeepUSPS_1.pth.tar')
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
            net.load_state_dict(state_dict)

        # Load checkpoint(s)
        else:
            nets = []
            assert cfg.eval_checkpoint is not None, 'no checkpoint specified'
            checkpoints = [cfg.eval_checkpoint] if isinstance(cfg.eval_checkpoint, str) else cfg.eval_checkpoint
            for checkpoint in checkpoints:
                net = UNet().eval()
                state_dict = torch.load(checkpoint, map_location='cpu')["state_dict"]
                state_dict = {k[4:]: v for k, v in state_dict.items()}  # remove "net."
                net.load_state_dict(state_dict)
                nets.append(net)
                print(f'Loaded checkpoint from {checkpoint}')
            net = Ensemble(nets)

        module = SementationModule(net, cfg).eval()

        # Compute test results
        trainer.test(module, test_dataloaders=test_dataloaders)

        # Pretty print results
        table = get_metrics_as_table(trainer.callback_metrics)
        print('\n' + str(table.round(decimals=3)))

    # Train and evaluate
    else:

        ############################
        ########### GAN ############
        ############################

        # Load images from disk
        if cfg.data_gen.load_from_disk:
            print('Loading images from disk')

            # Augmentations
            # NOTE: No resizing because all images are 128 x 128
            image_size, crop_size = cfg.data_gen.image_size, cfg.data_gen.crop_size
            normalize_and_to_tensor = [A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()]
            val_transform = A.Compose([A.Resize(image_size, image_size), *normalize_and_to_tensor])
            if cfg.data_gen.augmentations == 0:
                train_transform = val_transform
            elif cfg.data_gen.augmentations == 3:
                train_transform = A.Compose([
                    A.Resize(image_size, image_size), A.RandomResizedCrop(crop_size, crop_size),
                    A.Compose([A.RandomBrightnessContrast(p=1), A.HueSaturationValue(p=1)], p=0.4),
                    A.ToGray(p=0.1), A.GaussianBlur(3, p=0.2), *normalize_and_to_tensor])
            else:
                raise NotImplementedError('Invalid augmentation')

            # NEW! Entropy minimization
            if hasattr(cfg, 'entropy_lambda') and cfg.entropy_lambda:
                from datasets import ZipDataset, SimpleDataset, RawDataset
                # Create standard training dataset from directory of images
                from sklearn.model_selection import train_test_split
                val_size = cfg.data_gen.data.get('val_size', 0.01)
                random_seed = cfg.data_gen.data.get('random_seed', 42)
                files = list(filter(lambda x: str(x).endswith('.jpg'), Path(cfg.data_gen.data.root).iterdir()))
                train_files, val_files = train_test_split(files, test_size=val_size, random_state=random_seed)
                train_dataset = RawDataset(files=train_files, transform=train_transform)
                val_dataset = RawDataset(files=val_files, transform=val_transform)
                # Create ImageNet training dataset
                imagenet = SimpleDataset(
                    root="/home/luke/machine-learning-datasets/image-classification/imagenet/train",
                    transform=A.Compose([A.RandomResizedCrop(128, 128), *normalize_and_to_tensor]))
                # Zip datasets together
                train_dataset = ZipDataset([train_dataset, imagenet])
                # Dataloader
                gan_train_dataloader = DataLoader(train_dataset, **cfg.dataloader)
                gan_val_dataloader = DataLoader(val_dataset, **cfg.dataloader)
            else:

                # Datasets
                gan_train_dataloader, gan_val_dataloader = create_train_and_val_dataloaders(
                    cfg, train_transform=train_transform, val_transform=val_transform)

        # Load GAN and generate images on the fly
        else:

            # Create GAN dataset
            gan_train_dataset = create_gan_dataset(cfg.data_gen)

            # GAN training dataloader
            # NOTE: Only 1 process (num_workers=0) supported
            gan_train_dataloader = DataLoader(gan_train_dataset, batch_size=1)

            # Load or create GAN validation batches
            if Path(cfg.data_gen.gan_val_file).is_file() and not cfg.data_gen.regenerate_gan_val_file:
                print(f'Loading GAN validation set from file: {cfg.data_gen.gan_val_file}')
                gan_val_batches = torch.load(cfg.data_gen.gan_val_file, map_location='cpu')
            else:
                print('Creating and saving new GAN validation set.')
                num_batches = max(cfg.data_gen.gan_val_num_images, cfg.data_gen.kwargs.batch_size) // \
                    cfg.data_gen.kwargs.batch_size
                gan_val_batches = get_subset_of_dataset(
                    dataset=gan_train_dataset,
                    num_batches=num_batches)
                Path(cfg.data_gen.gan_val_file).parent.mkdir(parents=True, exist_ok=True)
                torch.save(gan_val_batches, cfg.data_gen.gan_val_file)
            gan_val_dataset = TensorDataset(*gan_val_batches)

            # Save example images from GAN validation dataset
            if cfg.data_gen.save_gan_val_vis:
                vis_filename = Path('vis/gen-val-vis.png')
                vis_filename.parent.mkdir(parents=True, exist_ok=True)
                save_overlayed_images(gan_val_batches, filename=vis_filename, is_mask=True)
                print(f'Saved vis images to {vis_filename}')

            # Validation dataloader
            gan_val_dataloader = DataLoader(gan_val_dataset, **cfg.dataloader)

        # Print summary of dataset/dataloader sizes
        def dataloader_size_helper(dl):
            return f'ds/dl size: {len(dl.dataset)} / {len(dl)}'
        print(f'GAN train {dataloader_size_helper(gan_train_dataloader)}')
        print(f'GAN val {dataloader_size_helper(gan_val_dataloader)}')
        for i, dl in enumerate(val_dataloaders):
            print(f'GAN val [{i}] {dataloader_size_helper(dl)}')

        # Validation dataloaders
        val_dataloaders = [gan_val_dataloader, *val_dataloaders]

        # Checkpointer
        callbacks = [
            pl.callbacks.ModelCheckpoint(monitor='train_loss', save_top_k=20, save_last=True, verbose=True),
            pl.callbacks.LearningRateMonitor('step')]

        # Logging
        logger = []
        assert len(cfg.logging.loggers) > 0, 'please specify a logger'
        if 'tensorboard' in cfg.logging.loggers:
            logger = [pl.loggers.TensorBoardLogger(
                save_dir=cfg.logging.log_dir, name=cfg.name, version=cfg.logging.version)]
        if 'wandb' in cfg.logging.loggers:
            logger += [pl.loggers.WandbLogger(
                project='ganseg', name=cfg.name, version=cfg.logging.version)]

        # Trainer
        trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

        # Lightning module
        net = UNet().train()
        module = SementationModule(net, cfg)

        # Checkpoint
        if cfg.resume_model_only:
            state_dict = torch.load(cfg.resume_model_only, map_location='cpu')["state_dict"]
            module.load_state_dict(state_dict)
            print(f'Loaded model from {cfg.resume_model_only}')

        # Train
        trainer.fit(module, train_dataloader=gan_train_dataloader, val_dataloaders=val_dataloaders)

        # Test
        trainer.test(module, test_dataloaders=test_dataloaders)

        # Pretty print results
        table = get_metrics_as_table(trainer.callback_metrics)
        print('\n' + str(table.round(decimals=3)))

        # Return for hyperparameter optimization
        return trainer.callback_metrics['val-CUB-iou']


@hydra.main(config_path='config', config_name='segment')
def main(cfg: DictConfig):
    return run(cfg)


if __name__ == '__main__':
    main()
