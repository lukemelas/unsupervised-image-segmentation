import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf, DictConfig
import logging
import hydra

from segmentation import utils
from segmentation import metrics
from models import UNet
from datasets import SegmentationDataset, create_gan_dataset, create_train_and_val_dataloaders


class SementationModule(pl.LightningModule):

    def __init__(self, net, cfg):
        super().__init__()
        self.net = net
        self.cfg = cfg
        self.save_hyperparameters(cfg)

    def configure_optimizers(self):
        r"""Create optimizer and scheduler"""
        Optimizer = getattr(torch.optim, self.cfg.optimizer.kind)
        opt = Optimizer(self.net.parameters(), **self.cfg.optimizer.kwargs)
        Scheduler = getattr(torch.optim.lr_scheduler, self.cfg.scheduler.kind)
        sch = Scheduler(opt, **self.cfg.scheduler.kwargs)
        return [opt], [sch]

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        if len(img.shape) > 4:
            img, mask = img.squeeze(0), mask.squeeze(0)
        out = self(img)
        loss = F.cross_entropy(out, mask)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        img, mask = batch
        out = self(img)
        loss_val = F.cross_entropy(out, mask)
        results = metrics.compute_metrics(
            preds=out, targets=mask, metrics=['acc', 'iou'],  # no f_score for val because slow
            threshold=self.cfg.data_seg.binary_threshold)
        results['loss'] = [loss_val.item()]
        return results

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        img, mask = batch
        out = self(img)
        out = F.interpolate(out, mask.shape[-2:], mode='bilinear', align_corners=False)
        results = metrics.compute_metrics(
            preds=out, targets=mask, metrics=['acc', 'iou', 'f_beta', 'f_max'],
            threshold=self.cfg.data_seg.binary_threshold)
        return results

    def validation_epoch_end(self, outputs):
        dataset_names = ['GAN', *[x.name for x in self.cfg.data_seg.data]]
        self.epoch_end(outputs, dataset_names, prefix='val')

    def test_epoch_end(self, outputs):
        dataset_names = [x.name for x in self.cfg.data_seg.data]
        self.epoch_end(outputs, dataset_names, prefix='test')

    def epoch_end(self, outputs, dataset_names, prefix):
        outputs = [outputs] if len(dataset_names) == 1 else outputs
        for outputs, dataset_name in zip(outputs, dataset_names):
            outputs = metrics.list_of_dict_of_lists_to_dict_of_lists(outputs)
            results = metrics.aggregate_metrics(outputs)
            for name, value in results.items():
                self.log(f"{prefix}-{dataset_name}-{name}", value)


@hydra.main(config_path='../config', config_name='segment')
def main(cfg: DictConfig):

    # This is here to collapse the code in VS Code
    if True:

        # Setup
        print = logging.getLogger(__name__).info
        print(OmegaConf.to_yaml(cfg))
        pl.seed_everything(cfg.seed)

        # Create validation and test segmentation datasets
        # NOTE: The batch size must be 1 for test because the masks are different sizes,
        # and evaluation should be done using the mask at the original resolution.
        val_dataloaders = []
        test_dataloaders = []
        for _cfg in cfg.data_seg.data:
            kwargs = dict(images_dir=_cfg.images_dir, labels_dir=_cfg.labels_dir,
                          image_size=cfg.data_seg.image_size)
            val_dataset = SegmentationDataset(**kwargs, crop=True)
            test_dataset = SegmentationDataset(**kwargs, crop=_cfg.crop, resize_mask=False)
            val_dataloaders.append(DataLoader(val_dataset, **cfg.dataloader))
            test_dataloaders.append(DataLoader(test_dataset, **{**cfg.dataloader, 'batch_size': 1}))

    # Evaluate only
    if not cfg.train:
        assert cfg.eval_checkpoint is not None

        # Print dataset info
        for i, dataloader in enumerate(test_dataloaders):
            dataset = dataloader.dataset
            print(f'Test dataset / dataloader size [{i}]: {len(dataset)} / {len(dataset)}')

        # Create trainer
        trainer = pl.Trainer(**cfg.trainer)

        # Load checkpoint(s)
        net = UNet().eval()
        checkpoint = torch.load(cfg.eval_checkpoint, map_location='cpu')
        state_dict = {k.replace('net.', ''): v for k, v in checkpoint["state_dict"].items()}
        net.load_state_dict(state_dict)
        print(f'Loaded checkpoint from {cfg.eval_checkpoint}')

        # Create module
        module = SementationModule(net, cfg).eval()

        # Compute test results
        trainer.test(module, test_dataloaders=test_dataloaders)

        # Pretty print results
        table = utils.get_metrics_as_table(trainer.callback_metrics)
        print('\n' + str(table.round(decimals=3)))

    # Train
    else:

        # Generated images: load from disk
        if cfg.data_gen.load_from_disk:
            print('Loading images from disk')

            # Transforms
            train_transform = val_transform = A.Compose([
                A.Resize(cfg.data_gen.image_size, cfg.data_gen.image_size),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()])

            # Loaders
            gan_train_dataloader, gan_val_dataloader = create_train_and_val_dataloaders(
                cfg, train_transform=train_transform, val_transform=val_transform)

        # Generated images: generate on the fly
        else:
            print('Loading images on-the-fly')

            # Create GAN dataset
            gan_train_dataset = create_gan_dataset(cfg.data_gen)

            # GAN training dataloader
            # NOTE: Only 1 process (num_workers=0) supported
            gan_train_dataloader = DataLoader(gan_train_dataset, batch_size=1)

            # Load or create GAN validation batches
            print('Creating new GAN validation set.')
            num_batches = max(1, cfg.data_gen.val_images // cfg.data_gen.kwargs.batch_size)
            gan_val_batches = utils.get_subset_of_dataset(dataset=gan_train_dataset, num_batches=num_batches)
            gan_val_dataset = TensorDataset(*gan_val_batches)

            # Save example images from GAN validation dataset
            fname = 'generated-val-examples.png'
            utils.save_overlayed_images(gan_val_batches, filename=fname, is_mask=True)
            print(f'Saved visualization images to {fname}')

            # Validation dataloader
            gan_val_dataloader = DataLoader(gan_val_dataset, **cfg.dataloader)

        # Summary of dataset/dataloader sizes
        print(f'Generated train {utils.get_dl_size(gan_train_dataloader)}')
        print(f'Generated val {utils.get_dl_size(gan_val_dataloader)}')
        for i, dl in enumerate(val_dataloaders):
            print(f'Seg val [{i}] {utils.get_dl_size(dl)}')

        # Validation dataloaders
        val_dataloaders = [gan_val_dataloader, *val_dataloaders]

        # Checkpointer
        callbacks = [
            pl.callbacks.ModelCheckpoint(monitor='train_loss', save_top_k=20, save_last=True, verbose=True),
            pl.callbacks.LearningRateMonitor('step')
        ]

        # Logging
        logger = pl.loggers.WandbLogger(name=cfg.name) if cfg.wandb else True

        # Trainer
        trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

        # Lightning
        net = UNet().train()
        module = SementationModule(net, cfg)

        # Train
        trainer.fit(module, train_dataloader=gan_train_dataloader, val_dataloaders=val_dataloaders)

        # Test
        trainer.test(module, test_dataloaders=test_dataloaders)

        # Pretty print results
        table = utils.get_metrics_as_table(trainer.callback_metrics)
        print('\n' + str(table.round(decimals=3)))


if __name__ == '__main__':
    main()
