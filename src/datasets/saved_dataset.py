from os import posix_fadvise
import numpy as np
import torch
from pathlib import Path
import time
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.folder import default_loader
import h5py
import cv2


def webdataset_transform_wrapper(transform):
    def webdataset_train_transform(d):
        image = np.array(d['jpg'].convert('RGB'))
        mask = np.array(d['png'].convert('L'))
        transformed = transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].long()
        mask[mask == 255] = 1
        return image, mask
    return webdataset_train_transform


def create_train_and_val_dataloaders(cfg, train_transform=None, val_transform=None):
    if cfg.data_gen.data.dataset_type == 'raw':
        from sklearn.model_selection import train_test_split
        val_size = cfg.data_gen.data.get('val_size', 0.01)
        random_seed = cfg.data_gen.data.get('random_seed', 42)
        files = list(filter(lambda x: str(x).endswith('.jpg'), Path(cfg.data_gen.data.root).iterdir()))
        train_files, val_files = train_test_split(files, test_size=val_size, random_state=random_seed)
        train_dataset = RawDataset(files=train_files, transform=train_transform)
        val_dataset = RawDataset(files=val_files, transform=val_transform)
        train_dataloader = DataLoader(train_dataset, **cfg.dataloader)
        val_dataloader = DataLoader(val_dataset, **cfg.dataloader)
    elif cfg.data_gen.data.dataset_type == 'webdataset':
        import webdataset
        dataloader_kwargs = dict(cfg.dataloader)
        batch_size = dataloader_kwargs.pop('batch_size')
        num_batches = cfg.data_gen.data.length // batch_size
        train_transform = webdataset_transform_wrapper(train_transform)
        val_transform = webdataset_transform_wrapper(val_transform)
        train_dataset = (
            webdataset.Dataset(cfg.data_gen.data.train_tar_file, length=num_batches)
            .shuffle(True)
            .decode('pil')
            .map(train_transform))
        val_dataset = (
            webdataset.Dataset(cfg.data_gen.data.val_tar_file, length=num_batches)
            .decode('pil')
            .map(val_transform))
        train_dataloader = DataLoader(train_dataset.batched(batch_size), batch_size=None, **dataloader_kwargs)
        val_dataloader = DataLoader(val_dataset.batched(batch_size), batch_size=None, **dataloader_kwargs)
    else:
        raise NotImplementedError()

    return train_dataloader, val_dataloader


class RawDataset(torch.utils.data.Dataset):
    def __init__(self, *, files, transform=None):
        super().__init__()
        self.files = files
        self.transform = transform

    def __getitem__(self, idx):
        image_filename = str(self.files[idx])
        mask_filename = str(image_filename).replace('.jpg', '.png')
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_filename, cv2.IMREAD_UNCHANGED)
        mask = mask.astype(float) / 255.0
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].long()
        return image, mask

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":

    # Augmentations
    import os
    from os.path import join
    from omegaconf import OmegaConf
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    train_transform = A.Compose([
        A.Resize(128, 128),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()])
    val_transform = A.Compose([
        A.Resize(128, 128),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()])

    # Config
    example_run = "ImageNet-ls10.0_lo2.0_rm2.0_rv0.5_lr5e-3"
    cfg = OmegaConf.create({
        'data_gen': {
            'data': {
                # # RawDataset
                # 'dataset_type': 'raw',
                # 'root': os.path.join(os.environ['GANSEG_DATA_ROOT'], example_run)
                # WebDataset
                'dataset_type': 'webdataset',
                'train_tar_file': os.path.join(os.environ['GANSEG_DATA_ROOT'], f"{example_run}.tar"),
                'val_tar_file': os.path.join(os.environ['GANSEG_DATA_ROOT'], f"{example_run}.tar"),
                'length': 1000
            }
        },
        'dataloader': {
            'batch_size': 1024,
            'num_workers': 16
        }
    })

    # Load
    print('Creating dataloaders')
    train_dataloader, val_dataloader = create_train_and_val_dataloaders(
        cfg, train_transform=train_transform, val_transform=val_transform)

    # Check
    import time
    for i, (img, mask) in enumerate(train_dataloader):
        if i == 0:
            print(img.shape, mask.shape)
        if i == 5:
            start = time.time()
        if i == 15:
            end = time.time()
            break
    print(end - start)
