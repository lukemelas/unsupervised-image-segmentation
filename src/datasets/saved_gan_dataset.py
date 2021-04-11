import numpy as np
import torch
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.folder import default_loader
import cv2
try:
    from sklearn.model_selection import train_test_split
except:
    pass


def create_train_and_val_dataloaders(cfg, train_transform=None, val_transform=None):

    # Split
    val_size = cfg.data_gen.data.get('val_size', 0.01)
    random_seed = cfg.data_gen.data.get('random_seed', 42)
    files = list(filter(lambda x: str(x).endswith('.jpg'), Path(cfg.data_gen.data.root).iterdir()))
    train_files, val_files = train_test_split(files, test_size=val_size, random_state=random_seed)

    # Create datasets
    train_dataset = RawDataset(files=train_files, transform=train_transform)
    val_dataset = RawDataset(files=val_files, transform=val_transform)

    # Create and return dataloaders
    train_dataloader = DataLoader(train_dataset, **cfg.dataloader)
    val_dataloader = DataLoader(val_dataset, **cfg.dataloader)
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
