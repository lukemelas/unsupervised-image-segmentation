from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def get_paths_from_folders(images_dir):
    """Returns list of files in folders of input"""
    paths = []
    for folder in Path(images_dir).iterdir():
        for p in folder.iterdir():
            paths.append(p)
    return paths


def central_crop(x):
    dims = x.size
    crop = T.CenterCrop(min(dims[0], dims[1]))
    return crop(x)


class SegmentationDataset(Dataset):

    def __init__(
            self,
            images_dir: str,
            labels_dir: str,
            image_size: int = 128,
            resize_image=True,
            resize_mask=True,
            crop=True,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
    ):
        # Find out if dataset is organized into folders or not
        has_folders = not any(str(next(Path(images_dir).iterdir())).endswith(ext)
                              for ext in IMG_EXTENSIONS)

        # Get and sort list of paths
        if has_folders:
            image_paths = get_paths_from_folders(images_dir)
            label_paths = get_paths_from_folders(labels_dir)
        else:
            image_paths = Path(images_dir).iterdir()
            label_paths = Path(labels_dir).iterdir()
        self.image_paths = list(sorted(image_paths))
        self.label_paths = list(sorted(label_paths))
        assert len(self.image_paths) == len(self.label_paths)

        # Transformation
        image_transform = [T.ToTensor(), T.Normalize(mean=mean, std=std)]
        mask_transform = [T.ToTensor()]
        if resize_image:
            image_transform.insert(0, T.Resize(image_size))
        if resize_mask:
            mask_transform.insert(0, T.Resize(image_size))
        if crop:
            image_transform.insert(0, central_crop)
            mask_transform.insert(0, central_crop)
        self.image_transform = T.Compose(image_transform)
        self.mask_transform = T.Compose(mask_transform)

        # # Transformation
        # if transform_type == 'resize':
        #     self.image_transform = T.Compose([T.Resize(image_size), central_crop, *image_to_tensor])
        #     self.mask_transform = T.Compose([T.Resize(image_size), central_crop, T.ToTensor()])
        # elif transform_type == 'resize_image_only':
        #     self.image_transform = T.Compose([central_crop, T.Resize(image_size), *image_to_tensor])
        #     self.mask_transform = T.Compose([central_crop, T.ToTensor()])
        # elif transform_type == 'crop_only':
        #     self.image_transform = T.Compose([central_crop, *image_to_tensor])
        #     self.mask_transform = T.Compose([central_crop, T.ToTensor()])
        # else:
        #     raise NotImplementedError()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        mask = Image.open(self.label_paths[idx])

        # Transform
        # TODO: Figure out why they preprocess like this (???)
        image = image.convert('RGB')  # to reproduce (???)
        mask = mask.convert('RGB')  # to reproduce (???)
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5)[0].long()  # TODO: this could be improved
        return image, mask
