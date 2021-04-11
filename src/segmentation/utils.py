from collections import defaultdict
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import ToPILImage
from tqdm.auto import tqdm, trange


def get_subset_of_dataset(dataset, num_batches):
    """Creates a dataset from a subset of another dataset. We use this
    to create a validation set from our (infinite) training dataset. Note
    that we keep the entire dataset in memory because this val set is small.
    We could write to disk instead, but there's no need to."""
    batches = []
    for i in trange(num_batches):
        batch = dataset[i]
        batches.append(batch)

    def collate(x): return [torch.cat([b[i] for b in x], dim=0) for i in range(len(x[0]))]
    return collate(batches)


def collate_and_remove_batch_dim(batch):
    """A collate function that removes the batch dimension"""
    return (x.squeeze(0) for x in default_collate(batch))


def to_image(t: torch.Tensor, is_mask: bool = False):
    t = t.cpu().detach()
    if len(t.shape) == 4:
        t = t[0]
    if is_mask:
        t = t.squeeze(dim=0)  # convert to 2-dimensional mask
        t = t.to(torch.uint8).numpy()
        return Image.fromarray(t)
    else:
        t = torch.clamp(t * 0.5 + 0.5, min=0, max=1)  # de-normalize
        return ToPILImage()(t)  # convert to image


def save_images(t: torch.Tensor, file_prefix, is_mask=False, num_images=8):
    """Saves images for visualization"""
    for i in range(min(num_images, len(t))):
        file_path = f"{file_prefix}-{i}.png"
        image = to_image(t[i])
        image.save(file_path)


def save_overlayed_images(imgs_and_masks, filename, is_mask=False, num_rows=3, num_cols=3):
    """Saves images for visualization"""
    assert len(imgs_and_masks[0]) >= num_rows * num_cols, f'too few images: {len(imgs_and_masks[0])}'
    idx = 0
    rows = []
    for _ in range(num_rows):
        row = []
        for _ in range(num_cols):
            img = np.array(to_image(imgs_and_masks[0][idx]).convert('RGB'))
            # img_shifted = np.array(to_image(imgs_and_masks[1][idx]).convert('RGB'))
            mask = np.array(to_image(imgs_and_masks[1][idx]).convert('RGB'))
            idx += 1
            row.append(img)
            # row.append(img_shifted)
            row.append(mask)
            row.append(img * 0)
        rows.append(np.hstack(row))
    img = np.vstack(rows)
    image = Image.fromarray(img)
    image.save(filename)


def get_metrics_as_table(callback_metrics):
    table = defaultdict(dict)
    for k, v in callback_metrics.items():
        if k.startswith('test'):
            prefix, dataset_name, metric_name = k.split('-')
            table[dataset_name][metric_name] = v.item()
        else:
            print(f'{k}: {v.item()}')
    return pd.DataFrame(table)


def setup_multirun():
    from hydra.core.hydra_config import HydraConfig
    if 'num' in HydraConfig.get().job:
        job_num = HydraConfig.get().job.num % torch.cuda.device_count()
        gpu = job_num % torch.cuda.device_count()
        torch.cuda.set_device(gpu)
        print(f'Job number {job_num:2d}')
        print(f'Setting active GPU to {gpu}')


def collate_to_list(batch):
    r"""Converts to tensor and returns list"""
    return list(zip(*batch))


def set_requires_grad(module, requires_grad=True):
    for p in module.parameters():
        p.requires_grad = requires_grad


def get_dl_size(dl):
    return f'ds/dl size: {len(dl.dataset)} / {len(dl)}'
