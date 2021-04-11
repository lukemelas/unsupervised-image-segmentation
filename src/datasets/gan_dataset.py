from pathlib import Path
from typing import List
from types import SimpleNamespace
import itertools
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from skimage.measure import label as measure_label
from skimage.measure import perimeter as measure_perimeter
from retry import retry
from omegaconf import DictConfig

from models import MODELS

from pytorch_pretrained_gans import make_gan


def to_cpu(x):
    return x.cpu() if isinstance(x, torch.Tensor) else x


def rgb2gray(img):
    gray = 0.2989 * img[:, 0] + 0.5870 * img[:, 1] + 0.1140 * img[:, 2]
    return gray


def onehot_to_int(y):
    r"""Some models return y in one-hot format, so we have to convert to int"""
    if y.dtype == torch.float and y.shape[-1] == 1000:
        y = torch.argmax(y, dim=-1)
    return y.reshape(-1)


def select_indices(s: SimpleNamespace, indices: torch.Tensor, error_msg: str):
    if torch.all(~indices):
        raise Exception(error_msg)
    for k, v in s.__dict__.items():
        s.__dict__[k] = v[indices]


def convert_images_to_mask(img, img_shifted, method='lighting', invert=False):
    """Converts generated images to binary segmentation mask"""
    # Note that we swap foreground and background if the invert flag is passed
    img_light = rgb2gray(img if invert else img_shifted)
    img_dark = rgb2gray(img_shifted if invert else img)
    if method == 'lighting':
        mask = img_light - img_dark > 0
    elif method == 'std':
        mask = img_light - img_dark > - 0.3 * (img_light - img_dark).std()
    else:
        raise NotImplementedError()
    return mask.to(torch.long)


def apply_connected_components_(m: np.ndarray, threshold: float):
    """Return masks with small connected components removed"""

    # Get connected components
    component, num = measure_label(m, return_num=True, background=0)
    areas = np.zeros([num + 1])
    for comp in range(1, num + 1, 1):
        areas[comp] = np.sum(component == comp)

    # Get area of biggest connected component
    max_component = np.argmax(areas)
    max_component_area = areas[max_component]

    # Create new mask (in-place) with filtered connected components
    m *= 0
    for comp in range(1, num + 1, 1):
        area = areas[comp]
        if float(area) / max_component_area > threshold:
            m[component == comp] = True
    return m


def apply_connected_components_filter(mask: torch.Tensor, threshold: float):
    """Iterates over mask and applies connected components filter"""
    processed_mask = mask.numpy()
    for m in processed_mask:
        apply_connected_components_(m, threshold)
    processed_mask = torch.from_numpy(processed_mask).to(mask.device)
    return processed_mask


def get_histogram_filter_indices(img_shifted: torch.Tensor, bins: int, r: int = 3):
    """Return indices of images not in the first/last histogram buckets"""
    indices = []
    weight = torch.ones([1, 1, r], device=img_shifted.device)
    for idx, img_shifted_i in enumerate(img_shifted):
        stats = torch.histc(img_shifted_i, bins, -1, 1)
        stats = F.conv1d(stats.view(1, 1, -1), weight, padding=r // 2)
        stats = stats.view(-1).cpu().numpy()
        maxes = np.r_[True, stats[1:] >= stats[:-1]] & np.r_[stats[:-1] >= stats[1:], True]
        maxes = np.nonzero(maxes)[0]
        indices.append(len(maxes) >= 2)
    return torch.tensor(indices)


def get_area_filter_indices(mask: torch.Tensor, thresholds: float):
    r"""Return indices of images with fraction of foreground below threshold"""
    assert len(thresholds) == 2 and thresholds[0] < thresholds[1]
    ref_size = mask.shape[-2] * mask.shape[-1]
    ref_fraction = mask.sum(dim=[-1, -2]).to(torch.float) / ref_size
    indices = (thresholds[0] <= ref_fraction) & (ref_fraction <= thresholds[1])
    return indices


def get_roundness_filter_indices(mask: torch.Tensor, threshold: float):
    r"""Filter by roundness, where roundness = (4 pi area) / perimeter^2"""

    # Loop over images
    indices = []
    num_pixels = mask.shape[-1] * mask.shape[-2]
    for i, m in enumerate(mask.numpy()):

        # Get connected components
        component, num = measure_label(m, return_num=True, background=0)
        if num == 0:
            return 1000000

        # Get area of biggest connected component
        areas, perimeters = [], []
        for i in range(1, num + 1):
            component_i = (component == i)
            area = np.sum(component_i)
            perimeter = measure_perimeter(component_i)
            areas.append(area)
            perimeters.append(perimeter)
        max_component = np.argmax(areas)
        max_component_area = areas[max_component]
        if num_pixels * 0.05 < max_component_area < num_pixels * 0.90:
            max_component_perimeter = perimeters[max_component]
            roundness = max_component_area / max_component_perimeter ** 2
            indices.append(roundness > threshold)
        else:
            indices.append(False)
    return torch.tensor(indices)


def get_overlap_filter_indices(mask_1: torch.Tensor, mask_2: torch.Tensor, threshold: float):
    indices = []
    for m1, m2 in zip(mask_1, mask_2):
        overlap = (m1 == m2).float().mean()
        indices.append(overlap >= threshold)
    return torch.tensor(indices)


class GANDatasetBase(torch.utils.data.Dataset):

    def make_noise(self, batch_size, idx=None):
        if self.zs is None:  # randomly sample noise
            return self.G.sample_latent(batch_size=batch_size, device=self.device)
        else:  # sample from z or from the neighborhood of z
            if idx is None:  # randomly sample z
                indices = torch.randint(high=len(self.zs), size=[batch_size], dtype=torch.long)
            else:  # get z
                indices = torch.arange(start=idx * batch_size, end=(idx + 1) * batch_size)
                indices = indices % len(self.zs)
            z = self.zs[indices]
            if self.z_noise > 0.0:  # sample from the neighborhood of z
                z = z + self.z_noise * torch.randn_like(z, device=z.device)
            return z

    def _get_partial_batch(self, idx=None):
        """Should return a dictionary"""
        raise NotImplementedError()

    @torch.no_grad()
    def __getitem__(self, idx, truncate_batch=False, return_save_format=False, return_keys=['img', 'mask']):
        """
        Since we are generating from a GAN and then filtering the result, we have to make sure
        we get the correct number of items in our batch.
        NOTE: the `idx` parameter is unused
        NOTE: this dataset is stochastic
        """

        # We continue building batch until it is of the desired size. It is formatted as a
        # list of dictionaries, for example:
        # [
        #   {'img': [...], 'img_shifted': [...], 'mask': [...], 'z': [...], 'd': [...], 'y': [...]},
        #   {'img': [...], 'img_shifted': [...], 'mask': [...], 'z': [...], 'd': [...], 'y': [...]},
        # ]
        batch = []
        while len(batch) < self.batch_size:
            partial_batch = self._get_partial_batch(idx)
            batch.extend(partial_batch)

        # Whether to keep "extra" generations above the specified batch size
        if truncate_batch:
            batch = batch[:self.batch_size]

        # Return either dictionaries for saving, or stacked images and masks for training
        if return_save_format:
            return batch  # a list of dictionaries
        else:
            return [torch.stack([d[k] for d in batch], dim=0) for k in return_keys]  # a list of tensors

    def __len__(self):
        return self.length


class GANDataset(GANDatasetBase):
    def __init__(
        self,
        G: nn.Module,
        model: nn.Module,
        zs: torch.Tensor = None,
        z_noise: int = 0.0,
        z_noise_truncation: float = 1.0,
        r: float = 5.0,
        batch_size: int = 64,
        generation_batch_size: int = None,
        mask_gen_method: str = 'lighting',
        filter_cc_threshold: float = 0.2,
        filter_size_thresholds: List[float] = [0.0, 0.5],
        filter_histogram_bins: int = 12,
        filter_roundness_threshold: float = 0.0,  # -0.015,
        invert: bool = False,
    ):
        super().__init__()

        # Generator
        self.G = G.eval()
        self.device = (next(G.parameters())).device

        # Model
        self.model = model.to(self.device).eval()

        # Generation parameters
        self.zs = zs
        self.z_noise = z_noise
        self.z_noise_truncation = z_noise_truncation
        self.r = r
        self.batch_size = batch_size
        self.gen_batch_size = generation_batch_size if generation_batch_size is not None else batch_size

        # Post-processing parameters
        self.mask_gen_method = mask_gen_method
        self.filter_cc_threshold = filter_cc_threshold
        self.filter_size_thresholds = filter_size_thresholds
        self.filter_histogram_bins = filter_histogram_bins
        self.filter_roundness_threshold = filter_roundness_threshold
        self.invert = invert

        # Infinite dataset (override this to set a batch size)
        self.length = -1

    @torch.no_grad()
    def gen_samples(self, z=None, batch_size=None, idx=None):
        """Sample from the generator G"""
        assert (z is None) ^ (batch_size is None), 'Expected exactly 1 of z or batch_size'
        if z is None:  # make noise if it is not provided
            z = self.make_noise(batch_size, idx).to(self.device)

        # Generate image with shifted latent vector
        d = self.model(z)

        # Here we generate the original and shifted images, dealing with the
        # conditional and unconditional cases separately
        if self.G.conditional:
            img, y = self.G(z, return_y=True)
            img_shifted = self.G(z + self.r * d, y=y)
            y = onehot_to_int(y)  # convert label to an integer, in case it is one-hot
        else:
            img = self.G(z)
            img_shifted = self.G(z + self.r * d)
            y = torch.zeros(len(z), device=self.device, dtype=torch.long)  # dummy label

        return img, img_shifted, y

    @retry(Exception, tries=25, delay=0.01)
    def _get_partial_batch(self, idx=None):

        # Generate samples
        img, img_shifted, y = list(map(to_cpu, self.gen_samples(batch_size=self.gen_batch_size, idx=idx)))

        # Convert images to mask
        mask = convert_images_to_mask(img, img_shifted, method=self.mask_gen_method, invert=self.invert).cpu()

        # Filter connected components in mask
        if self.filter_cc_threshold > 0.0:
            mask = apply_connected_components_filter(mask, threshold=self.filter_cc_threshold)

        # Filter generated images by area
        if self.filter_size_thresholds:
            indices = get_area_filter_indices(mask, thresholds=self.filter_size_thresholds)
            if torch.all(~indices):
                raise Exception('No images with masks of sufficient area')
            img, img_shifted, mask, y = img[indices], img_shifted[indices], mask[indices], y[indices]

        # Filter generated images by intensity histogram
        if self.filter_histogram_bins > 0:
            indices = get_histogram_filter_indices(img_shifted, bins=self.filter_histogram_bins)
            if torch.all(~indices):
                raise Exception('No images with histogram of sufficient intensity')
            img, img_shifted, mask, y = img[indices], img_shifted[indices], mask[indices], y[indices]

        # Filter generated images by roundness
        if self.filter_roundness_threshold:
            indices = get_roundness_filter_indices(mask, threshold=self.filter_roundness_threshold)
            if torch.all(~indices):
                raise Exception('No images with sufficiently round masks')
            img, img_shifted, mask, y = img[indices], img_shifted[indices], mask[indices], y[indices]

        # Return generated images/masks/labels/etc
        partial_batch = [dict(zip(['img', 'img_shifted', 'mask'], item))
                         for item in zip(img, img_shifted, mask)]
        return partial_batch


class DualGANDataset(GANDatasetBase):
    def __init__(
        self,
        G: nn.Module,
        model_light: nn.Module,
        model_dark: nn.Module,
        r_light: int = 5.0,
        r_dark: int = 5.0,
        mask_gen_method_light: str = 'lighting',
        mask_gen_method_dark: str = 'lighting',
        filter_cc_threshold_light: float = 0.2,
        filter_size_thresholds_light: float = 0.5,
        filter_histogram_bins_light: int = 12,
        filter_roundness_threshold_light: float = -0.015,
        filter_cc_threshold_dark: float = 0.2,
        filter_size_thresholds_dark: float = 0.5,
        filter_histogram_bins_dark: int = 12,
        filter_roundness_threshold_dark: float = -0.015,
        filter_overlap_threshold: float = 0.0,
        batch_size: int = 95,
        generation_batch_size: int = None,
        zs: torch.Tensor = None,
        z_noise: int = 0.0,
        z_noise_truncation: float = 1.0,
    ):
        super().__init__()

        # Generator
        self.G = G.eval()
        self.device = (next(G.parameters())).device

        # Model
        self.model_light = model_light.to(self.device).eval()
        self.model_dark = model_dark.to(self.device).eval()

        # Generation parameters (model-agnostic)
        self.zs = zs
        self.z_noise = z_noise
        self.z_noise_truncation = z_noise_truncation
        self.batch_size = batch_size
        self.gen_batch_size = generation_batch_size if generation_batch_size is not None else batch_size

        # Generation parameters (model-specific)
        self.r_light = r_light
        self.r_dark = r_dark

        # Mask parameters
        self.mask_gen_method_light = mask_gen_method_light
        self.mask_gen_method_dark = mask_gen_method_dark
        self.filter_cc_threshold_light = filter_cc_threshold_light
        self.filter_cc_threshold_dark = filter_cc_threshold_dark
        self.filter_size_thresholds_light = filter_size_thresholds_light
        self.filter_size_thresholds_dark = filter_size_thresholds_dark
        self.filter_histogram_bins_light = filter_histogram_bins_light
        self.filter_histogram_bins_dark = filter_histogram_bins_dark
        self.filter_roundness_threshold_light = filter_roundness_threshold_light
        self.filter_roundness_threshold_dark = filter_roundness_threshold_dark
        self.filter_overlap_threshold = filter_overlap_threshold

    @torch.no_grad()
    def gen_samples(self, z=None, batch_size=None, idx=None):
        """Create images and shifted images by generating from G"""
        assert (z is None) ^ (batch_size is None), 'Expected exactly 1 of z or batch_size'
        if z is None:  # make noise if it is not provided
            z = self.make_noise(batch_size, idx).to(self.device)

        # Generate image with shifted latent vector
        d_light = self.model_light(z)
        d_dark = self.model_dark(z)

        # Generate the original and shifted images, dealing with the conditional and
        # unconditional cases separately
        assert self.r_light == self.r_dark
        if self.G.conditional:
            img, y = self.G(z, return_y=True)
            img_light = self.G(z + self.r_light * d_light, y=y)
            img_dark = self.G(z + self.r_dark * d_dark, y=y)
            # img_new = self.G(z + self.r_light * d_light + self.r_dark * d_dark)
            y = onehot_to_int(y)  # convert label to an integer, in case it is one-hot
        else:
            img = self.G(z)
            img_light = self.G(z + self.r_light * d_light)
            img_dark = self.G(z + self.r_dark * d_dark)
            # img_new = self.G(z + self.r_light * d_light + self.r_dark * d_dark)
            y = torch.zeros(len(z), device=self.device, dtype=torch.long)  # dummy label
        return dict(img=img, img_light=img_light, img_dark=img_dark, y=y)

    @retry(Exception, tries=25, delay=0.01)
    def _get_partial_batch(self, idx=None):

        # Generate samples and store in a SimpleNamespace
        s = self.gen_samples(batch_size=self.gen_batch_size, idx=idx)
        s = SimpleNamespace(**{k: to_cpu(v) for k, v in s.items()})

        # Convert images to mask
        method = self.mask_gen_method_light
        s.mask_light = convert_images_to_mask(s.img, s.img_light, method).cpu()
        s.mask_dark = convert_images_to_mask(s.img, s.img_dark, method, invert=True).cpu()
        s.mask = convert_images_to_mask(s.img * 0, s.img_light - s.img_dark, method).cpu()

        # Filter connected components in mask
        if self.filter_cc_threshold_light > 0.0:
            s.mask_light = apply_connected_components_filter(s.mask_light, threshold=self.filter_cc_threshold_light)
        if self.filter_cc_threshold_dark > 0.0:
            s.mask_dark = apply_connected_components_filter(s.mask_dark, threshold=self.filter_cc_threshold_dark)
        if self.filter_cc_threshold_light > 0.0:
            s.mask = apply_connected_components_filter(s.mask, threshold=self.filter_cc_threshold_light)

        # Filter generated images by area
        if self.filter_size_thresholds_light or self.filter_size_thresholds_dark:
            indices_light = get_area_filter_indices(s.mask_light, thresholds=self.filter_size_thresholds_light)
            indices_dark = get_area_filter_indices(s.mask_dark, thresholds=self.filter_size_thresholds_dark)
            indices = indices_dark & indices_light
            select_indices(s, indices, error_msg='No images with masks of sufficient area')

        # Filter generated images by intensity histogram
        if self.filter_histogram_bins_light > 0:
            indices_light = get_histogram_filter_indices(s.img_light, bins=self.filter_histogram_bins_light)
            indices_dark = get_histogram_filter_indices(s.img_dark, bins=self.filter_histogram_bins_dark)
            indices = indices_dark & indices_light
            select_indices(s, indices, error_msg='No images with histogram of sufficient intensity')

        # Filter generated images by roundness
        if self.filter_roundness_threshold_light > 0:
            indices_light = get_roundness_filter_indices(s.mask_light, threshold=self.filter_roundness_threshold_light)
            indices_dark = get_roundness_filter_indices(s.mask_dark, threshold=self.filter_roundness_threshold_dark)
            indices = indices_dark & indices_light
            select_indices(s, indices, error_msg='No images with sufficiently round masks')

        # Filter generated images by roundness
        if self.filter_overlap_threshold > 0:
            indices = get_overlap_filter_indices(s.mask_light, s.mask_dark, threshold=self.filter_overlap_threshold)
            select_indices(s, indices, error_msg='No images with sufficient overlap')

        # Return generated images/masks/labels/etc
        partial_batch = [dict(zip(s.__dict__.keys(), v)) for v in zip(*(s.__dict__.values()))]
        return partial_batch


def create_gan_dataset(_cfg: DictConfig):
    r"""Create a GAN dataset from a config dictionary. Note that this is not the full
        config, only the part corresponding to the GAN dataset (e.g. cfg.data_gen)"""

    # Devices
    print(f'GAN dataset device: {_cfg.device}')
    device = torch.device(_cfg.device)

    # Load GAN
    G = make_gan(gan_type=_cfg.generator.gan_type, **_cfg.generator.kwargs)
    G.eval().to(device)

    # Load dataset embeddings if specified by the user
    if _cfg.zs is None:
        zs = None
        print('Generating z ~ N(0,1)')
    else:
        zs = torch.from_numpy(np.load(_cfg.zs)).to(device)
        print(f'Generating z from file: {_cfg.zs}')
        print(f'Size of zs: {zs.shape}')

    # Whether or not we are building a dual-direction dataset
    is_dual = hasattr(_cfg, 'model_light_checkpoint')

    def load_checkpoint(model: nn.Module, checkpoint_file: Path):
        print(f'Loading model from checkpoint: {checkpoint_file}')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)

    # Create model(s) and GAN training dataset
    init_shape = G.sample_latent(batch_size=1, device=device).shape
    if is_dual:
        model_light = MODELS[_cfg.model_type](shape=init_shape).to(device)
        model_dark = MODELS[_cfg.model_type](shape=init_shape).to(device)
        load_checkpoint(model_light, Path(_cfg.model_light_checkpoint))
        load_checkpoint(model_dark, Path(_cfg.model_dark_checkpoint))
        gan_train_dataset = DualGANDataset(G=G, model_light=model_light, model_dark=model_dark, zs=zs, **_cfg.kwargs)
        for p in itertools.chain(model_light.parameters(), model_dark.parameters(), G.parameters()):
            p.requires_grad_(False)
    else:
        model = MODELS[_cfg.model_type](shape=init_shape).to(device)
        load_checkpoint(model, Path(_cfg.checkpoint))
        gan_train_dataset = GANDataset(G=G, model=model, zs=zs, **_cfg.kwargs)
        for p in itertools.chain(model.parameters(), G.parameters()):
            p.requires_grad_(False)

    # Manually set length of training dataset (i.e. epoch length)
    gan_train_dataset.length = _cfg.iters_per_epoch
    return gan_train_dataset
