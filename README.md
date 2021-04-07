- [ ] Update optimize to 

<div align="center">    
 
## Finding an Unsupervised Image Segmenter in each of your Deep Generative Models

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
<!-- [![Conference](http://img.shields.io/badge/CVPR-2021-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018) -->

</div>

<!-- TODO: Add video -->

### Description
Recent research has shown that numerous human-interpretable directions exist in the latent space of GANs. In this paper, we develop an automatic procedure for finding directions that lead to foreground-background image separation, and we use these directions to train an image segmentation model without human supervision. Our method is generator-agnostic, producing strong segmentation results with a wide range of different GAN architectures. Furthermore, by leveraging GANs pretrained on large datasets such as ImageNet, we are able to segment images from a range of domains without further training or finetuning. Evaluating our method on image segmentation benchmarks, we compare favorably to prior work while using neither human supervision nor access to the training data. Broadly, our results demonstrate that automatically extracting foreground-background structure from pretrained deep generative models can serve as a remarkably effective substitute for human supervision. 

### How to run   

#### GAN Dependencies

This code depends on [pytorch-pretrained-gans](), a repository I developed that exposes a standard interface for a variety of pretrained GANs. With this dependency, we can use: 
 - [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
 - [BigBiGAN](https://arxiv.org/abs/1907.02544)
 - [Many GANs from StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN)
 - [Self-Conditioned GANs](https://arxiv.org/abs/2006.10728)
 - [StyleGAN-2-ADA](https://arxiv.org/abs/1912.04958)

#### Standard Dependencies

 - PyTorch (tested on version 1.7.1, but should work on any version)
 - Hydra 1.1: `pip install hydra-core --pre`
 - Other: `pip install albumentations tqdm tensorboard`
 - WandB (optional): `pip install wandb`

#### Code Structure
The code is split into two parts: `optimization` and `segmetnation`. 

The optimization portion of the code finds a salient direction (or two salient directions) in the latent space of a given pretrained GAN that leads to foreground-background image separation. 

The segmentation portion of the code uses the latent direction found in the first stage to generate a synthetic dataset for training a standard segmentation network (a UNet). This can be done in one of two ways: (1) you can generate the images on-the-fly during training, or (2) you can generate the images, save them to disk, and then train the segmentation network separately. 

#### Configuration and Logging
We use Hydra for configuration and Weights and Biases for logging. With Hydra, you can specify a config file (found in `configs/`) with `--config-name=myconfig.yaml`. You can also override the config from the command line by specifying the overriding arguments (without `--`). For example, you can disable Weights and Biases with `wandb=False` and you can name the run with `name=myname`. 

Configs live in `optimization/config` and `segmentation/config`.

#### Code Structure

```bash
├── UNet
│   ├── __init__.py
│   ├── ensemble.py
│   ├── unet_model.py
│   └── unet_parts.py
├── config
│   ├── base.yaml
│   ├── data_gen
│   │   ├── generated-dual.yaml
│   │   ├── generated.yaml
│   │   ├── generator
│   │   │   ├── bigbigan.yaml
│   │   │   ├── pretrainedbiggan.yaml
│   │   │   ├── selfconditionedgan.yaml
│   │   │   ├── studiogan.yaml
│   │   │   └── stylegan2.yaml
│   │   └── saved.yaml
│   ├── default.yaml
├── datasets
│   ├── __init__.py
│   ├── biggan_dataset.py
│   ├── saved_dataset.py
│   └── seg_dataset.py
├── helpers
│   └── generate-dataset.py
├── metrics.py
├── models.py
├── optimization_utils.py
├── optimize.py
├── sweep.py
├── train.py
└── utils.py
```

#### Datasets
The datasets should be placed in a directory we will refer to as `$DATASETS_DIR`. You should then update this 
```bash
├── CUB_200_2011
│   ├── train_images
│   │   └── ...
│   ├── train_segmentations
│   │   └── ...
│   ├── test_images
│   │   └── ...
│   ├── test_segmentations
│   │   └── ...
├── DUT_OMRON
│   ├── DUT-OMRON-image
│   │   └── ...
│   └── pixelwiseGT-new-PNG
│   │   └── ...
├── DUTS
│   ├── DUTS-TE
│   │   ├── DUTS-TE-Image
│   │   │   └── ...
│   │   └── DUTS-TE-Mask
│   │       └── ...
│   ├── DUTS-TR
│   │   ├── DUTS-TR-Image
│   │   │   └── ...
│   │   └── DUTS-TR-Mask
│           └── ...
├── ECSSD
│   ├── ground_truth_mask
│   │   └── ...
│   └── images
│       └── ...
└── Flowers
    ├── train_images
    │   └── ...
    ├── train_segmentations
    │   └── ...
    ├── test_images
    │   └── ...
    └── test_segmentations
        └── ...
```


#### Evaluation
To evaluate, set the `train` argument to False. For example:
```bash
python train.py \
name="eval" \
train=False \
eval_checkpoint=${checkpoint} \
data_seg.root=${DATASETS_DIR} \
data_seg.image_size=${size}
```


#### Pretrained models
 * Name: [Download](url)


#### Citation   
```bibtex
@inproceedings{melaskyriazi2021finding,
  author    = {Melas-Kyriazi, Luke and Rupprecht, Christian and Laina, Iro and Vedaldi, Andrea},
  title     = {Finding an Unsupervised Image Segmenter in each of your Deep Generative Models},
  booktitle = arxiv,
  year      = {2021}
}
```
