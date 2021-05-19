<div align="center">    
 
## Finding an Unsupervised Image Segmenter in each of your Deep Generative Models

[![Paper](http://img.shields.io/badge/paper-arxiv.2105.08127-B31B1B.svg)](https://arxiv.org/abs/2105.08127)
<!-- [![Conference](http://img.shields.io/badge/CVPR-2021-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018) -->

</div>

<!-- TODO: Add video -->

### Description
Recent research has shown that numerous human-interpretable directions exist in the latent space of GANs. In this paper, we develop an automatic procedure for finding directions that lead to foreground-background image separation, and we use these directions to train an image segmentation model without human supervision. Our method is generator-agnostic, producing strong segmentation results with a wide range of different GAN architectures. Furthermore, by leveraging GANs pretrained on large datasets such as ImageNet, we are able to segment images from a range of domains without further training or finetuning. Evaluating our method on image segmentation benchmarks, we compare favorably to prior work while using neither human supervision nor access to the training data. Broadly, our results demonstrate that automatically extracting foreground-background structure from pretrained deep generative models can serve as a remarkably effective substitute for human supervision. 

### How to run   

#### Dependencies

This code depends on [pytorch-pretrained-gans](https://github.com/lukemelas/pytorch-pretrained-gans), a repository I developed that exposes a standard interface for a variety of pretrained GANs. Install it with:
```bash
pip install git+https://github.com/lukemelas/pytorch-pretrained-gans
```
The pretrained weights for most GANs are downloaded automatically. For those that are not, I have provided scripts in that repository. 

There are also some standard dependencies:
 - PyTorch (tested on version 1.7.1, but should work on any version)
 - [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
 - [Hydra](https://github.com/facebookresearch/hydra) 1.1
 - [Albumentations](https://github.com/albumentations-team/albumentations)
 - [Kornia](https://github.com/kornia/kornia)
 - [Retry](https://github.com/invl/retry)
 - [Optional] [Weights and Biases](https://wandb.ai/)
 
 Install them with:
 ```bash
pip install hydra-core==1.1.0dev5 pytorch_lightning albumentations tqdm retry kornia
 ```
 

#### General Approach

Our unsupervised segmentation approach has two steps: (1) finding a good direction in latent space, and (2) training a segmentation model from data and masks that are generated using this direction. 

In detail, this means:
 1. We use `optimization/main.py` finds a salient direction (or two salient directions) in the latent space of a given pretrained GAN that leads to foreground-background image separation. 
 2. We use `segmentation/main.py` to train a standard segmentation network (a UNet) on generated data. The data can be generated in two ways: (1) you can generate the images on-the-fly during training, or (2) you can generate the images before training the segmentation model using `segmentation/generate_and_save.py` and then train the segmentation network afterward. The second approach is faster, but requires more disk space (~10GB for 1 million images). We will also provide a pre-generated dataset (coming soon). 

#### Configuration and Logging
We use Hydra for configuration and Weights and Biases for logging. With Hydra, you can specify a config file (found in `configs/`) with `--config-name=myconfig.yaml`. You can also override the config from the command line by specifying the overriding arguments (without `--`). For example, you can enable Weights and Biases with `wandb=True` and you can name the run with `name=myname`. 

The structure of the configs is as follows: 
```bash
config
├── data_gen
│   ├── generated.yaml  # <- for generating data with 1 latent direction
│   ├── generated-dual.yaml   # <- for generating data with 2 latent directions
│   ├── generator  # <- different types of GANs for generating data
│   │   ├── bigbigan.yaml
│   │   ├── pretrainedbiggan.yaml
│   │   ├── selfconditionedgan.yaml
│   │   ├── studiogan.yaml
│   │   └── stylegan2.yaml 
│   └── saved.yaml  # <- for using pre-generated data
├── optimize.yaml  # <- for optimization
└── segment.yaml   # <- for segmentation
```

#### Code Structure

The code is structured as follows:
```bash
src
├── models  # <- segmentation model
│   ├── __init__.py
│   ├── latent_shift_model.py  # <- shifts direction in latent space
│   ├── unet_model.py  # <- segmentation model
│   └── unet_parts.py
├── config  # <- configuration, explained above
│   ├── ... 
├── datasets  # <- classes for loading datasets during segmentation/generation
│   ├── __init__.py
│   ├── gan_dataset.py  # <- for generating dataset
│   ├── saved_gan_dataset.py  # <- for pre-generated dataset
│   └── real_dataset.py  # <- for evaluation datasets (i.e. real images)
├── optimization
│   ├── main.py  # <- main script
│   └── utils.py  # <- helper functions
└── segmentation
    ├── generate_and_save.py  # <- for generating a dataset and saving it to disk
    ├── main.py  # <- main script, uses PyTorch Lightning 
    ├── metrics.py  # <- for mIoU/F-score calculations
    └── utils.py  # <- helper functions
```

#### Datasets

The datasets should have the following structure. You can easily add you own datasets or use only a subset of these datasets by modifying `config/segment.yaml`. You should specify your directory by modifying `root` in that file on line 19, or by passing `data_seg.root=MY_DIR` using the command line whenever you call `python segmentation/main.py`. 

```bash
├── DUT_OMRON
│   ├── DUT-OMRON-image
│   │   └── ...
│   └── pixelwiseGT-new-PNG
│       └── ...
├── DUTS
│   ├── DUTS-TE
│   │   ├── DUTS-TE-Image
│   │   │   └── ...
│   │   └── DUTS-TE-Mask
│   │       └── ...
│   └── DUTS-TR
│       ├── DUTS-TR-Image
│       │   └── ...
│       └── DUTS-TR-Mask
│           └── ...
├── ECSSD
│   ├── ground_truth_mask
│   │   └── ...
│   └── images
│       └── ...
├── CUB_200_2011
│   ├── train_images
│   │   └── ...
│   ├── train_segmentations
│   │   └── ...
│   ├── test_images
│   │   └── ...
│   └── test_segmentations
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

The datasets can be downloaded from:
* [DUT_OMRON](http://saliencydetection.net)
* [DUTS](http://saliencydetection.net/duts/)
* [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
* [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
* [Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

#### Training

Before training, make sure you understand the general approach (explained above). 

_Note:_ All commands are called from within the `src` directory. 

In the example commands below, we use BigBiGAN. You can easily switch out BigBiGAN for another model if you would like to. 

**Optimization**
```bash
PYTHONPATH=. python optimization/main.py data_gen/generator=bigbigan name=NAME
```
This should take less than 5 minutes to run. The output will be saved in `outputs/optimization/fixed-BigBiGAN-NAME/DATE/`, with the final checkpoint in `latest.pth`.

**Segmentation with precomputed generations**

The recommended way of training is to generate the data first and train afterward. An example generation script would be: 
```bash
PYTHONPATH=. python segmentation/generate_and_save.py \
name=NAME \
data_gen=generated \
data_gen/generator=bigbigan \
data_gen.checkpoint="YOUR_OPTIMIZATION_DIR_FROM_ABOVE/latest.pth" \
data_gen.save_dir="YOUR_OUTPUT_DIR" \
data_gen.save_size=1000000 \
data_gen.kwargs.batch_size=1 \
data_gen.kwargs.generation_batch_size=128
```
This will generate 1 million image-label pairs and save them to `YOUR_OUTPUT_DIR/images`. Note that `YOUR_OUTPUT_DIR` should be an _absolute path_, not a relative one, because Hydra changes the working directory. You may also want to tune the `generation_batch_size` to maximize GPU utilization on your machine. It takes around 3-4 hours to generate 1 million images on a single V100 GPU.

Once you have generated data, you can train a segmentation model:
```bash
PYTHONPATH=. python segmentation/main.py \
name=NAME \
data_gen=saved \
data_gen.data.root="YOUR_OUTPUT_DIR_FROM_ABOVE"
```
It takes around 3 hours on 1 GPU to complete 18000 iterations, by which point the model has converged (in fact you can probably get away with fewer steps, I would guess around ~5000). 

**Segmentation with on-the-fly generations** 

Alternatively, you can generate data while training the segmentation model. An example script would be: 
```bash
PYTHONPATH=. python segmentation/main.py \
name=NAME \
data_gen=generated \
data_gen/generator=bigbigan \
data_gen.checkpoint="YOUR_OPTIMIZATION_DIR_FROM_ABOVE/latest.pth" \
data_gen.kwargs.generation_batch_size=128
```


#### Evaluation
To evaluate, set the `train` argument to False. For example:
```bash
python train.py \
name="eval" \
train=False \
eval_checkpoint=${checkpoint} \
data_seg.root=${DATASETS_DIR} 
```


#### Pretrained models
 * ... are coming soon!


#### Available GANs

It should be possible to use any GAN from [pytorch-pretrained-gans](https://github.com/lukemelas/pytorch-pretrained-gans), including:
 - [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
 - [BigBiGAN](https://arxiv.org/abs/1907.02544)
 - [Many GANs from StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN)
 - [Self-Conditioned GANs](https://arxiv.org/abs/2006.10728)
 - [StyleGAN-2-ADA](https://arxiv.org/abs/1912.04958)


#### Citation   
```bibtex
@inproceedings{melaskyriazi2021finding,
  author    = {Melas-Kyriazi, Luke and Rupprecht, Christian and Laina, Iro and Vedaldi, Andrea},
  title     = {Finding an Unsupervised Image Segmenter in each of your Deep Generative Models},
  booktitle = arxiv,
  year      = {2021}
}
```
