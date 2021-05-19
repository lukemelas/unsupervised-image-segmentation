// Project title
export const title = "Finding an Unsupervised Image Segmenter in each of your Deep Generative Models"

// Short version of the abstract
export const description = "We propose a method to automatically find a universal latent direction in a GAN that can separate the foreground from the background. We can then generate an unlimited supply of samples with masks to train a segmentation network. The whole process is automatic and unsupervised and achieves state-of-the-art unsupervised segmentation performance."

// Abstract
export const abstract = "Recent research has shown that numerous human-interpretable directions exist in the latent space of GANs. In this paper, we develop an automatic procedure for finding directions that lead to foreground-background image separation, and we use these directions to train an image segmentation model without human supervision. Our method is generator-agnostic, producing strong segmentation results with a wide range of different GAN architectures. Furthermore, by leveraging GANs pretrained on large datasets such as ImageNet, we are able to segment images from a range of domains without further training or finetuning. Evaluating our method on image segmentation benchmarks, we compare favorably to prior work while using neither human supervision nor access to the training data. Broadly, our results demonstrate that automatically extracting foreground-background structure from pretrained deep generative models can serve as a remarkably effective substitute for human supervision."

// Institutions
export const institutions = {
  1: "Oxford University"
}

// Authors
export const authors = [
  {
    'name': 'Luke Melas-Kyriazi',
    'institutions': [1],
    'url': "https://github.com/lukemelas/"
  },
  {
    'name': 'Christian Rupprecht',
    'institutions': [1],
    'url': "https://chrirupp.github.io/"
  },
  {
    'name': 'Iro Laina',
    'institutions': [1],
    'url': "http://campar.in.tum.de/Main/IroLaina"
  },
  {
    'name': 'Andrea Vedaldi',
    'institutions': [1],
    'url': "https://www.robots.ox.ac.uk/~vedaldi/"
  }
]

// Links
export const links = {
  'paper': "https://arxiv.org/abs/2105.08127",
  'github': "https://github.com/lukemelas/unsupervised-image-segmentation"
}

// Acknowledgements
export const acknowledgements = "C. R. is supported by Innovate UK (project 71653) on behalf of UK Research and Innovation (UKRI) and by the European Research Council (ERC) IDIU-638009. I. L. is supported by the EPSRC programme grant Seebibyte EP/M013774/1 and ERC starting grant IDIU-638009. A. V. is funded by ERC grant IDIU-638009."

// Citation
export const citationId = "melaskyriazi2021finding"
export const citationAuthors = "Luke Melas-Kyriazi and Christian Rupprecht and Iro Laina and Andrea Vedaldi"
export const citationYear = "2021"
export const citationBooktitle = "Arxiv"

// Video
export const video_url = "https://www.youtube.com/embed/ScMzIvxBSi4"