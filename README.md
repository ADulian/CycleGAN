# CycleGAN
![CycleGAN Example](/figures/cycle_gan_example.png)
## Introduction
This project was made specifically for the Kaggle Competition [*"I'm something of a Painter Myself"*](https://www.kaggle.com/competitions/gan-getting-started), you can download the Monet dataset from Kaggle to get started.

**CycleGAN** is a powerful technique for unsupervised image-to-image translation, 
which learns to map images from one domain to another without requiring paired examples. 
This repository implements the CycleGAN model as proposed in 2017 in the paper 
[*"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"*](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)
by Jun-Yan Zhu et al.

## Installation
Install required packages with simple
```
pip install -r requirements.txt
```
Note that PyTorch will be installed with cu118 by default as specified in the `requirements.txt`

## Quick start
You can download the dataset directly from the following Kaggle's [website](https://www.kaggle.com/competitions/gan-getting-started/data). This implementation uses the `.jpg` files rather than `tfrec` by default. The structure of your dataset folder should look as follows:

```
- data/
  - monet_jpg/  # All monet images
  - photo_jpg/  # All photos
```

Once you've got your dataset and virtualenv all in place, navigate to the project dir and run `main.py` with relevant arguments. For example, the following will train the model on a `batch_size=64` and for `num_epochs=10` whilst loading data from `data_root=/data`: 
```
cd CycleGAN/
python main.py --data_root=/data --batch_size=64 --num_epochs=10
```

### Weight and Biases
If you want to use Weights and Biases just call `Trainer.fit_wandb(**kwargs)`. This will initialise WandB Project with a default name of `Monet_CycleGAN` so either modify this or make that project in your WandB account.<br> 
To watch the model set `--wandb_watch=True`

## Results

