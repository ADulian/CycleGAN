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
You can download the dataset directly from the following Kaggle's [website](https://www.kaggle.com/competitions/gan-getting-started/data). This implementation uses the `.jpg` files rather than `tfrec` by default. The structure of your dataset folder should look as follows

```
- data/
  - monet_jpg/  # All monet images
  - photo_jpg/  # All photos
```

Once you've got your dataset and virtualenv all in place, navigate to the project dir and run `main.py` with relevant arguments. For example, the following will train the model on a `batch_size=64` and for `num_epochs=10` whilst loading data from `data_root=/data` 
```
cd CycleGAN/
python main.py --data_root=/data --batch_size=64 --num_epochs=10
```

### Weight and Biases
Running with Weights and Biases, also setting `wandb.watch(model)` to `True`
```
python main.py --wandb=True --wandb_watch=True
```
If you're not using `main.py` you can use `Trainer` class as standalone manager for training the model. The standard way of fitting with `Trainer` is by calling `Trainer.fit(**kwargs)`, however, if you want to run `fit` with `wandb` then instead use `Trainer.fit_wandb(**kwargs)` which just wraps `fit(**kwargs)` into the `wandb` context manager that initialises `wandb` project with a default name of `Monet_CycleGAN` which you can either modify or make the project in your `wandb` account.

## Results
Checkout the `results` directory to see some sample images from various experiments and see how the model evolved over time.

## ToDo
- [ ] Discriminators History
- [ ] Identity Loss
- [ ] Better Architecture, acutally based on the paper

## CycleGAN Output Explorer
I thought it might be useful to be able to visualise the output of CycleGAN in sort of a live mode. Therefore, I decided to write a simple and interactive tool to explore how the input of the Generator gets transformed from domain `A` to `B` and back to `A` (cycle) and also see how the Discriminator judges its input. In a case of a CycleGAN the output of a Discriminator is not a single value (`True/False` or `Real/Fake`), but rather a grid. That grid can be visualised to see which areas of an input are more likely to be judged as `Real/Fake`. 

Below is a small demonstration of the tool. To use it, simply run the following notebook `notebooks/qualitative_eval.ipynb`.
![CycleGAN Explorer](/figures/cycle_gan_explorer.gif)
