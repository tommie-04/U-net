# My First U-Net Reproduction: Car Segmentation with PyTorch

## Why I chose this project

After reading [*U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597)
by Olaf Ronneberger, Philipp Fischer, and Thomas Brox, I became interested in how
a network can predict an object's outline. Segmentation assigns a class to each
pixel, so I could directly compare its output with the shape in an image.

The connection between U-Net's two paths caught my attention. The contracting
path reduces spatial resolution and learns features. The expanding path uses
those features together with information from earlier layers to locate
boundaries. I wanted to connect that idea with actual Python code and tensors.

I found [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet), which
provides a PyTorch implementation and a Carvana training example. I chose car
segmentation because the images and masks were easy to inspect. My aim was to
complete a small reproduction and become more familiar with Python and PyTorch.

## Using a cloud GPU

My MacBook does not have an NVIDIA GPU, so I rented an AutoDL RTX 3080 Ti with
12 GB of GPU memory. I used my Mac for editing and reviewing results, and the
cloud terminal for data preparation and training.

The instance already had a Python/PyTorch environment. I cloned the official
repository separately and checked the package versions and CUDA access. This
helped me understand the difference between having code on my Mac and actually
running it on a remote GPU. The [environment record](../docs/ENVIRONMENT_STATUS.md)
lists the versions used.

## Preparing data and training

I downloaded `train_hq.zip` and `train_masks.zip` from the
[Kaggle Carvana competition](https://www.kaggle.com/c/carvana-image-masking-challenge).
The image download worked through the Kaggle CLI. When the mask download stalled,
I downloaded it in my browser and uploaded it to AutoDL.

I checked both ZIP files and matched all 5,088 images with their masks. Mask
values were 0 for background and 1 for the car. The original loader created
4,580 training images and 508 validation images. I kept its image-level split;
the individual split IDs were not retained.

I used the repository's model, loss, and optimizer with batch size 1, image scale
0.5, and learning rate 1e-5. The model was trained from random initialization
for five epochs, making 22,900 parameter updates.

My first attempt used automatic mixed precision (AMP). The loss became `NaN`,
and the epoch-1 checkpoint contained 100 tensors with non-finite values. I
stopped the run and tried ten updates with a fresh FP32 model. That test passed,
so I restarted the five-epoch run with AMP disabled. The FP32 run completed.

I used `nohup` so training could continue after disconnecting the terminal,
saved the output to a log file, and recorded W&B data offline. The successful
process took **1 hour 7 minutes 39 seconds**, including dataset initialization
and logging shutdown. The command and model settings are in
[experiment details](EXPERIMENT_DETAILS.md).

## Results

| Epoch | Last validation Dice in that epoch |
| --- | --- |
| 1 | 0.915408 |
| 2 | 0.940063 |
| 3 | 0.981803 |
| 4 | 0.971803 |
| 5 | 0.968135 |

Dice measures overlap between the predicted car region and its reference mask.
The training script evaluates foreground Dice at the resized resolution.
The [training curves](../README.md#results) show loss decreasing while validation
Dice rises and then fluctuates. The final value was **0.968135**; the highest
logged value during training was **0.990044**.

I reloaded the final checkpoint with `predict.py` and noticed extra background
in its predicted car region. I then compared it with epoch 3, which had the
highest validation Dice among the saved epoch-end models. On the same image,
original-resolution Dice was **0.9908 for epoch 3** and **0.9337 for epoch 5**.
The [prediction example](../README.md#prediction-comparison) shows the epoch-3
result. I also compared both models on nine images; the
[comparison table](CHECKPOINT_COMPARISON.md) contains those scores.

This helped me understand that the last checkpoint is not always the best one
to use. I selected epoch 3 for the downloadable model and kept the other weights
in my backup.

## What I learned

I practiced using Pillow and NumPy to inspect images and masks, PyTorch to work
with tensors and model outputs, and Matplotlib to plot training records. Seeing
the input shape `[1, 3, 640, 959]` and output shape `[1, 2, 640, 959]` made the
task clearer: the model produces two class scores at each pixel location.

The failed run taught me to inspect loss and validation output early. I also
learned to check saved weights before using them and to keep separate records
when restarting an experiment.

Using a rented GPU introduced me to remote Python environments, file uploads,
background processes, and model downloads. After training, I backed up the
logs, metrics, plots, and weights, then verified the files with SHA256 checksums.

For a next experiment, I would save the split IDs and random seeds before
training and compare more prediction examples. For this project, I completed
the full workflow from acquiring data to training a model and examining its
predictions.
