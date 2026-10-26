# My First U-Net Reproduction: Car Segmentation with PyTorch

## Why I chose this project

After reading [*U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597)
by Olaf Ronneberger, Philipp Fischer, and Thomas Brox, I became interested in how
a network can predict the outline of an object. Instead of giving one label to
an entire image, segmentation assigns a class to each pixel. I wanted to see
how this works with a real dataset and what the predicted masks would look like.

The part of U-Net that caught my attention was the connection between its two
paths. The contracting path reduces spatial resolution and learns image
features. The expanding path increases the resolution again, while using
features from the contracting path to help locate boundaries. I wanted to
connect this idea from the paper with the tensors and operations in actual code.
The architecture and these connections are described in the
[original paper](https://arxiv.org/html/1505.04597v1).

I then found [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet),
which provides a PyTorch implementation and instructions for training on the
Carvana dataset. Cars and their masks seemed like a task whose results I could
inspect directly, so I used this repository as the basis of a small reproduction.
My goal was to understand the workflow and become more comfortable with Python
and PyTorch before trying to change the model.

The paper studies biomedical images, while this repository adapts U-Net to car
segmentation. My project reproduces the repository's training workflow on
Carvana; it does not repeat the paper's biomedical experiments.

## Setting up the experiment

My MacBook does not have an NVIDIA GPU, so I rented a cloud GPU from AutoDL to
run the repository's CUDA training workflow. I kept the source and project
notes on my Mac and used the cloud terminal for data preparation and training.
This was my first experience using a rented GPU for a project.

I chose an RTX 3080 Ti with 12 GB of GPU memory. I used a community image for
the existing Python and PyTorch environment, but cloned the original GitHub
repository separately. I did not use the image publisher's modified code,
prepared data subset, or existing model weights. I fixed the repository version
to `21d7850f2af30a9695bbeea75f3136aa538cfc4a` so I could record exactly
which source I used.

| Component | Environment used |
| --- | --- |
| Platform | AutoDL Linux container |
| GPU | NVIDIA GeForce RTX 3080 Ti, 12 GB |
| Python | 3.8.10 |
| PyTorch / torchvision | 2.0.0+cu118 / 0.15.1+cu118 |
| PyTorch CUDA build | 11.8 |
| NumPy / Pillow | 1.23.5 / 9.3.0 |
| Matplotlib / tqdm / W&B | 3.6.2 / 4.64.1 / 0.13.5 |

I checked the installed versions and CUDA access before training. I also learned
that the CUDA version shown by the NVIDIA driver is not necessarily the CUDA
build used by PyTorch. I did not need to install Docker on my Mac or move my
Mac's Python environment to the server.

## Preparing data and running training

I used the original images and masks from the
[Kaggle Carvana competition](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data).
The repository's download workflow uses `train_hq.zip` and `train_masks.zip`.
I downloaded the images through the Kaggle CLI. When the mask download stalled,
I downloaded that ZIP in my browser and uploaded it to the cloud instance.
Both archives passed integrity checks.

After extraction, I checked that all 5,088 images had matching masks. The mask
values were 0 for background and 1 for the car. The original loader split the
data into 4,580 training images and 508 validation images.

I kept the repository's model, loss, and optimizer unchanged. The model takes an
RGB image and produces two class scores for each pixel. I used batch size 1,
image scale 0.5, learning rate 1e-5, and five epochs. One epoch means going through
the training set once, so this run made 22,900 parameter updates. I trained from
random weights rather than loading the repository's pretrained model.

The first attempt used automatic mixed precision (AMP), following the quick
start. However, the loss became `NaN`, and validation Dice stayed close to zero.
A saved checkpoint contained 100 tensors with non-finite values. This showed me
that saving a checkpoint does not mean a model has trained successfully.

I stopped that run and kept its logs and weights. A short test with a fresh
model in full precision (FP32) completed ten updates without non-finite values.
I then started a new five-epoch run without `--amp`, keeping the other training
settings unchanged. This run completed successfully. I have not established the
exact cause of the first failure, so I cannot conclude that AMP itself was the
only cause.

The successful training command was:

```bash
export WANDB_MODE=offline
python -u train.py --epochs 5 --batch-size 1 --learning-rate 1e-5 --scale 0.5 --validation 10
```

I used `nohup` to keep training running if my terminal disconnected, and saved
the output to a log file. W&B recorded the experiment in offline mode. The
successful process took **1 hour 7 minutes 39 seconds**, including data
initialization and logging shutdown. This time does not include downloads,
setup, or the failed attempt.

## Results and what I noticed

| Epoch | Last validation Dice in that epoch |
| --- | --- |
| 1 | 0.915408 |
| 2 | 0.940063 |
| 3 | 0.981803 |
| 4 | 0.971803 |
| 5 | 0.968135 |

Dice measures the overlap between the predicted car region and the reference
mask; a value closer to 1 means more overlap. Here it is the repository's
internal validation foreground Dice at the resized resolution, not pixel
accuracy or a Kaggle test score.

![Training loss samples and validation Dice](figures/training_curves.png)

The displayed batch losses generally decreased, although there were spikes.
Validation Dice improved overall but did not increase at every check. The final
value was **0.968135**, and the highest logged value was **0.990044** during
epoch 5. That highest point was not saved as a separate checkpoint, because the
original code saves models only at the end of each epoch. The loss plot uses
samples from terminal progress updates, not epoch-average losses.

After training, I checked that all five checkpoints contained only finite
floating-point values. I then loaded the final checkpoint with the original
`predict.py` script and saved a predicted mask.

![Original image, reference mask, and selected epoch-3 prediction](figures/prediction_comparison_epoch3.png)

The first prediction used the final checkpoint, epoch 5, and included too much
background on both sides of the car. I then compared it with epoch 3, which had
the highest logged validation Dice at an epoch boundary among the five saved
models. Using the same image, the original-resolution Dice improved from
**0.9337 with epoch 5 to 0.9908 with epoch 3**. The figure above shows epoch 3;
the [epoch-5 result](figures/prediction_comparison.png) is still available.

This helped me understand why the last saved model is not always the one I
should choose for prediction. The choice of epoch 3 comes from its logged
validation result; I did not retrain or edit the predicted mask. These two
single-image scores use the original image resolution and are separate from
the training script's validation averages.

I kept the original image instead of replacing it with a different car. Its
actual split membership was not saved, so I cannot call it a held-out example.
I also compared the two checkpoints on a fixed set of nine images; the full
scores and local CPU inference setup are in the
[comparison notes](CHECKPOINT_COMPARISON.md). Those samples are descriptive and
do not provide a new independent test score.

## What I learned

I became more familiar with using Python libraries together for a complete
experiment. I used Pillow and NumPy to inspect images and masks, PyTorch to
follow the data and model outputs, and Matplotlib to plot the recorded results.
The input shape `[1, 3, 640, 959]` and output shape `[1, 2, 640, 959]` made the
pixel-classification task easier for me to understand: the output has a score
for each of the two classes at every image location.

The cloud setup also taught me to separate editing from computation. Cloning a
repository on my Mac does not automatically make it use a rented GPU. The
training process, its Python environment, and the dataset all had to be on the
cloud instance. Closing a browser or terminal also does not shut down that
instance.

The failed run was an important part of the project. I learned to read the loss
and validation output early, stop a run that is producing invalid values, and
check the saved weights before using them. I also learned to keep a failed
attempt separate from a new run instead of overwriting its evidence.

Finally, I practiced recording package versions, commands, logs, metrics, and
model files. I downloaded a backup and used SHA256 checksums to verify that it
matched the cloud copy. This gave me a record I could inspect after training,
rather than depending on a terminal window or temporary cloud session.

## Limits and what I would do next

This was one successful run, and I did not change the architecture or reproduce
the original paper's results. I also did not reproduce the repository author's
competition score. I used the original image-level split, so different views
of the same car may be present in both training and validation.

I did not save the actual split IDs during training. Although the split uses
seed 0, the loader does not sort the input filenames, so that seed alone is not
enough to recreate the same split on another machine. If I repeat the experiment,
I would save the split IDs and random seeds before training, compare saved
checkpoints, and inspect more prediction examples with known split membership.

For this first reproduction, the main result is that I completed and recorded
the path from downloading data to training, loading my own model, and examining
its predictions. I still need more practice explaining the training code and
the causes of numerical failures.

The [experiment details](EXPERIMENT_DETAILS.md) contain the full configuration,
metric definitions, source and checkpoint records, and backup checks.
