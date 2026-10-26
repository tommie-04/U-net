# U-Net Reproduction on Carvana

After reading [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597),
I wanted to understand how a network predicts an object's outline. I used
[milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) to practice
Python, PyTorch, and the complete training workflow. Since my MacBook has no
NVIDIA GPU, I rented an AutoDL RTX 3080 Ti for training.

The [project report](reports/REPRODUCTION_REPORT.md) covers the experiment and
what I learned. [Experiment details](reports/EXPERIMENT_DETAILS.md) contain the
training settings and recorded files.

## Results

| Item | Result |
| --- | --- |
| Dataset | 5,088 Carvana image/mask pairs |
| Training / validation | 4,580 / 508 images |
| Training | 5 epochs, batch size 1, scale 0.5, FP32 |
| Final validation Dice (epoch 5) | **0.968135** |
| Best saved checkpoint | **Epoch 3: validation Dice 0.981803** |
| Highest logged validation Dice | 0.990044 |
| Run time | **1 h 7 min 39 s** |

![Training loss and validation Dice](reports/figures/training_curves.png)

The first AMP attempt produced NaN loss. Restarting with a fresh model in FP32
completed all five epochs.

## Dataset

Download the training data from the
[Carvana Image Masking Challenge on Kaggle](https://www.kaggle.com/c/carvana-image-masking-challenge).
Only these two archives are needed:

| File | Purpose | Extracted folder in the official repository |
| --- | --- | --- |
| `train_hq.zip` | RGB car images used as model inputs | `data/imgs/` |
| `train_masks.zip` | Reference masks marking car and background pixels | `data/masks/` |

For example, `a1038b207299_14.jpg` pairs with `a1038b207299_14_mask.gif`.
Place the files directly in these folders. The training script creates the
training/validation split. Download the archives on the cloud instance, or
upload them after downloading through the Kaggle website.

## Training

Clone the [official repository](https://github.com/milesial/Pytorch-UNet), prepare
its dependencies and dataset, then run from that repository's directory:

```bash
export WANDB_MODE=offline
python -u train.py --epochs 5 --batch-size 1 --learning-rate 1e-5 --scale 0.5 --validation 10
```

The community image supplied the Python/PyTorch environment. Training used a
separate official checkout, the original Carvana files, and randomly initialized
weights. Environment versions are listed in [the environment record](docs/ENVIRONMENT_STATUS.md).

## Download the trained model

The selected model is available in [release v1.0.0](https://github.com/tommie-04/U-net/releases/tag/v1.0.0):

- [checkpoint_epoch3.pth](https://github.com/tommie-04/U-net/releases/download/v1.0.0/checkpoint_epoch3.pth) — approximately 118.5 MiB.
- [SHA256 checksum](https://github.com/tommie-04/U-net/releases/download/v1.0.0/checkpoint_epoch3.pth.sha256).

From the official `Pytorch-UNet` directory:

```bash
mkdir -p checkpoints
curl --fail --location https://github.com/tommie-04/U-net/releases/download/v1.0.0/checkpoint_epoch3.pth --output checkpoints/checkpoint_epoch3.pth
curl --fail --location https://github.com/tommie-04/U-net/releases/download/v1.0.0/checkpoint_epoch3.pth.sha256 --output checkpoints/checkpoint_epoch3.pth.sha256
(cd checkpoints && shasum -a 256 -c checkpoint_epoch3.pth.sha256)
```

## Try a single image

With the model and an input image ready, run:

```bash
mkdir -p predictions
python predict.py --model checkpoints/checkpoint_epoch3.pth --input data/imgs/a1038b207299_14.jpg --output predictions/a1038b207299_14_pred.png --scale 0.5
```

Replace the input and output paths to try another RGB image. A reference mask
is useful for checking how closely the prediction follows the correct shape.

### Prediction comparison

![Original image, reference mask, and epoch-3 prediction](reports/figures/prediction_comparison_epoch3.png)

White marks the car and black marks the background. On this image, epoch 3
achieved Dice **0.9908**, compared with **0.9337** for epoch 5. Epoch 3 follows
the car's outline more closely and removes most of the extra background.

See the [side-by-side model comparison](reports/figures/original_sample_epoch_comparison.png)
and [nine-image results](reports/CHECKPOINT_COMPARISON.md).

## Project files

| Folder | Contents |
| --- | --- |
| `reports/` | Project report, training settings, and comparison figures |
| `records/` | Metrics, environment snapshots, checksums, and failure records |
| `scripts/` | Code for comparing saved checkpoints |
| `docs/` | Cloud training environment |
