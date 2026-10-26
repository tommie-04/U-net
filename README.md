# U-Net Reproduction on Carvana

After reading [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597),
I wanted to understand how an image becomes a pixel-level segmentation mask.
I used [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) to run a
small Carvana reproduction and practice Python, PyTorch, and experiment recording.
Since my MacBook has no NVIDIA GPU, I rented an AutoDL RTX 3080 Ti for training.

The [project report](reports/REPRODUCTION_REPORT.md) describes why I chose the
project, what happened during training, and what I learned. The
[experiment details](reports/EXPERIMENT_DETAILS.md) preserve the technical records.

## Results

| Item | Observed result |
| --- | --- |
| Official source | `21d7850f2af30a9695bbeea75f3136aa538cfc4a`; captured tracked-source diff is empty |
| Data | 5,088 official Carvana training image/mask pairs |
| Training / validation | 4,580 / 508 images, upstream image-level split |
| Successful run | 5 epochs, batch size 1, scale 0.5, FP32 |
| Final internal foreground Dice | **0.968135** |
| Highest logged validation Dice | 0.990044; intermediate weights were not saved |
| Process duration | **1 h 7 min 39 s**, including initialization and log shutdown |
| Saved models | Five finite checkpoints; final model reloaded and used for prediction |

![Training loss samples and internal validation Dice](reports/figures/training_curves.png)

The initial AMP attempt produced NaN loss and non-finite checkpoint tensors.
A fresh FP32 run completed successfully without changing upstream source.

The report includes the failed AMP attempt, successful FP32 run, results,
prediction errors, and limitations of this first reproduction.

## Dataset

The training data is available from the
[Carvana Image Masking Challenge on Kaggle](https://www.kaggle.com/c/carvana-image-masking-challenge).
Sign in and accept the competition rules if requested. Only these two archives
are needed for this reproduction:

| File | Purpose | Extracted location in the official repository |
| --- | --- | --- |
| `train_hq.zip` | 5,088 high-quality RGB car images used as model inputs | `data/imgs/` |
| `train_masks.zip` | 5,088 reference masks identifying car and background pixels | `data/masks/` |

An image such as `a1038b207299_14.jpg` pairs with
`a1038b207299_14_mask.gif`. Put the extracted files directly in these folders.
The original training code splits these pairs into training and validation sets.
`train_masks.csv.zip`, `test.zip`, and `test_hq.zip` are not needed for this run.
Download the archives on the cloud training instance, or upload them there after
downloading through the Kaggle website.

## Reproduction workflow

1. Clone the official source and select the recorded commit.
2. Verify cloud Python, PyTorch/torchvision, CUDA access, and upstream dependencies.
3. Obtain official Kaggle `train_hq.zip` and `train_masks.zip`; verify archives and pairs.
4. Run the original training entry, recording commands, environment, logs, and weights.
5. Reload the final checkpoint with the original predictor and inspect the output.
6. Export metrics/figures and verify the downloaded backup before cloud shutdown.

The [AutoDL runbook](docs/AUTODL_RUNBOOK.md) documents the operational route.
The Mac is the editor/client; training executes on the Linux cloud GPU.
The community image supplies the runtime only; its code, subset, and old weights
are excluded. Local Docker is unnecessary.

The successful command, in the official checkout with prepared data, was:

```bash
export WANDB_MODE=offline
python -u train.py --epochs 5 --batch-size 1 --learning-rate 1e-5 --scale 0.5 --validation 10
```

## Files and evidence

| Path | Contents |
| --- | --- |
| `reports/` | Completed English report and selected figures |
| `records/carvana-baseline-02-fp32/` | Exported metrics, timestamps, source/dependency records, and checksums |
| `records/carvana-baseline-01/` | Failure notes and diagnostic results |
| `docs/` | AutoDL operations, environment record, upstream review, report template |
| `Pytorch-UNet/` | Separate official checkout, ignored by the outer repository |
| `artifacts/` | Verified local backup including raw logs, W&B files, and weights; ignored |
| `notes/` | Private learning notes; ignored |

The [artifact checklist](records/README.md) describes the evidence collected.
The raw package freeze includes container-specific paths and is not a portable
installation lockfile. Actual train/validation IDs were not saved; a fixed split
seed alone cannot recover membership across directory enumerations. The displayed
sample is not established as held-out data. No independent test score is claimed.

The official implementation retains its GPL-3.0 license in the separate checkout.
This repository contributes reproduction documentation and experiment records.
Public documents are in English. Git staging, commits, and pushes remain
user-operated. Cloud shutdown is not yet confirmed in the local record.

## Try a single image

After training, run the following from the official `Pytorch-UNet` repository
directory, using the prepared Python environment. It loads the epoch-5 weights
and predicts a mask for one image:

```bash
mkdir -p predictions
python predict.py \
  --model checkpoints/checkpoint_epoch5.pth \
  --input data/imgs/a1038b207299_14.jpg \
  --output predictions/a1038b207299_14_pred.png \
  --scale 0.5
```

The weights and dataset are not included in this documentation repository.
Use the checkpoint produced by your training run or restore it from your backup.
To try another RGB image, replace the input and output paths. A reference mask
is not needed for prediction; it is needed to compare against the correct
segmentation.

### Prediction comparison

The figure below shows the original image, its reference mask, and the prediction
from the reloaded epoch-5 model. White marks the car and black marks the background
in both masks.

![Original image, reference mask, and epoch-5 prediction](reports/figures/prediction_comparison.png)

The model captures much of the car silhouette but incorrectly includes some
background on both sides. This is a visual example; the sample's membership in
the training or validation split was not recorded. To inspect another example,
compare its original image, matching reference mask (if available), and saved
prediction side by side.
