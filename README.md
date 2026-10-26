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
The numerical root cause remains unconfirmed. The highest validation point
is not the score of the saved final checkpoint. This is a quick-start workflow
reproduction, not a reproduction of the author's competition result.

The report includes the failed AMP attempt, successful FP32 run, results,
prediction errors, and limitations of this first reproduction.

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
