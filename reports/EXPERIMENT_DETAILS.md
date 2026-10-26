# U-Net Reproduction: Experiment Details

## Outcome

The unmodified upstream training code completed five FP32 epochs on 5,088
Carvana training pairs. Final internal validation foreground Dice was
**0.968135**. The process exited with code 0, all five saved checkpoints passed
finite-tensor checks, and the final checkpoint was reloaded successfully by the
official prediction script. The successful run lasted **1 hour 7 minutes 39 seconds**,
from `2026-10-26T00:10:32Z` to `2026-10-26T01:18:11Z`, including dataset
initialization and logging shutdown. This is a documented quick-start
reproduction, not a claim to reproduce the author's competition score.

## Source and objective

The objective was to learn the complete workflow: acquire data, prepare a cloud
environment, train an existing segmentation model, diagnose an execution failure,
validate checkpoints, run inference, and preserve experiment evidence.

- Repository: [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).
- Commit: `21d7850f2af30a9695bbeea75f3136aa538cfc4a`.
- The captured tracked-source diff for the successful run is empty.
- The AutoDL community image supplied the runtime only. Its modified code,
  converted data subset, and preloaded checkpoints were not used.
- Training started from random initialization. No pretrained weights were loaded.

The upstream implementation retains its GPL-3.0 license. Local additions are
experiment documentation, artifact checks, metric extraction, and presentation.

## Environment

| Component | Observed value |
| --- | --- |
| Platform | AutoDL Linux container |
| GPU | NVIDIA GeForce RTX 3080 Ti, 12,288 MiB |
| Allocated CPU / RAM | 12 cores / 90 GB, from the platform banner |
| Storage allocation | 30 GB system disk and 50 GB data disk |
| Python | 3.8.10 |
| PyTorch / torchvision | 2.0.0+cu118 / 0.15.1+cu118 |
| PyTorch CUDA build | 11.8 |
| NVIDIA driver | 595.71.05 |
| NumPy / Pillow | 1.23.5 / 9.3.0 |
| Matplotlib / tqdm / W&B | 3.6.2 / 4.64.1 / 0.13.5 |
| Kaggle CLI | 1.6.17 |
| Logging | W&B offline; successful run ID `vh26aivs` |

The driver displayed CUDA 13.2 compatibility; this is distinct from PyTorch's
CUDA 11.8 build. `pip check` reported no broken requirements. The community
page's advertised PyTorch version did not match the installed runtime, so actual
versions were recorded. The Mac served as the editor, remote client, and backup
location; CUDA training ran on the cloud instance. Local Docker was unnecessary.

The [package snapshot](../records/carvana-baseline-02-fp32/requirements.freeze.txt)
includes image-specific build paths and is evidence of the installed environment,
not a portable installation lockfile.

## Data and training method

Official source: [Carvana Image Masking Challenge](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data).
`train_hq.zip` was downloaded through the Kaggle CLI. After the mask download
stalled, `train_masks.zip` was downloaded through the Kaggle website and uploaded
to AutoDL. Both ZIPs passed CRC checks. Their
[locally recorded SHA256 values](../records/carvana-baseline-02-fp32/data_archives.sha256)
identify the acquired files; they are not independently published Kaggle checksums.

Extraction produced 5,088 images and 5,088 matching `*_mask.gif` files. The
official loader found mask values `[0, 1]` for background and car. The initial
sample had input shape `[1, 3, 640, 959]` at scale 0.5.

The original loader and image-level random split were retained: 4,580 training
images and 508 validation images. The split generator seed is 0, but the loader
uses unsorted directory enumeration. Actual split IDs were not captured during
training, so the sorted image/mask pair manifest cannot recover exact membership.
The original entry also does not globally seed model initialization or shuffling.

The model uses three RGB input channels, two output classes, and transposed
convolutions for upsampling. Loss is cross-entropy plus soft Dice loss. The
original RMSprop configuration, gradient clipping, and validation-driven
ReduceLROnPlateau scheduler were retained. Batch size was 1, learning rate 1e-5,
image scale 0.5, and the successful run used FP32 without AMP.

Validation runs approximately five times per epoch. Upstream computes hard
foreground Dice after argmax, excluding background, at the resized resolution.
With validation batch size 1, the reported value is the mean of per-image Dice,
not pixel accuracy or one global intersection-over-union score.

## AMP failure and FP32 retry

The first run followed the AMP quick start. Displayed batch loss became NaN,
validation Dice remained approximately 8.06e-12, and the epoch-1 checkpoint
contained 100 non-finite floating-point tensors. The run was stopped with
SIGTERM (exit code 143), and its weights were archived separately.

A fresh FP32 diagnostic completed ten updates with finite losses, gradient norms,
and model state; loss decreased from 1.157753 to 0.382148. A new five-epoch run
then omitted `--amp`, retaining the other CLI settings and starting from new
random weights. It completed successfully. AMP-related numerical behavior is a
candidate explanation, but the precise root cause was not established. The runs
were not a controlled same-seed comparison.

Successful training command, executed in the official cloud checkout:

```bash
export WANDB_MODE=offline
python -u train.py --epochs 5 --batch-size 1 --learning-rate 1e-5 --scale 0.5 --validation 10
```

`nohup` kept the job independent of the interactive terminal. The launcher saved
stdout/stderr, start/end UTC timestamps, and the exit code. W&B data remained
offline and was included in the downloaded backup.

## Quantitative results

| Epoch | Completed training steps | Last validation Dice in epoch |
| --- | --- | --- |
| 1 | 4,580 | 0.915408 |
| 2 | 9,160 | 0.940063 |
| 3 | 13,740 | 0.981803 |
| 4 | 18,320 | 0.971803 |
| 5 | 22,900 | 0.968135 |

There are 25 logged validation events. The highest logged Dice was **0.990044**
at step **21,068** in epoch 5. That intermediate point was not checkpointed:
upstream saved weights only at epoch boundaries. The final checkpoint's logged
Dice is **0.968135**, not the best intermediate score. The final displayed batch
loss was 0.00808; it is not an epoch-average loss.

![Sampled batch loss and internal validation Dice](figures/training_curves.png)

Loss values were extracted from terminal progress updates, which do not retain
every batch or full numeric precision. Validation values come from the logged
validation events. No smoothing or invented epoch-average loss was added.
The exported [loss samples](../records/carvana-baseline-02-fp32/batch_loss_sampled.csv)
and [validation history](../records/carvana-baseline-02-fp32/validation_dice.csv)
are retained for inspection.

## Reloaded-model prediction

The final checkpoint was loaded by the original `predict.py` on CUDA, with scale
0.5. The script resizes logits back to the original image dimensions before
argmax and saves the predicted mask.

![Original image, ground truth, and reloaded-model prediction](figures/prediction_comparison.png)

Sample `a1038b207299_14` was selected for the earlier input check. Its membership
in the actual training/validation split is unverified, so this is an inference
example rather than held-out evaluation. The prediction captures the central
car silhouette but includes visible background on both sides, particularly
along the horizontal midsection. The example documents an error rather than
establishing uniformly accurate boundaries. No per-image Dice was measured here.

## Artifact verification and limitations

All five successful checkpoints are 124,241,221 bytes each. Each passed the
finite-tensor check; the final model also passed strict state loading and actual
prediction. The downloaded archive passed its supplied SHA256 check. After
local extraction, all five model files matched the cloud-generated
[checkpoint manifest](../records/carvana-baseline-02-fp32/checkpoints.sha256).
Raw logs, weights, W&B run files, and the original archive are retained locally;
large weights, raw competition data, and credentials are excluded from Git.

This is a single successful run with an image-level split, not a split by car
identity. Views of the same car may occur on both sides. Exact split membership,
global random states, and optimizer/scheduler/scaler resume state were not saved.
Consequently, bit-for-bit retraining and full-state continuation are not claimed.
There was no separate test-set evaluation or Kaggle submission. Internal
validation is not directly comparable to the repository's competition score.
Cloud shutdown and total billed cost have not been verified in the local record.

## Skills exercised

The reproduction covered repository/version selection, Linux shell commands,
Python dependency checks, CUDA/runtime distinctions, Kaggle authentication and
file transfer, image/mask pairing, tensors and U-Net outputs, training versus
validation, numerical-failure diagnosis, checkpoint loading, metric extraction,
Matplotlib figures, and checksum-verified artifact backup. The upstream algorithm
was not presented as a new implementation or novel contribution.
