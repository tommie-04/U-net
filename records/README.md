# Experiment Records

`baseline` is the name used for the initial reproduction experiment, before
adding model changes. These two folders record separate attempts, not two
model architectures or two datasets. The first failed; the second completed.

## Initial AMP attempt: `carvana-baseline-01/`

| File | What it records |
| --- | --- |
| `issues.md` | NaN failure, investigation, and subsequent FP32 retry |
| `checkpoint_check.json` | 100 non-finite tensors found in the failed epoch-1 checkpoint |
| `fp32_check.json` | Losses and pass result of the fresh 10-update FP32 diagnostic |
| `exit_code.txt` | Exit code 143 after the failed run was stopped with SIGTERM |

## Successful FP32 run: `carvana-baseline-02-fp32/`

| File | What it records |
| --- | --- |
| `adjustment.txt` | Why AMP was disabled and a new random model was used |
| `batch_loss_sampled.csv` | Displayed batch losses extracted from the training log |
| `validation_dice.csv` | All 25 logged validation Dice events |
| `metrics_summary.json` | Final and highest logged validation Dice |
| `reported_results.json` | Run summary and later artifact verification status |
| `checkpoint_checks.json` | Sizes and finite-tensor checks for the five successful checkpoints |
| `checkpoints.sha256` | Hashes that identify the five model files and verify copies |
| `data_archives.sha256` | Hashes of the downloaded image and mask ZIPs |
| `start_utc.txt`, `end_utc.txt` | Process start/end timestamps |
| `exit_code.txt` | Exit code 0, indicating successful completion |
| `requirements.freeze.txt` | Package snapshot from the cloud runtime; not a portable install file |
| `source_commit.txt` | Upstream Git commit used for training |
| `source_changes.patch` | Empty file: no captured tracked-source changes |

## Later inference comparison: `checkpoint-comparison/`

This folder records epoch-3 versus epoch-5 predictions on nine fixed images.
It contains the per-image scores, inference setup/selection notes, and a
one-image consistency check between local CPU and previous cloud GPU outputs.
These are descriptive examples with unknown original split membership.

## Model files and raw backup

The five successful model files are stored locally at:

```text
artifacts/autodl-backup-2026-10-26/checkpoints/checkpoint_epoch1.pth
artifacts/autodl-backup-2026-10-26/checkpoints/checkpoint_epoch2.pth
artifacts/autodl-backup-2026-10-26/checkpoints/checkpoint_epoch3.pth
artifacts/autodl-backup-2026-10-26/checkpoints/checkpoint_epoch4.pth
artifacts/autodl-backup-2026-10-26/checkpoints/checkpoint_epoch5.pth
```

Each is 124,241,221 bytes (about 118.5 MiB). Large weights, raw logs, and W&B
files are in the ignored backup, not the Git repository. The failed checkpoint
is archived separately inside that backup. The original downloaded backup is
also retained in Downloads. No checkpoint is deleted by the documentation cleanup.

For inference, epoch 3 is selected by the highest logged epoch-end validation
Dice among the saved models. The files contain model state and mask-value mapping,
not a complete optimizer/scheduler resume state. If model downloads are published,
use a separate release asset or model host and provide its SHA256 alongside it.
The selected epoch-3 weights and their SHA256 are published as attachments to
[release v1.0.0](https://github.com/tommie-04/U-net/releases/tag/v1.0.0).
Other successful checkpoints remain in the local backup.
