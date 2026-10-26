# Experiment Settings

## Source and model

The experiment used [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
at commit `21d7850f2af30a9695bbeea75f3136aa538cfc4a`, with the original model,
data loader, loss, and training code. Software versions are in the
[environment record](../docs/ENVIRONMENT_STATUS.md).

| Setting | Value |
| --- | --- |
| Input / output | RGB image / two classes: background and car |
| Upsampling | Transposed convolutions, `bilinear=False` |
| Initialization | Random |
| Epochs | 5 |
| Batch size | 1 |
| Image scale | 0.5 |
| Validation fraction | 10% |
| Split generator seed | 0 |
| Precision | FP32 |
| Loss | Cross-entropy + soft Dice |
| Optimizer | RMSprop, `foreach=True` |
| Learning rate | 1e-5 |
| Weight decay | 1e-8 |
| Momentum | 0.999 |
| Gradient clipping | Maximum norm 1.0 |
| Scheduler | ReduceLROnPlateau, maximize validation Dice, patience 5 |
| Logging | W&B offline, run ID `vh26aivs` |

## Successful command

From the official repository directory with data prepared:

```bash
export WANDB_MODE=offline
python -u train.py --epochs 5 --batch-size 1 --learning-rate 1e-5 --scale 0.5 --validation 10
```

The process ran from `2026-10-26T00:10:32Z` to `2026-10-26T01:18:11Z` and
exited with code 0. Checkpoints were saved at each epoch boundary.

## Experiment files

| Record | Location |
| --- | --- |
| AMP failure and FP32 diagnostic | [First attempt](../records/carvana-baseline-01/) |
| Training loss values | [Loss CSV](../records/carvana-baseline-02-fp32/batch_loss_sampled.csv) |
| Validation scores | [Dice CSV](../records/carvana-baseline-02-fp32/validation_dice.csv) |
| Installed packages | [Package snapshot](../records/carvana-baseline-02-fp32/requirements.freeze.txt) |
| Data archive checksums | [SHA256](../records/carvana-baseline-02-fp32/data_archives.sha256) |
| Checkpoint checksums | [SHA256](../records/carvana-baseline-02-fp32/checkpoints.sha256) |
| Model integrity checks | [Checkpoint checks](../records/carvana-baseline-02-fp32/checkpoint_checks.json) |
| Downloadable model | [Epoch 3 release](https://github.com/tommie-04/U-net/releases/tag/v1.0.0) |

The epoch-3 download contains model parameters, BatchNorm state, and mask-value
mapping `[0, 1]`. The full logs and five checkpoints are retained in the local
backup. Results and discussion are in the [project report](REPRODUCTION_REPORT.md).
