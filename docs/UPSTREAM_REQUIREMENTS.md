# Upstream Reproduction Requirements

Reviewed on 2026-10-21. The GitHub API reported `master` at
`21d7850f2af30a9695bbeea75f3136aa538cfc4a`, matching the existing, unmodified local
reference checkout. This is a source review, not an AutoDL execution result.

## Environment and data

The [README](https://github.com/milesial/Pytorch-UNet/blob/21d7850f2af30a9695bbeea75f3136aa538cfc4a/README.md)
describes CUDA with PyTorch 1.13 or later and a separate dependency installation.
The five pins in `requirements.txt` are matplotlib 3.6.2, NumPy 1.23.5,
Pillow 9.3.0, tqdm 4.64.1, and wandb 0.13.5. Neither torch nor torchvision is
installed by that file, although both are imported by the source. The README's
Python 3.6+ label is not a complete compatibility specification for this stack.
An actual Python/PyTorch/CUDA combination must be selected and tested before
training. The Dockerfile uses `nvcr.io/nvidia/pytorch:22.11-py3`; Docker is an
alternative installation route, not a requirement for the native workflow.

The download script requests `train_hq.zip` and `train_masks.zip` from Kaggle's
Carvana competition. Its authentication flow expects a legacy
`~/.kaggle/kaggle.json` username/key file. Existing verified copies can be staged
instead, with their origin and hashes recorded. No credentials belong in logs or
the project repository.

Run from the repository root. Put RGB inputs directly in `data/imgs/` and paired
masks directly in `data/masks/`, without nested directories or unrelated files.
For example, `example_01.jpg` pairs with `example_01_mask.gif`. Dimensions must
match and each sample must have exactly one image and one mask.

## Original execution and outputs

The quick-start training entry point is `python train.py --amp`. It starts from
random weights unless `--load` is supplied. The model defaults to three input
channels, two output classes, and transposed-convolution upsampling. Training
performs validation and writes model files; `predict.py --model PATH` loads a
chosen saved model and writes predicted masks. `evaluate.py` exposes a function
used by training, rather than a standalone evaluation CLI.

The epoch-end checkpoint contains the model state and mask-value mapping. It
does not contain the optimizer, scheduler, AMP scaler, random states, or training
position. Loading it starts a new training invocation rather than restoring the
complete previous run. An interrupted first epoch produces no normal epoch-end
checkpoint. Reusing an output directory can overwrite matching epoch filenames.

W&B initialization is part of the original training loop. Its authentication,
network access, and record retention need to be resolved before a timed run.
Export logs and metrics rather than relying on a temporary dashboard alone.

## Reproduction boundaries

The README does not specify a required GPU model, minimum VRAM, training duration,
or a complete recipe connecting its advertised score to the quick-start defaults.
Its reported competition result is not directly interchangeable with this
script's internal validation score. A one-hour budget must therefore be checked
against actual epoch timing, validation overhead, and artifact export time.

For a documented reproduction, preserve the upstream commit, environment,
dataset hashes, actual sample IDs and split, commands, logs, completed epochs,
metrics, checkpoint hashes, and prediction examples. The loader uses unsorted
directory enumeration: a fixed split seed alone does not fix split membership
across machines. The original script also does not globally seed initialization
and training. Record limitations and any compatibility patches explicitly.

Status updated 2026-10-25: the user has started an AutoDL community-image
instance. Python 3.8.10, PyTorch 2.0.0+cu118, and CUDA detection are evidenced by
the user transcript. The image supplies the runtime only; the official cloud checkout and fresh Kaggle
data were subsequently verified. A five-epoch FP32 run completed after an AMP
failure; see [the report](../reports/REPRODUCTION_REPORT.md). The previous Colab route and diagnostic
artifacts have been removed from the active project and are not baseline evidence.
