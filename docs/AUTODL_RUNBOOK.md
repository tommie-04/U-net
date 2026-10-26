# AutoDL Reproduction Runbook

Status: five FP32 epochs completed and artifacts verified locally. The initial
AMP attempt failed with non-finite tensors. The successful command omits `--amp`;
see [the completed report](../reports/REPRODUCTION_REPORT.md). The procedure below
describes the setup and uses the successful FP32 command. It is not a
recommendation to overwrite the completed attempts. Cloud shutdown was subsequently observed before local inference comparison.

## Target and resources

The user selected AutoDL host 140 in a 2026-10-25 screenshot: one RTX 3080 Ti
12 GB, 12 allocated CPU cores, 90 GB RAM, and 50 GB data disk. Advertised driver
595.71.05 and CUDA label 13.2 are host metadata, not the installed Python/PyTorch
runtime. The screenshot quotes CNY 1.08/hour; a running instance is now evidenced, but incurred charges have not been checked.
The community image supplies the runtime only; the official source/data and training checks remain pending. This route
supersedes the earlier unexecuted Vast.ai proposal.

## Local source checkout

The user may first clone the official repository under the Mac documentation
workspace. Run these commands in the local VS Code terminal:

```bash
cd /Users/leung/PythonML
git clone https://github.com/milesial/Pytorch-UNet.git
git -C Pytorch-UNet checkout 21d7850f2af30a9695bbeea75f3136aa538cfc4a
git -C Pytorch-UNet rev-parse HEAD
```

This creates `/Users/leung/PythonML/Pytorch-UNet`; it does not install packages,
download data, or start training. The outer repository ignores this nested
checkout. Use the separate cloud checkout below for CUDA execution. Data is
downloaded on the cloud during the data stage, avoiding a redundant Mac copy.

## Official source and existing runtime

Use the community image `milesial/Pytorch-UNet/unet-pytorch:v1` for its runtime
only. The user transcript reports Python 3.8.10 and PyTorch 2.0.0+cu118 with
CUDA available. Check torchvision and the official dependency requirements;
do not reinstall torch solely because the image page has a stale version label.
The publisher's bundled source, converted data subset, and existing checkpoints
are excluded from this baseline. Leave `/root/UNet-master` untouched.

The user executes the following in the cloud JupyterLab terminal or a VS Code
terminal connected to this instance through Remote SSH. These are source setup
commands, not training commands:

```bash
cd /root/autodl-tmp
git clone https://github.com/milesial/Pytorch-UNet.git
cd Pytorch-UNet
git checkout 21d7850f2af30a9695bbeea75f3136aa538cfc4a
git rev-parse HEAD
```

If the destination already exists, inspect it instead of overwriting it. Confirm
the printed revision before proceeding. This checkout uses the data disk.
VS Code is an editor: a Mac-local checkout does not automatically use the cloud
GPU. Remote SSH opens the cloud files and executes the cloud interpreter.
No Mac Docker installation or transfer of a Mac virtual environment is required.

In the dependency stage, inspect the installed packages and install the official
`requirements.txt` with the cloud interpreter if necessary. Its five pinned
packages do not install torch or torchvision. Record any installation changes
and verify imports and CUDA access before training. Do not silently substitute
the publisher's requirements or modified training entry point.

One RTX 3080 Ti 12 GB is the selected target. A 50 GB data disk is expected to
accommodate the training archives, extracted data, five checkpoints, and bounded
logs/exports; measure actual free space before and after extraction. The large
competition test archives are not required for this baseline. Training duration
and peak VRAM must be measured rather than promised.

## Kaggle authentication preparation

For the existing Python 3.8 runtime, the planned download CLI is pinned to
`kaggle==1.6.17` instead of blindly upgrading to the latest CLI. The current
[CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)
requires Python 3.11+. Confirm installation and API access on the actual runtime.
Do not execute the upstream script's unconditional upgrade before this check.

The user obtains their own legacy API credential file `kaggle.json` from
Kaggle account settings and privately uploads it to `/root/autodl-tmp/` using
JupyterLab. Move it into a separate `/root/.kaggle-unet/` directory with directory
mode 700 and file mode 600. Set `KAGGLE_CONFIG_DIR=/root/.kaggle-unet` in each
terminal session used for Kaggle commands. This avoids using publisher-supplied
credential files. Never print the file or add it to source control.

First test access with `kaggle competitions files -c
carvana-image-masking-challenge`. Listing files is an authentication/access check,
not proof that archive downloads have succeeded. Record only sanitized output.

## Data and experiment records

At the data stage, download `train_hq.zip` and `train_masks.zip` from the official
Kaggle Carvana competition, using the user's own account and credentials. Do not
reuse publisher-supplied authentication files. The original
`scripts/download_data.sh` uses legacy `kaggle.json` username/key authentication;
confirm that authentication and the installed Kaggle CLI work before running
it. Record any necessary compatibility adjustment. Never print or commit secrets.

The official script downloads and extracts both archives, moves files into the
expected folders, and deletes the archives. To retain archive SHA256 evidence,
use the same official archive downloads with a documented extraction sequence
that records hashes before deletion. Check ZIP integrity before extraction.
Download data only when the data-preparation stage starts; source cloning alone
does not require it. Do not reuse the community image's converted subset.

Put image files directly in `data/imgs/` and matching `*_mask` files directly
in `data/masks/`. Check 5,088 pairs, matching IDs/dimensions, and expected mask
classes. Default splitting yields 4,580 training and 508 validation images.

Before training, save the actual file enumeration and train/validation IDs.
Upstream uses split seed 0 but does not sort enumeration or globally seed
initialization/training. Saving the actual split is a small recording addition
that must be supplied during setup; the training CLI does not implement it.
Do not silently replace the upstream image-level split with a car-level split.

Create `out/<run-id>/` for the source revision/diff, environment and package
freeze, GPU/driver/CUDA/resources, data/split records, exact command/working
directory, raw log, metrics export, and predictions. `out/` is our convention,
not an upstream output directory. Keep secrets and private account details out
of public records.

Resolve W&B operation before redirecting output: either an authenticated online
run or an offline mode verified against the installed version. Preserve local
records and export the measured loss/validation history. Confirm logging works
before a long job; a temporary dashboard is not the only experiment record.

## Training

Use `screen` or `tmux` so that losing the Mac's SSH connection does not end the
job. Verify a forward/backward/validation pass before the formal run if needed.
No custom architecture or complete recovery system is required.

Use the clean official checkout and a fresh checkpoint directory. The following
uses the reviewed official CLI and explicit file logging. Pass source,
environment, data, and W&B checks before executing it:

```bash
mkdir -p out
mkdir out/carvana-baseline-01
python -m pip freeze > out/carvana-baseline-01/requirements.freeze.txt
python -u train.py --epochs 5 --batch-size 1 --learning-rate 1e-5 --scale 0.5 --validation 10 > out/carvana-baseline-01/train.log 2>&1
```

Use a new run ID rather than overwriting an old run. Save the other experiment
records described above as well; this shell example does not create all of them.
Inspect the log from a separate terminal using
`tail -f out/carvana-baseline-01/train.log` and monitor GPU utilization.

Measure the first complete epoch, including validation, to estimate remaining
time. There is no one-hour limit or verified duration promise. Original model
files are saved as `checkpoints/checkpoint_epochN.pth` at epoch end. An interrupted
first epoch may leave no normal checkpoint. `--load` restores weights rather
than optimizer/scheduler/scaler/epoch state; document a restart as a new attempt.
Do not rely on the source's OOM fallback as a validated recovery mechanism.

## Prediction, acceptance, and shutdown

After five epochs, load `checkpoints/checkpoint_epoch5.pth` with `predict.py`.
Use recorded validation images, the matching class/upsampling settings, and
`--scale 0.5`. Explicitly pass `--model`, `--input`, and an `--output` under
`out/<run-id>/predictions/`. Save files on the headless server rather than using
`--viz`. Make original/ground-truth/prediction comparisons.

Completion requires a successful five-epoch log, readable epoch checkpoints,
saved loss/validation-Dice history, predictions from the reloaded final model,
and an English report. There is no invented minimum Dice requirement. Explain
that internal validation differs from the author's advertised competition score.

Download records, model files and SHA256 values, W&B exports, and figures to the
Mac and verify them. Then shut down the instance: closing SSH or a browser alone
does not stop it. Use the report template for selected GitHub material; raw
data, credentials, and large weights remain outside Git. The user performs all
Git staging, commits, remote changes, and pushes.

## Primary references

- [Upstream source](https://github.com/milesial/Pytorch-UNet/tree/21d7850f2af30a9695bbeea75f3136aa538cfc4a)
- [AutoDL environment/storage](https://www.autodl.com/docs/env/)
- [AutoDL environment setup](https://www.autodl.com/docs/base_config/)
- [AutoDL background execution](https://www.autodl.com/docs/daemon/)
- [AutoDL data retention](https://www.autodl.com/docs/instance_data/)
- [PyTorch version combinations](https://pytorch.org/get-started/previous-versions/)
- [Official Kaggle CLI](https://github.com/Kaggle/kaggle-cli)
