# Carvana U-Net Reproduction Report

Status: template, without measurements. Replace prompts with observed values
after execution; planned settings are not results.

## Objective and source

Execute the original Carvana training/prediction workflow for five epochs.
Record the upstream commit, source hashes, dependency versions, and every patch
with its rationale. Identify upstream and locally added material separately.

## Execution environment

Record AutoDL base-image identity, GPU/VRAM, allocated CPU/RAM, disk/free space, OS, Python, PyTorch,
torchvision, CUDA build, driver, package freeze, and W&B mode. Explain the Mac's
role as the remote client.

## Data and settings

Record Kaggle source, archive names/hashes, preparation, paired sample counts,
actual train/validation IDs, randomness limitations, scale, batch size, AMP,
model/classes/upsampling, loss, optimizer, learning rate, and epochs. Explain
the image-level split rather than claiming separation by car identity.

## Execution record

Include run ID, UTC start/end, working directory, exact commands, logs, first
epoch and total timing, errors, interruptions, and how records were exported.
Report actual costs only when verified. Describe weight reloads accurately.

## Results

Present training loss and validation Dice curves with labeled axes. Identify the
evaluation resolution/aggregation, final epoch, and checkpoint. Show original,
ground-truth, and predicted masks for recorded validation samples, including a
difficult example. Confirm successful prediction after reloading the final model.
Do not substitute results from the author's pretrained weights.

## Artifact index

List checkpoint files, sizes and SHA256 values, environment/data/split records,
raw metrics/logs, and report figures. Use artifact links without exposing
credentials, private account details, or restricted dataset files.

## Limitations and learning

Discuss randomness, split limitations, training budget, and the different
evaluation context of the author's competition score. Explain Dataset/DataLoader,
U-Net, loss/backpropagation, validation, model saving, and prediction in your own
words. Record actual configuration problems and their resolutions.
