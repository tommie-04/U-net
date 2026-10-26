# Training Environment

| Component | Version or configuration |
| --- | --- |
| Platform | AutoDL Linux container |
| GPU | NVIDIA GeForce RTX 3080 Ti, 12 GB |
| CPU / RAM | 12 allocated cores / 90 GB |
| System / data disk | 30 GB / 50 GB |
| Community image | `milesial/Pytorch-UNet/unet-pytorch:v1` |
| Python | 3.8.10 |
| PyTorch | 2.0.0+cu118 |
| torchvision | 0.15.1+cu118 |
| PyTorch CUDA build | 11.8 |
| NVIDIA driver | 595.71.05 |
| NumPy | 1.23.5 |
| Pillow | 9.3.0 |
| Matplotlib | 3.6.2 |
| tqdm | 4.64.1 |
| W&B | 0.13.5, offline mode |
| Kaggle CLI | 1.6.17 |

PyTorch and torchvision imported successfully, CUDA was available, and
`pip check` reported no broken requirements. Training used a separate official
checkout in `/root/autodl-tmp/Pytorch-UNet`.

The [package snapshot](../records/carvana-baseline-02-fp32/requirements.freeze.txt)
contains the full installed package list. The
[experiment settings](../reports/EXPERIMENT_DETAILS.md) list model and training parameters.
