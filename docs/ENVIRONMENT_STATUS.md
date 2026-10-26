# Observed Community-Image Environment

Recorded on 2026-10-25 (America/Los_Angeles) from the user's pasted cloud-terminal
output. These are transcript observations, not independent live inspection by
the assistant. Subsequent downloaded experiment records establish completion of
the FP32 run; see the reproduction report.

| Item | Observed value |
| --- | --- |
| Platform | AutoDL, running community-image instance |
| GPU | NVIDIA GeForce RTX 3080 Ti, 12,288 MiB |
| Allocated CPU / RAM | 12 cores / 90 GB, according to the platform banner |
| Python | 3.8.10 |
| Python executable | `/root/miniconda3/bin/python` |
| PyTorch | 2.0.0+cu118 |
| torchvision | 0.15.1+cu118; import succeeded |
| pip | 21.1.3 |
| PyTorch CUDA build | 11.8 |
| CUDA available | True |
| NVIDIA driver | 595.71.05 |
| Driver-reported CUDA capability label | 13.2 |
| System disk available | Approximately 29G |
| Data disk available | Approximately 50G at `/root/autodl-tmp` |
| Visible project directory | `/root/UNet-master` |

The installed torch version differs from the community page's 1.8.1 label and
meets the reviewed current README's minimum framework version. No upgrade is
justified solely by that stale label. Keep the existing environment initially.

A subsequent user transcript confirms the official cloud checkout at
`/root/autodl-tmp/Pytorch-UNet`, HEAD
`21d7850f2af30a9695bbeea75f3136aa538cfc4a`. Both torch and torchvision imports
succeeded. Source revision and installed package versions are retained with the
successful run under `records/carvana-baseline-02-fp32/`.
The five upstream requirements were subsequently verified at their pinned
versions; pip check passed. Official data preparation and a five-epoch FP32 run
completed. The final checkpoint was reloaded for prediction. See
[the report](../reports/REPRODUCTION_REPORT.md) and verified local backup records. The bundled `/root/UNet-master` project is excluded.

This record omits authentication files, access tokens, and private instance URLs.
