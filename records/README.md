# Experiment Artifact Checklist

Exported measurements are available in `carvana-baseline-02-fp32/`. The failed
AMP attempt and diagnostics are documented in `carvana-baseline-01/`. Raw logs,
W&B files, and model weights remain in the verified ignored local backup.
Create a separate directory for each future run and record observed values.

| File | Contents |
| --- | --- |
| `source.txt` | Official URL, commit, source changes or clean status |
| `environment.txt` | OS, Python, torch/torchvision, CUDA, GPU, driver, CPU/RAM, disk |
| `requirements.freeze.txt` | Actual installed package versions |
| `data_manifest.csv` | Image/mask identifiers, dimensions, source archive hashes |
| `split.csv` | Actual training/validation sample IDs |
| `command.txt` | Run ID, working directory, exact command, UTC start/end |
| `train.log` | Raw training and validation log |
| `metrics.csv` | Exported step/epoch, loss, validation Dice, timing |
| `checkpoints.sha256` | Names, sizes, hashes of exported model files |
| `issues.md` | Errors, compatibility changes, interruptions, and resolutions |

Store large model files separately and record their location. Put selected
prediction comparisons and curves in `reports/` with a completed report based
on `docs/EXPERIMENT_REPORT_TEMPLATE.md`. Preserve local W&B data until exports
are verified. Never include credentials, private access URLs, or restricted
raw competition data in public records.
