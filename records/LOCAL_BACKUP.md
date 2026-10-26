# Local Backup Verification

The downloaded AutoDL archive passed its supplied SHA256 check. It was extracted
into `artifacts/autodl-backup-2026-10-26/`, which is excluded from Git.
All five successful-run checkpoints matched the server-generated SHA256 manifest.
The run exit code is 0. Logs, command script, installed dependency versions,
metrics CSV files, checkpoint checks, prediction, figures, and W&B run files
are present. Portable extraction skipped only three W&B convenience symlinks;
the underlying run files were retained. The downloaded archive remains in Downloads.

See `artifacts/autodl-backup-2026-10-26/LOCAL_BACKUP_VERIFICATION.json` for the
verification receipt. Raw competition data is not included in the archive.
Cloud shutdown was subsequently observed in the AutoDL console before the
local checkpoint comparison. No cloud restart was performed for that comparison.
