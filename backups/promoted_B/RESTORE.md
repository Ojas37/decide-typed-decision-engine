# Promoted Candidate B snapshot
Restore: Copy-Item backups\promoted_B\artifacts\* artifacts\ -Force ; Copy-Item backups\promoted_B\model.py, backups\promoted_B\train.py . -Force
Verify with MANIFEST.sha256. The pre-B baseline is in backups\baseline_v1.
