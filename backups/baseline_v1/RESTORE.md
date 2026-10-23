# Baseline v1 backup (the model that scored 82.07 on public dev)

Frozen copy of: `model.py`, `train.py`, `smoke_test.py`, `requirements*.txt`, and `artifacts/`
(`weights.npz`, `vocab.json`, `meta.json`, `training_report.json`). SHA-256 of every file is in `MANIFEST.sha256`.

## Restore the baseline completely (from the starter/ directory)

PowerShell:
```powershell
Copy-Item backups\baseline_v1\artifacts\*  artifacts\ -Force
Copy-Item backups\baseline_v1\model.py, backups\baseline_v1\train.py, backups\baseline_v1\smoke_test.py,
          backups\baseline_v1\requirements.txt, backups\baseline_v1\requirements-train.txt  . -Force
```
bash:
```bash
cp backups/baseline_v1/artifacts/* artifacts/
cp backups/baseline_v1/{model.py,train.py,smoke_test.py,requirements.txt,requirements-train.txt} .
```

Verify: compare each file's SHA-256 with `MANIFEST.sha256`
(`Get-FileHash <file> -Algorithm SHA256` or `sha256sum -c` after adjusting paths).

## Notes
* The experimental `model.py` in the project root is backward compatible: it still loads these baseline
  artifacts (`feature_version` 1, temperature route calibration) and reproduces their outputs exactly.
  That is verified in `experiments/reports/` (baseline parity check). So the artifacts alone are enough to
  serve the baseline even with the newer `model.py`; restoring `model.py` is only needed for a byte-exact rollback.
* Candidates live in `experiments/candidates/<name>/artifacts/` and are served without touching `artifacts/`:
  `ARTIFACTS_DIR=experiments/candidates/<name>/artifacts uvicorn app:app ...`
