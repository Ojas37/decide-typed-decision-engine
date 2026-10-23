# Tools

Optional scripts. None are needed to run the service — the judge only needs
`app.py`, `model.py`, `artifacts/`, `requirements.txt`.

- `generate_oof.py` — regenerate out-of-fold logits on train
- `compare_candidates.py` — paired four-way comparison (baseline / A / B / AB)
- `check_route_cal.py` — numerical stress test of the exported route calibrator

## Reproduce the four-way comparison

    python tools/generate_oof.py --seed 13 --features 1
    python tools/generate_oof.py --seed 14 --features 1
    python tools/generate_oof.py --seed 15 --features 1
    python tools/generate_oof.py --seed 13 --features 2
    python tools/generate_oof.py --seed 14 --features 2
    python tools/generate_oof.py --seed 15 --features 2

    python tools/compare_candidates.py

Output goes to `tools/reports/` (gitignored).

