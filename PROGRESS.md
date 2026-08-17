# Progress Log

- 2026-08-17 — Phase 0 (Data): DONE. `python scripts/prepare_data.py` exits 0,
  `data/processed/sample.parquet` has 200,000 rows (real TLC Jan 2023 data,
  downloaded fresh since network access was available). Acceptance check passed.
- 2026-08-17 — Phase 1 (Features/Training): DONE. `python -m src.train` exits 0
  in 40.8s (well under 3-min budget). Metrics table printed for all 3 models.
  XGBoost R2 = 0.8207 (threshold >= 0.65). Temporal split assertion passes.
  Acceptance check passed.
- 2026-08-17 — Phase 2 (MLflow): DONE. `mlruns/` has 3 runs under experiment
  `nyc-taxi-duration`. Champion registered with alias `champion`.
  `python scripts/load_champion.py` loads via alias URI (no fallback needed)
  and predicts 15.83 min for the hardcoded sample input. Acceptance check passed.
- 2026-08-17 — Phase 3 (FastAPI service): DONE. `app/main.py` loads the model
  once via lifespan handler, exposes /health, /predict, /predict/batch.
  `pytest tests/test_api.py` passes 4/4 (health, valid predict in 0-120 range,
  malformed -> 422, batch predict) using TestClient, no real port needed.
  Acceptance check passed.
- 2026-08-17 — Phase 4 (Tests): DONE. Full suite (`test_features.py`,
  `test_cleaning.py`, `test_split.py`, `test_api.py`, `test_pipeline.py`) —
  16/16 passed, 0 failures, 0 skipped. `ruff check .` clean. Acceptance check
  passed.
- 2026-08-17 — Phase 5 (Docker): DONE. `docker build -t taxi-duration:latest .`
  succeeds (~47s, layer-cached deps). Model is trained inside the build (see
  DECISIONS.md for why: host/image Python version mismatch broke cloudpickle
  loading, and MLflow's file store bakes absolute host paths into meta.yaml).
  Container run + curl verified: `/health` -> model_loaded true, `/predict` ->
  15.83 min (matches host). `docker compose config` validates. Acceptance check
  passed.
- 2026-08-17 — Phase 6 (GitHub Actions): DONE, partially verified. Workflow YAML
  is valid; every constituent step (lint, prepare_data, train, test, docker
  build) was individually verified locally and passes. Actual execution on
  GitHub Actions is not possible in this environment — logged in
  UNVERIFIED.md with the exact command to verify after a push.
- 2026-08-17 — Phase 7 (Documentation): DONE. README.md has all required
  sections (problem statement, mermaid architecture diagram, results table,
  quickstart, feature engineering, design decisions, not-implemented).
  `notebooks/walkthrough.ipynb` executed end to end with `jupyter nbconvert
  --execute` — 12/12 cells ran with zero errors, plots embedded.
