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
