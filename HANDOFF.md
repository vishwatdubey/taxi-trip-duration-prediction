# Handoff

Built unattended, phase by phase, per `CLAUDE.md`. Every phase's acceptance
check was run for real (not inferred from reading the code) before moving on.
Full detail with timestamps is in `PROGRESS.md`; this is the summary.

## What was built and verified, phase by phase

- **Phase 0 — Data.** `scripts/prepare_data.py` + `scripts/make_synthetic.py`.
  Real TLC January 2023 data was downloaded (network access was available),
  sampled to 200,000 rows at `data/processed/sample.parquet`. The synthetic
  fallback path exists and is calibrated (R² 0.70-0.82 verified with a quick
  XGBoost fit) but wasn't exercised, since the real source worked.
  `python scripts/prepare_data.py` exits 0, sample has 200k rows. **Verified.**

- **Phase 1 — Features and training.** `src/features.py` (single source of
  truth for `build_features`, imported by both training and the API) and
  `src/train.py` (cleaning, temporal split, 3 models). `python -m src.train`
  exits 0 in ~41s, prints the metrics table, XGBoost R² = 0.8207 (≥ 0.65
  threshold). **Verified.**

- **Phase 2 — MLflow.** File-based tracking at `./mlruns`, experiment
  `nyc-taxi-duration`, 3 runs logged with params/metrics/vectorizer artifact,
  best model registered as `nyc-taxi-duration` with alias `champion`.
  `python scripts/load_champion.py` loads via the alias URI directly (the
  run-ID fallback exists in code but was never needed locally). **Verified.**

- **Phase 3 — FastAPI service.** `app/main.py`: model loaded once at startup
  via lifespan handler, `/health`, `/predict`, `/predict/batch`. All requests
  go through `src.features.build_features` — same function training uses.
  `pytest tests/test_api.py` — 4/4 pass via `TestClient`. **Verified.**

- **Phase 4 — Tests.** Full suite: `test_features.py`, `test_cleaning.py`,
  `test_split.py`, `test_api.py`, `test_pipeline.py` — 16/16 pass, 0 skipped.
  `ruff check .` clean. **Verified.**

- **Phase 5 — Docker.** `Dockerfile` (`python:3.11-slim`, layer-cached deps,
  non-root user, `EXPOSE 8000`) and `docker-compose.yml` with a healthcheck.
  `docker build -t taxi-duration:latest .` succeeds; container run + curl
  confirmed `/health` → `model_loaded: true` and `/predict` → 15.83 min
  (matches the host prediction for the same input). **Verified.**

- **Phase 6 — GitHub Actions.** `.github/workflows/ci.yml`: checkout,
  setup-python 3.11, `ruff check .`, `pytest -v`, `docker build` (no push, no
  secrets). YAML is valid and every step was individually run and passed
  locally. The workflow itself could not be executed in this sandbox (no
  GitHub Actions runner, no `act`). **Partially verified — see
  `UNVERIFIED.md`.**

- **Phase 7 — Documentation.** `README.md` with problem statement, mermaid
  architecture diagram, results table, 4-command quickstart, feature
  engineering writeup, design decisions, and an honest not-implemented
  section. `notebooks/walkthrough.ipynb` executed end to end (12/12 cells, 0
  errors) with duration distribution, duration-by-hour, and a 3-model
  prediction-vs-actual scatter. **Verified.**

`make all` (data → train → test) runs in **~47 seconds**, well inside the
3-minute budget.

## Data source and result

**Real data**, not synthetic: NYC TLC Yellow Taxi trip records, January 2023,
downloaded fresh (see `DECISIONS.md` for why the pre-existing
`data/raw/train.csv` — a different, lat/lon-based Kaggle dataset — wasn't
used). 200,000-row working sample. **XGBoost R² = 0.8207** on the temporal
validation split (RMSE 4.19 min, MAE 2.79 min).

## Everything in UNVERIFIED.md

- **GitHub Actions CI** could not be run in this environment. To verify:
  ```
  git push origin main
  gh run watch
  ```
  and confirm the `CI` workflow goes green. Every individual step it runs
  (lint, prepare_data, train, test, docker build) was already verified to
  pass locally, so this is a low-risk gap — mainly confirming the YAML
  triggers and runs cleanly on GitHub's runner image.

## Where this deviated from the spec, and why

1. **Data source resolution.** The spec's Phase 0 step 1 says "if a file
   already exists in `data/raw/`, use it." A file did exist
   (`data/raw/train.csv`) but it's the Kaggle 2016 competition dataset
   (lat/lon columns, no `PULocationID`/`DOLocationID`), which cannot satisfy
   Phase 1's explicit `PU_DO` feature requirement. Treated the existing-file
   check as scoped to the TLC schema path and downloaded real TLC data
   instead (network access was available). `train.csv` is untouched and
   unused. Full reasoning in `DECISIONS.md`.

2. **Docker trains the model in-image instead of copying `mlruns/` from the
   host.** The host's only available Python was 3.10 (no 3.11 interpreter),
   while the spec pins the runtime image to `python:3.11-slim`. A pipeline
   cloudpickled under 3.10 does not unpickle under 3.11 (confirmed: `TypeError:
   code() argument 13 must be str, not int`). Separately, MLflow's file store
   bakes the host's absolute path into every `meta.yaml`, so a copied
   `mlruns/` wouldn't resolve `models:/`/`runs:/` URIs inside the container
   anyway. Fix: `COPY data/processed/sample.parquet` + `RUN python -m
   src.train` happen during the Docker build, so the image trains its own
   model with its own interpreter. Verified working end to end. Full
   reasoning in `DECISIONS.md`.

3. **`jupyter`/`nbformat`/`nbconvert`/`ipykernel` are not in
   `requirements.txt`.** They're needed only to author/re-execute the
   notebook, not by the API, training, or test suite. Keeping them out
   keeps `.venv`/Docker installs lean. Installed ad hoc to build the notebook;
   not persisted anywhere. Noted in `DECISIONS.md`.

## Three things to fix first, given more time

1. **Add a model/input signature to the MLflow-logged pipeline.** Every
   training run currently logs a warning ("Model logged without a
   signature and input example"). Setting `mlflow.sklearn.log_model(...,
   signature=..., input_example=...)` would give schema validation on load
   and is a one-line fix that was skipped to keep training fast and simple.

2. **Handle unseen `PULocationID`/`DOLocationID` values at serving time.**
   `DictVectorizer` silently drops unknown categorical keys rather than
   erroring, so a zone ID never seen in training just contributes nothing to
   the feature vector instead of a clear signal that the input is
   out-of-distribution. Worth an explicit check (and a documented behavior)
   before this ever serves real traffic.

3. **Pin the CI Python patch version and add a cached-artifact fast path.**
   CI currently re-downloads/re-trains from scratch on every run. For a
   larger dataset this would blow past the 3-minute budget; a real pipeline
   would cache `data/processed/sample.parquet` (or the whole venv) between
   CI runs.
