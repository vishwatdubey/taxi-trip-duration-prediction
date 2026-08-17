# Decisions

- Pre-existing `data/raw/train.csv` (Kaggle 2016 NYC taxi competition dataset:
  lat/lon columns, no `PULocationID`/`DOLocationID`, no zone IDs) does not satisfy
  the schema Phase 1 requires (`PU_DO` needs zone IDs). Treated the Phase 0
  "file already exists" check as scoped to the TLC schema path
  (`data/raw/yellow_tripdata_2023-01.parquet`), which did not exist. Network access
  was confirmed available, so the real TLC January 2023 parquet was downloaded per
  Phase 0 step 2, rather than reusing `train.csv` or falling back to synthetic data.
  `train.csv` is left untouched in `data/raw/` and is not used anywhere in this
  pipeline.
- Phase 0 data source: `existing-file` -> `data/raw/yellow_tripdata_2023-01.parquet` (200,000 rows sampled to `data/processed/sample.parquet`).
- Docker image trains the model at build time instead of copying `mlruns/` from
  the host. Two reasons: (1) MLflow's file store bakes the host's absolute path
  into every `meta.yaml` (`artifact_location`, `source`, `storage_location`), so
  a copied `mlruns/` only resolves `models:/` and `runs:/` URIs if the container's
  WORKDIR happens to match the host path byte-for-byte; (2) more fundamentally,
  the local dev venv runs Python 3.10 (no 3.11 interpreter was available on the
  host) while the spec pins the runtime image to `python:3.11-slim` — a pipeline
  cloudpickled under 3.10 fails to unpickle under 3.11 (`TypeError: code()
  argument 13 must be str, not int`, a CPython code-object format change).
  Training inside the image with the image's own interpreter sidesteps both
  problems and keeps the image genuinely self-contained: `COPY
  data/processed/sample.parquet` + `RUN python -m src.train` happen during the
  build, so `mlruns/` and `models/champion.json` are produced fresh by
  Python 3.11 before the app code ever runs. Verified: container's
  `/health` returns `model_loaded: true` and `/predict` returns the same
  15.83-minute prediction as the host.
- `jupyter`, `nbformat`, `nbconvert`, and `ipykernel` were installed into
  `.venv` to author and execute `notebooks/walkthrough.ipynb`, but were not
  added to `requirements.txt`. They are needed only to edit/re-run the
  notebook, not by the API, training, or tests, and keeping them out of
  `requirements.txt` keeps the Docker image lean (the notebook isn't copied
  into the image either). To re-run the notebook yourself:
  `pip install jupyter nbconvert ipykernel && jupyter notebook`.
