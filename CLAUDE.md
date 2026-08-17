# Project Spec: NYC Taxi Trip Duration — MLOps Pipeline

## Operating rules (read first)

You are building this **unattended**. The user is not available to answer questions.

1. **Never stop to ask a question.** If a decision is ambiguous, pick the option this
   spec names, or the simplest one that satisfies the acceptance criteria, and record
   the choice in `DECISIONS.md`.
2. **Never assume network access.** Test for it once. If unavailable, use the synthetic
   data path (Phase 0) and continue. Do not retry downloads in a loop.
3. **A phase is done only when its acceptance check passes.** Run the check yourself.
   Do not mark a phase complete based on the code "looking correct."
4. **If a phase cannot be verified** (e.g. no Docker daemon), do not silently move on.
   Write the reason in `UNVERIFIED.md` with the exact command the user must run, then
   continue to the next phase.
5. **Keep a running log** in `PROGRESS.md`: one line per phase with status and timestamp.
6. **Use a virtual environment.** Create `.venv` in the project root (`python3 -m venv .venv`),
   activate it, and install all dependencies there. Every `make` target and every script
   must run inside this venv. Never `pip install` into the system Python. Add a
   `requirements.txt` at the project root listing all dependencies with pinned versions.

Target total runtime: the full pipeline (`make all`) must complete in under 3 minutes on
sample data. Optimize for that — this is a teaching artifact, not a leaderboard entry.

---

## Phase 0 — Data

**Preferred source:** NYC TLC Yellow Taxi trip records, January 2023 parquet, placed at
`data/raw/yellow_tripdata_2023-01.parquet`.

Order of attempts:

1. If a file already exists in `data/raw/`, use it. Do not download anything.
2. Else try downloading from the TLC public URL into `data/raw/`.
3. Else run `scripts/make_synthetic.py` (you will write this) to generate
   `data/raw/synthetic_trips.parquet` — 200,000 rows with the same schema and
   plausible statistical structure:
   - `PULocationID`, `DOLocationID`: ints 1–265, non-uniform (a handful of zones
     should dominate, like real Manhattan data)
   - `tpep_pickup_datetime`: spread across one month, with a realistic hourly
     distribution (rush-hour peaks around 08:00 and 18:00, trough at 04:00)
   - `trip_distance`: log-normal, median ~2 miles
   - `passenger_count`: 1–6, heavily skewed to 1
   - `tpep_dropoff_datetime`: derived from a *latent* duration built as a function of
     distance, hour-of-day congestion multiplier, and the PU/DO pair, plus noise.
     Calibrate the noise so a gradient-boosted model lands at **R² between 0.70 and
     0.82** — the synthetic data must be learnable but not trivially so.

Whichever source is used, downsample to **200,000 rows** for the working dataset and
write it to `data/processed/sample.parquet`. Record the source used in `DECISIONS.md`.

**Acceptance:** `python scripts/prepare_data.py` exits 0 and `data/processed/sample.parquet`
exists with ≥100k rows.

---

## Phase 1 — Features and training

### Cleaning
- Compute `duration_min = (dropoff - pickup).total_seconds() / 60`
- Keep rows where `1 <= duration_min <= 60`
- Drop `trip_distance <= 0` and `trip_distance > 100`
- Drop implied speeds > 100 mph

### Features (`src/features.py`)
This module is **the single source of truth** for feature engineering. The API in Phase 3
must import from this exact module. Do not reimplement any of it elsewhere — a duplicated
haversine or hour-extraction is a build failure.

Expose one function:

```python
def build_features(df: pd.DataFrame) -> pd.DataFrame
```

producing:
- `PU_DO`: string concat of `PULocationID` and `DOLocationID` (categorical)
- `PULocationID`, `DOLocationID` as strings (categorical)
- `trip_distance` (numeric)
- `pickup_hour`, `pickup_dow` (int)
- `is_weekend`, `is_rush_hour` (bool → int)

Encode categoricals with `sklearn.feature_extraction.DictVectorizer`.

### Leakage rules (non-negotiable)
- `tpep_dropoff_datetime` and anything derived from it, other than the target, must
  never appear in the feature matrix. Write a test asserting this.
- Split **temporally**, not randomly: sort by pickup time, first 80% train, last 20%
  validation. No `train_test_split(shuffle=True)`.

### Models
Train three, logging each as a separate MLflow run:
1. `LinearRegression` (baseline)
2. `RandomForestRegressor` (n_estimators=50, capped depth — keep it fast)
3. `XGBRegressor` — this is the production model

No hyperparameter search. Use sensible fixed params. Speed matters more than the last
0.01 of R².

### Target
Train on the **raw duration in minutes**, not log-transformed. This avoids the
"R² measured in log space" trap. If you deviate, you must inverse-transform before
computing any metric, and say so loudly in the README.

### Metrics (log all three, for every model)
`rmse`, `mae`, `r2` — all on the validation split, in minutes.

**Acceptance:** `make train` exits 0, prints a metrics table for all three models, and
the XGBoost R² is ≥ 0.65.

---

## Phase 2 — MLflow

- Tracking URI: **file-based** — `mlflow.set_tracking_uri("file:./mlruns")`.
  Do **not** start an `mlflow server` process. No daemons, no ports, no Postgres.
- Experiment name: `nyc-taxi-duration`
- Per run, log: all hyperparameters, all three metrics, and the fitted
  `DictVectorizer` as an artifact.
- Log the model as a **sklearn Pipeline containing the vectorizer**, so serving cannot
  drift from training. This is the point — do not log a bare estimator.
- Register the best run's model to the registry under the name `nyc-taxi-duration`
  and set the alias `champion` on it.

The API loads via `models:/nyc-taxi-duration@champion`. If alias loading proves flaky
with the file store, fall back to loading by run ID read from `models/champion.json`,
and note the fallback in `DECISIONS.md`.

**Acceptance:** after `make train`, `mlruns/` contains ≥3 runs, and
`python scripts/load_champion.py` loads the model and prints a prediction for one
hardcoded input.

---

## Phase 3 — FastAPI service

`app/main.py`:

- Load the model **once at startup** via a lifespan handler. Never per request.
- `GET /health` → `{"status": "ok", "model_loaded": true}`
- `POST /predict` → request body validated by a Pydantic model:

```json
{
  "PULocationID": 142,
  "DOLocationID": 236,
  "trip_distance": 3.4,
  "passenger_count": 1,
  "pickup_datetime": "2023-01-15T08:30:00"
}
```

  Response: `{"predicted_duration_minutes": 14.2, "model_version": "1"}`

- `POST /predict/batch` accepting a list, returning a list.
- The request payload must be converted to a DataFrame and passed through
  `src.features.build_features` — the same function training uses.
- Return HTTP 422 on invalid input (Pydantic default) and 503 if the model failed to load.

**Acceptance:** start the app with uvicorn in the background, `curl` both endpoints,
assert 200 and a numeric prediction in a plausible range (0–120), then kill the process.
Automate this as `tests/test_api.py` using FastAPI's `TestClient` so it needs no real port.

---

## Phase 4 — Tests

`pytest` suite, all of which must pass before you consider the build done:

- `test_features.py` — output columns are exactly as specified; no dropoff-derived
  column present; `build_features` is deterministic
- `test_cleaning.py` — outlier bounds actually filter
- `test_split.py` — max train timestamp < min validation timestamp
- `test_api.py` — health, valid predict, malformed predict → 422, batch predict
- `test_pipeline.py` — the logged pipeline accepts raw-shaped input, i.e. the
  vectorizer travels with the model

**Acceptance:** `make test` exits 0 with zero failures. Do not skip tests to make this pass.

---

## Phase 5 — Docker

`Dockerfile`: `python:3.11-slim`, requirements copied and installed before app code
(layer caching), non-root user, `EXPOSE 8000`,
`CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]`.

Include the trained model artifacts in the image (copy `mlruns/` or a exported
`models/` directory) so the container is self-contained.

Also write `docker-compose.yml` with a single `api` service and a healthcheck.

Then: attempt `docker build -t taxi-duration:latest .`
- If it succeeds, run the container and curl `/health`, then stop it.
- If no Docker daemon is available, **do not fail the build**. Write to
  `UNVERIFIED.md`: the exact build and run commands, and a note that the image is
  untested.

---

## Phase 6 — GitHub Actions

`.github/workflows/ci.yml`, triggered on push and pull_request to `main`:

1. checkout, setup-python 3.11, pip install with cache
2. `ruff check .`
3. `pytest -v`
4. `docker build` (build only — **no push**)

Do **not** write a deploy job, do not reference registry secrets, do not add anything
requiring credentials. The user has not configured any secrets and a red workflow is
worse than a small one.

---

## Phase 7 — Documentation

`README.md`, written for someone being taught this project:

- One-paragraph problem statement
- Architecture diagram as a mermaid block: data → features → train → MLflow registry
  → FastAPI → Docker, with CI alongside
- Results table: all three models, RMSE / MAE / R², with one sentence on why the
  gradient-boosted model wins
- Quickstart: the exact four commands to go from clone to a live prediction
- A "Feature engineering" section explaining *why* `PU_DO` and `pickup_hour` carry
  most of the signal — this is the teaching payload, spend words here
- A "Design decisions" section: why temporal split, why no log transform, why the
  vectorizer is inside the pipeline, why file-based MLflow
- An honest "Not implemented" section: drift monitoring, orchestrated retraining,
  cloud deploy, load testing

Also produce `notebooks/walkthrough.ipynb` — a short narrative notebook that loads the
sample data, shows the duration distribution, shows duration by hour of day, and
compares the three models' predictions against actuals on a scatter plot. This is what
gets screen-shared during teaching.

---

## Makefile

```
make setup     # create .venv, activate it, install deps
make data      # Phase 0
make train     # Phases 1-2
make test      # Phase 4
make serve     # run API locally
make docker    # build image
make all       # data -> train -> test
```

---

## Final deliverable

When every phase is done, write `HANDOFF.md` containing:
- What was built and verified, phase by phase
- Which data source was used (real or synthetic) and the resulting R²
- Everything in `UNVERIFIED.md`, with the commands to verify it
- Any place you deviated from this spec, and why
- The three things you'd fix first given more time