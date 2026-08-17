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
