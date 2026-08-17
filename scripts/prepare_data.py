"""Phase 0: select a data source and write the 200k-row working sample.

Order of attempts:
1. Use an existing file in data/raw/ matching the TLC schema, if present.
2. Download the TLC yellow trip parquet.
3. Fall back to scripts/make_synthetic.py.

Writes data/processed/sample.parquet (<=200,000 rows) and appends the chosen
source to DECISIONS.md.
"""
import sys
import urllib.request
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TLC_FILENAME = "yellow_tripdata_2023-01.parquet"
TLC_PATH = RAW_DIR / TLC_FILENAME
TLC_URL = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{TLC_FILENAME}"
SYNTHETIC_PATH = RAW_DIR / "synthetic_trips.parquet"
SAMPLE_SIZE = 200_000
REQUIRED_COLS = {
    "PULocationID",
    "DOLocationID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "passenger_count",
}


def _has_required_schema(path: Path) -> bool:
    try:
        import pyarrow.parquet as pq

        cols = set(pq.ParquetFile(path).schema.names)
    except Exception:
        return False
    return REQUIRED_COLS.issubset(cols)


def _try_download(url: str, dest: Path, timeout: int = 15) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        return dest.exists() and dest.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001 - any network failure means "unavailable"
        print(f"Download failed ({exc}); falling back to synthetic data.", file=sys.stderr)
        return False


def select_source() -> tuple[Path, str]:
    if TLC_PATH.exists() and _has_required_schema(TLC_PATH):
        return TLC_PATH, "existing-file"

    if _try_download(TLC_URL, TLC_PATH):
        return TLC_PATH, "download"

    sys.path.insert(0, str(Path(__file__).parent))
    from make_synthetic import generate

    SYNTHETIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    generate().to_parquet(SYNTHETIC_PATH, index=False)
    return SYNTHETIC_PATH, "synthetic"


def _record_decision(source_path: Path, source_kind: str, n_rows: int) -> None:
    decisions_path = Path("DECISIONS.md")
    line = (
        f"- Phase 0 data source: `{source_kind}` -> `{source_path}` "
        f"({n_rows:,} rows sampled to `data/processed/sample.parquet`).\n"
    )
    if decisions_path.exists():
        existing = decisions_path.read_text()
        if line.strip() in existing:
            return
        decisions_path.write_text(existing + line)
    else:
        decisions_path.write_text("# Decisions\n\n" + line)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    source_path, source_kind = select_source()

    df = pd.read_parquet(source_path)
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    out_path = PROCESSED_DIR / "sample.parquet"
    df.to_parquet(out_path, index=False)
    _record_decision(source_path, source_kind, len(df))

    print(f"Source: {source_kind} ({source_path})")
    print(f"Wrote {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    main()
