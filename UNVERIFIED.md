# Unverified

- **GitHub Actions CI (`.github/workflows/ci.yml`)**: cannot be executed in this
  environment (no `act` tool, and running the real workflow requires pushing to
  GitHub). Every individual step in the workflow was run and verified locally,
  which is the best available substitute:
  - `ruff check .` — passes (verified in Phase 4)
  - `python scripts/prepare_data.py` — passes (Phase 0 acceptance check)
  - `python -m src.train` — passes (Phase 1/2 acceptance check)
  - `pytest -v` — 16/16 pass (Phase 4 acceptance check)
  - `docker build -t taxi-duration:latest .` — succeeds (Phase 5 acceptance check)

  To verify for real: push this branch to GitHub (or open a PR against `main`)
  with Actions enabled, and confirm the `CI` workflow run is green.
  ```
  git push origin main
  gh run watch   # after the push triggers the workflow
  ```
