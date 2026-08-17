# Unverified

Nothing. Everything, including the GitHub Actions CI workflow, has been
verified for real.

- **GitHub Actions CI** was originally unverifiable in the sandbox (no `act`,
  no way to run the real workflow without a GitHub push). After pushing to
  `origin/main`, the `CI` workflow run was watched to completion via
  `gh run watch` and finished green in 3m29s: checkout, setup-python 3.11,
  lint, prepare_data, train, test, and docker build all passed.
  Run: https://github.com/vishwatdubey/taxi-trip-duration-prediction/actions/runs/31982853813
