FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY scripts/ scripts/
COPY data/processed/sample.parquet data/processed/sample.parquet

# Train inside the image so the pickled sklearn Pipeline is produced by the
# exact Python/library versions that will load it at runtime (a model trained
# on the host's Python 3.10 venv is not cloudpickle-compatible with 3.11 here).
RUN python -m src.train

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
