.PHONY: setup data train test serve docker all

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -r requirements.txt

data:
	$(PY) scripts/prepare_data.py

train:
	$(PY) -m src.train

test:
	$(PY) -m pytest tests/ -v

serve:
	$(VENV)/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

docker:
	docker build -t taxi-duration:latest .

all: data train test
