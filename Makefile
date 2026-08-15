PYTHON ?= python3
.PHONY: install lint test data features train dashboard api docker docs all
install:
	$(PYTHON) -m venv .venv && .venv/bin/pip install -e '.[dev,ml]'
lint:
	.venv/bin/ruff check src tests scripts && .venv/bin/ruff format --check src tests scripts
test:
	.venv/bin/pytest
data:
	$(PYTHON) scripts/run_quality.py
features:
	$(PYTHON) -c "from src.data.loaders import load_csv; from src.features.material_features import featurize_materials; print(featurize_materials(load_csv('data/raw/DS1_material_properties_5500.csv')).shape)"
train:
	$(PYTHON) scripts/train.py
dashboard:
	streamlit run src/dashboard/app.py
api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000
docker:
	docker compose -f docker/docker-compose.yml up --build
docs:
	sphinx-build -b html docs docs/_build/html
all: data features train
	@echo "Pipeline complete. Run 'make dashboard' to launch the decision cockpit."

