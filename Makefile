.PHONY: setup test run docker-build docker-run docker-up docker-down docker-logs clean db-check redis-check

setup:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v --tb=short

run:
	streamlit run dashboard/app.py

pipeline:
	python -m src.pipeline

predict:
	python src/predict.py --input data/raw/transactions.csv --output predictions.csv

db-check:
	python -c "import asyncio; from src.database import check_health; print('DB healthy:', asyncio.run(check_health()))"

generate-data:
	python -m src.data_generator --clear

feature-store:
	python -m src.feature_store

run-worker:
	celery -A src.celery_app worker --loglevel=info --pool=solo

run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t buyer-persona-ml .

docker-run:
	docker run -p 8501:8501 buyer-persona-ml

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-up-prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	rm -rf models/*.pkl models/*.npy
	rm -rf data/processed/*.csv
