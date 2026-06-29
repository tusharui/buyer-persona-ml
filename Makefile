.PHONY: setup test run docker-build docker-run clean

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

docker-build:
	docker build -t buyer-persona-ml .

docker-run:
	docker run -p 8501:8501 buyer-persona-ml

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	rm -rf models/*.pkl models/*.npy
	rm -rf data/processed/*.csv
