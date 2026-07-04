FROM python:3.12-slim

WORKDIR /mlflow

COPY mlflow/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV MLFLOW_TRACKING_URI=http://0.0.0.0:5000
ENV MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
ENV MLFLOW_ARTIFACT_ROOT=/mlflow/artifacts

RUN mkdir -p /mlflow/artifacts

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlflow/artifacts"]
