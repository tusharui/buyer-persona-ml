import datetime
import os
from pathlib import Path
from typing import Optional

import mlflow
from mlflow import MlflowClient

from src.config import REPORTS_DIR

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


class ExperimentLogger:
    def __init__(self, name="pipeline", save_dir=None):
        self.save_dir = Path(save_dir or REPORTS_DIR)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self._mlflow_active = False
        self._run_id: Optional[str] = None
        self._client: Optional[MlflowClient] = None
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(name)

    def log_param(self, key, value):
        if self._mlflow_active and self._run_id:
            mlflow.log_param(key, value)

    def log_params(self, d: dict):
        if self._mlflow_active and self._run_id:
            mlflow.log_params(d)

    def log_metric(self, key, value):
        if self._mlflow_active and self._run_id:
            mlflow.log_metric(key, round(float(value), 6))

    def log_metrics(self, d: dict):
        if self._mlflow_active and self._run_id:
            mlflow.log_metrics({k: round(float(v), 6) for k, v in d.items()})

    def log_artifact(self, path: str):
        if self._mlflow_active and self._run_id:
            mlflow.log_artifact(path)

    def start_run(self):
        try:
            self._run_id = mlflow.start_run(run_name=f"{self.name}_{datetime.date.today().isoformat()}").info.run_id
            self._mlflow_active = True
            self._client = MlflowClient()
            mlflow.log_param("experiment_name", self.name)
        except Exception as e:
            print(f"[tracking] MLflow unavailable, falling back to JSON: {e}")
            self._mlflow_active = False

    def end_run(self):
        if self._mlflow_active and self._run_id:
            mlflow.end_run()
            self._mlflow_active = False

    def save(self, filename=None):
        filename = filename or f"{self.name}_{datetime.date.today().isoformat()}.json"
        path = self.save_dir / filename
        return path
