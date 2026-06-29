"""Experiment tracking — lightweight JSON logger."""

import json
import datetime
from pathlib import Path
from src.config import REPORTS_DIR


class ExperimentLogger:
    """Logs pipeline params and metrics to a JSON file."""

    def __init__(self, name="pipeline", save_dir=None):
        self.save_dir = Path(save_dir or REPORTS_DIR)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.data = {
            "experiment": name,
            "timestamp": datetime.datetime.now().isoformat(),
            "params": {},
            "metrics": {},
            "artifacts": [],
        }

    def log_param(self, key, value):
        self.data["params"][key] = value

    def log_params(self, d: dict):
        self.data["params"].update(d)

    def log_metric(self, key, value):
        self.data["metrics"][key] = value

    def log_metrics(self, d: dict):
        self.data["metrics"].update(d)

    def log_artifact(self, path: str):
        self.data["artifacts"].append(str(path))

    def save(self, filename=None):
        filename = filename or f"{self.name}_{datetime.date.today().isoformat()}.json"
        path = self.save_dir / filename
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
        return path
