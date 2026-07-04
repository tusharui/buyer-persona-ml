import os
from typing import Optional

import mlflow
from mlflow import MlflowClient, MlflowException
from mlflow.entities.model_registry import ModelVersion

from src.cache import cache

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "buyer_persona_kmeans"
REDIS_ACTIVE_KEY = "model:active"


class ModelRegistry:
    def __init__(self):
        self._client: Optional[MlflowClient] = None
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    @property
    def client(self) -> MlflowClient:
        if self._client is None:
            self._client = MlflowClient()
        return self._client

    def register_model(self, run_id: str, model_path: str = "model", description: str = "") -> Optional[ModelVersion]:
        try:
            result = mlflow.register_model(
                model_uri=f"runs:/{run_id}/{model_path}",
                name=MODEL_NAME,
            )
            if description:
                self.client.update_registered_model(
                    name=MODEL_NAME,
                    description=description,
                )
            return result
        except MlflowException as e:
            print(f"[registry] Failed to register model: {e}")
            return None

    def get_latest_versions(self, stages: list[str] | None = None) -> list[ModelVersion]:
        try:
            return self.client.get_latest_versions(MODEL_NAME, stages=stages)
        except MlflowException:
            return []

    def promote_to_staging(self, version: str) -> bool:
        try:
            self.client.transition_model_version_stage(
                name=MODEL_NAME,
                version=int(version),
                stage="Staging",
            )
            return True
        except (MlflowException, ValueError) as e:
            print(f"[registry] Promote to staging failed: {e}")
            return False

    def promote_to_production(self, version: str) -> bool:
        try:
            self.client.transition_model_version_stage(
                name=MODEL_NAME,
                version=int(version),
                stage="Production",
            )
            return True
        except (MlflowException, ValueError) as e:
            print(f"[registry] Promote to production failed: {e}")
            return False

    def archive_model(self, version: str) -> bool:
        try:
            self.client.transition_model_version_stage(
                name=MODEL_NAME,
                version=int(version),
                stage="Archived",
            )
            return True
        except (MlflowException, ValueError) as e:
            print(f"[registry] Archive failed: {e}")
            return False

    def rollback(self, current_version: str) -> Optional[str]:
        versions = self.get_latest_versions(stages=["Staging", "Production"])
        candidates = [v for v in versions if v.version != current_version and v.current_stage in ("Staging", "Production")]
        if not candidates:
            candidates = self.get_latest_versions()
            candidates = [v for v in candidates if v.version != current_version]
        if not candidates:
            return None
        candidates.sort(key=lambda v: int(v.version), reverse=True)
        target = candidates[0]
        self.client.transition_model_version_stage(
            name=MODEL_NAME,
            version=int(target.version),
            stage="Production",
        )
        return target.version

    async def sync_active_version(self, version: str):
        await cache.set(REDIS_ACTIVE_KEY, version)

    async def get_active_version(self) -> Optional[str]:
        return await cache.get(REDIS_ACTIVE_KEY)


model_registry = ModelRegistry()
