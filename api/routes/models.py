import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas import ModelVersion
from api.dependencies import model_loader
from src.cache import cache

router = APIRouter(tags=["models"])

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

MODEL_KEY = "model:active"
HISTORY_KEY = "model:history"


@router.get("/models", response_model=list[ModelVersion])
async def list_models():
    if not MODELS_DIR.exists():
        return []
    versions = []
    for p in MODELS_DIR.iterdir():
        if p.suffix == ".pkl":
            continue
        if p.is_dir():
            versions.append(ModelVersion(
                version=p.name,
                status="available",
            ))
    if not versions:
        active = model_loader.version if model_loader.is_loaded() else "unknown"
        versions.append(ModelVersion(version=active, status="available"))
    active_version = await cache.get(MODEL_KEY)
    for v in versions:
        if v.version == active_version:
            v.status = "production"
    return versions


@router.post("/models/deploy", response_model=ModelVersion)
async def deploy_model(version: str):
    model_path = MODELS_DIR / f"v_{version}"
    if not model_path.exists():
        artifact_paths = [MODELS_DIR / f"{a}" for a in ["scaler.pkl", "pca.pkl", "kmeans.pkl", "selected_features.pkl"]]
        if not all(p.exists() for p in artifact_paths):
            raise HTTPException(status_code=404, detail=f"Model version '{version}' not found")
        model_path.mkdir(exist_ok=True)

    await cache.set(MODEL_KEY, version)
    await cache.set("model:version", version)

    history_raw = await cache.get(HISTORY_KEY)
    history = json.loads(history_raw) if history_raw else []
    if version not in history:
        history.append(version)
        await cache.set(HISTORY_KEY, json.dumps(history[-5:]))

    model_loader._version = version
    return ModelVersion(version=version, status="production")


@router.post("/models/rollback", response_model=ModelVersion)
async def rollback_model():
    history_raw = await cache.get(HISTORY_KEY)
    if not history_raw:
        raise HTTPException(status_code=400, detail="No rollback history available")
    history = json.loads(history_raw)
    if len(history) < 2:
        raise HTTPException(status_code=400, detail="No previous version to rollback to")
    history.pop()
    prev_version = history[-1]
    await cache.set(MODEL_KEY, prev_version)
    await cache.set(HISTORY_KEY, json.dumps(history))
    model_loader._version = prev_version
    return ModelVersion(version=prev_version, status="production")
