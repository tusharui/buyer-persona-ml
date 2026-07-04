from fastapi import APIRouter

from src.drift_detector import drift_detector

router = APIRouter(tags=["health"])


@router.get("/health/drift")
async def health_drift():
    loaded = drift_detector.load_baseline()
    if not loaded:
        return {
            "status": "no_baseline",
            "drift_detected": None,
            "message": "No training baseline available. Run pipeline first.",
        }
    results = drift_detector.results
    return {
        "status": "degraded" if results.get("overall", {}).get("drift_detected") else "healthy",
        "drift_detected": results.get("overall", {}).get("drift_detected", False),
        "features_checked": results.get("overall", {}).get("features_checked", 0),
        "features_with_drift": results.get("overall", {}).get("features_with_drift", 0),
        "drift_ratio": results.get("overall", {}).get("drift_ratio", 0),
        "baseline_features": drift_detector.n_features,
        "details": results.get("features", {}),
    }
