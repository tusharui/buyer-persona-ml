import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.schemas import HealthResponse
from api.dependencies import model_loader
from api.routes.predict import router as predict_router
from api.routes.training import router as training_router
from api.routes.models import router as models_router
from api.routes.drift import router as drift_router
from api.exception_handlers import (
    http_exception_handler,
    validation_exception_handler,
    unhandled_exception_handler,
)
from api.middleware import RequestLoggingMiddleware
from src.database import check_health as check_db_health, close as close_db
from src.cache import cache
from src.drift_detector import drift_detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    await cache.connect()
    try:
        model_loader.load()
        await model_loader.load_active_version()
    except FileNotFoundError as e:
        print(f"[api] WARNING: {e}")
    drift_detector.load_baseline()
    yield
    await cache.close()
    await close_db()


app = FastAPI(
    title="Buyer Persona ML API",
    description="ML-powered customer segmentation and persona prediction API. "
                "Supports async training via Celery, model versioning, "
                "and Redis-cached predictions.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={"name": "Tushar", "email": "tusharanshu18@gmail.com"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(predict_router)
app.include_router(training_router)
app.include_router(models_router)
app.include_router(drift_router)


@app.get("/health", response_model=HealthResponse)
async def health():
    db_ok = await check_db_health()
    model_ok = model_loader.is_loaded()
    return HealthResponse(
        status="healthy" if db_ok and model_ok else "degraded",
        database="connected" if db_ok else "disconnected",
        redis="connected" if cache._client is not None else "unavailable",
        model="loaded" if model_ok else "unavailable",
        model_version=model_loader.version if model_ok else None,
    )


@app.get("/")
async def root():
    return {"service": "Buyer Persona ML API", "docs": "/docs"}
