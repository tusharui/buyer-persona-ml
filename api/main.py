import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse
from api.dependencies import model_loader
from api.routes.predict import router as predict_router
from src.database import check_health as check_db_health, close as close_db
from src.cache import cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    await cache.connect()
    try:
        model_loader.load()
    except FileNotFoundError as e:
        print(f"[api] WARNING: {e}")
    yield
    await cache.close()
    await close_db()


app = FastAPI(
    title="Buyer Persona ML API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(predict_router)


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
