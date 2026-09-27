from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_settings
from api.routers import deployment, figures, fingerprints, registry, reports, results, runs
from api.schemas import HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FaceEval-X API",
    description=(
        "REST API for the FaceEval-X multi-dimensional face recognition "
        "benchmarking framework. Serves experiment results, behavioral "
        "fingerprints, deployment scores, and generated figures to the "
        "React dashboard."
    ),
    version="0.1.0",
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router)
app.include_router(results.router)
app.include_router(fingerprints.router)
app.include_router(deployment.router)
app.include_router(figures.router)
app.include_router(reports.router)
app.include_router(registry.router)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
def health_check() -> HealthResponse:
    """Liveness check for the API and a quick framework version readout."""
    from faceeval.core import __version__
    return HealthResponse(status="ok", framework_version=__version__)


@app.get("/", tags=["health"])
def root() -> dict[str, str]:
    return {
        "name": "FaceEval-X API",
        "docs": "/docs",
        "health": "/api/health",
    }
