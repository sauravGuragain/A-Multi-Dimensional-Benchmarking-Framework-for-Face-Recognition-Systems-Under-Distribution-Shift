from __future__ import annotations

import logging

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_settings
from api.routers import deployment, figures, fingerprints, registry, reports, results, runs
from api.readonly_store import DatabaseUnavailableError, ReadOnlyResultStore, StoredDataError
from api.schemas import HealthResponse, StatusResponse

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
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],   # read-only API
    allow_headers=["*"],
)


@app.exception_handler(DatabaseUnavailableError)
async def _db_unavailable(_: Request, exc: DatabaseUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(StoredDataError)
async def _stored_data(_: Request, exc: StoredDataError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


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


@app.get("/api/status", response_model=StatusResponse, tags=["health"])
def status() -> StatusResponse:
    """What the API can see right now (database, figures, reports). Never writes."""
    settings = get_settings()
    db = Path(settings.db_path)
    run_count, err = None, None
    if db.is_file():
        try:
            run_count = ReadOnlyResultStore(db).count_runs()
        except DatabaseUnavailableError as exc:
            err = str(exc)
    else:
        err = f"Results database not found at '{db}'."
    return StatusResponse(
        db_path=str(db), db_exists=db.is_file(), db_readable=err is None,
        db_error=err, run_count=run_count,
        figures_dir=settings.figures_dir, figures_dir_exists=Path(settings.figures_dir).is_dir(),
        reports_dir=settings.reports_dir, reports_dir_exists=Path(settings.reports_dir).is_dir(),
    )


@app.get("/", tags=["health"])
def root() -> dict[str, str]:
    return {
        "name": "FaceEval-X API",
        "docs": "/docs",
        "health": "/api/health",
    }
