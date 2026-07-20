import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import router as api_router
from app.retrieval import start_idle_monitor


app = FastAPI(
    title="Bayan: BinBaz Fatwa Assistant",
    description="مساعد فتاوى مبني على فتاوى الشيخ ابن باز رحمه الله",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router, prefix="/api", tags=["chat"])


@app.on_event("startup")
def _startup() -> None:
    # Free the BGE-M3 model after the service is idle so other projects can reuse the RAM.
    start_idle_monitor(
        timeout=int(os.getenv("MODEL_IDLE_TIMEOUT", "600")),
        interval=int(os.getenv("MODEL_IDLE_CHECK_INTERVAL", "60")),
    )


@app.get("/health", tags=["health"])
def health() -> dict:
    """Lightweight liveness probe (does NOT load the models)."""
    return {"status": "ok"}


# Serve the static chat UI at "/". Mounted LAST so it never shadows /api or /health.
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
