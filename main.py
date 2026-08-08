import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import router as api_router
from app.ratelimit import RateLimiter
from app.retrieval import start_idle_monitor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Free the BGE-M3 model after the service is idle so other projects can
    # reuse the RAM. Replaces the deprecated @app.on_event("startup") hook.
    start_idle_monitor(
        timeout=int(os.getenv("MODEL_IDLE_TIMEOUT", "600")),
        interval=int(os.getenv("MODEL_IDLE_CHECK_INTERVAL", "60")),
    )
    yield


app = FastAPI(
    title="Bayan: BinBaz Fatwa Assistant",
    description="مساعد فتاوى مبني على فتاوى الشيخ ابن باز رحمه الله",
    version="2.0.0",
    lifespan=lifespan,
)

# allow_origins=["*"] together with allow_credentials=True is rejected by
# browsers and is a misconfiguration regardless. This API is unauthenticated,
# so credentials are off and a wildcard origin is genuinely fine. Set
# CORS_ORIGINS to a comma-separated list to lock it down.
_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Each question costs at least two hosted-LLM calls (gate + generation), so an
# unthrottled endpoint burns the HF token's quota fast.
_limiter = RateLimiter(
    max_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "20")),
    window_seconds=float(os.getenv("RATE_LIMIT_WINDOW", "60")),
)
_RATE_LIMITED_PREFIX = "/api/"


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path.startswith(_RATE_LIMITED_PREFIX):
        client = request.headers.get("x-forwarded-for", "")
        client = client.split(",")[0].strip() or (
            request.client.host if request.client else "unknown"
        )
        allowed, retry_after = _limiter.check(client)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "عدد الطلبات كبير. أعد المحاولة بعد قليل."},
                headers={"Retry-After": str(int(retry_after) + 1)},
            )
    return await call_next(request)


app.include_router(api_router, prefix="/api", tags=["chat"])


@app.get("/health", tags=["health"])
def health() -> dict:
    """Lightweight liveness probe (does NOT load the models)."""
    return {"status": "ok"}


# Serve the static chat UI at "/". Mounted LAST so it never shadows /api or /health.
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
