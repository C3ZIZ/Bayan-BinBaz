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


# X-Forwarded-For is client-supplied: trusting it unconditionally lets anyone
# rotate the header and get a fresh window per request, which defeats the limit
# entirely. Only honour it when a trusted proxy actually sits in front (Coolify,
# nginx), which the operator asserts via TRUST_PROXY_HEADER=1.
_TRUST_PROXY = os.getenv("TRUST_PROXY_HEADER", "0") == "1"


def _client_key(request: Request) -> str:
    if _TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path.startswith(_RATE_LIMITED_PREFIX):
        allowed, retry_after = _limiter.check(_client_key(request))
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
    """Lightweight liveness probe (does NOT load the models).

    Reports which LLM is actually serving requests, so an operator can tell at a
    glance whether the app is on the metered hosted API or a local model.
    """
    from app.llm import describe_backend

    return {"status": "ok", "llm": describe_backend()}


# Serve the static chat UI at "/". Mounted LAST so it never shadows /api or /health.
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
