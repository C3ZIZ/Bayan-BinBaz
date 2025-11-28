from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router as api_router


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
