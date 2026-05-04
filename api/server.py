# api/server.py
# FastAPI application factory.
# Import and run via: uvicorn api.server:app --port 17432 --reload

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import scan, classify

app = FastAPI(
    title="MediaSort Sound Classifier",
    description="Audio classification pipeline — genre, BPM, key, vibe",
    version="1.0.0",
)

# CORS — allow Tauri WebView and local dev
app.add_middleware(
    CORSMiddleware,
    # allow_origins=[
    #     "tauri://localhost",
    #     "http://tauri.localhost",
    #     "http://localhost:1420",    # Tauri default dev port
    #     "http://localhost:5173",    # Vite dev server (future)
    # ],
    # allow_credentials=True,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scan.router, prefix="/scan", tags=["scan"])
app.include_router(classify.router, prefix="/classify", tags=["classify"])


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}