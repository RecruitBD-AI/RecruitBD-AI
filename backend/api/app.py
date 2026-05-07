import asyncio
import json
import logging
from typing import Callable

import numpy as np
import torch
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from core.config import INDEX_DIR, SBERT_MODEL
from services import build_index, scrape_jobs

from .routes import router

logging.basicConfig(format="%(levelname)s:\t%(message)s")

EMB_PATH = INDEX_DIR / "job_index_embeddings.npy"
META_PATH = INDEX_DIR / "job_index_metadata.json"


async def _load_resources(app: FastAPI):
    """Helper function to load the SBERT model and job index into memory."""
    try:
        print("Loading SBERT model...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        loop = asyncio.get_running_loop()

        # SentenceTransformer is CPU-bound, run in thread pool so the event loop stays responsive
        app.state.model = await loop.run_in_executor(None, lambda: SentenceTransformer(SBERT_MODEL, device=device))

        print("Loading job index embeddings...")
        app.state.job_embeddings = await loop.run_in_executor(None, lambda: np.load(EMB_PATH, mmap_mode="r"))

        print("Loading job index metadata...")
        with open(META_PATH, "r", encoding="utf-8") as f:
            app.state.job_metadata = json.load(f)

        app.state.ready = True
        print("Resources loaded successfully, API is ready to serve requests.")
    except Exception:
        logging.exception("Failed to load resources during startup")
        app.state.ready = False


async def _nightly_job(app: FastAPI):
    try:
        print("Starting nightly job: scraping jobs and rebuilding index...")
        await scrape_jobs()
        await build_index()
        print("Nightly job completed successfully.")
        await _load_resources(app)  # Reload resources after rebuilding index
    except Exception:
        logging.exception("Nightly job failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load resources on startup."""
    # Initialize state variables
    app.state.model = None
    app.state.job_embeddings = None
    app.state.job_metadata = None
    app.state.ready = False

    scheduler = AsyncIOScheduler()
    scheduler.add_job(_nightly_job, "cron", hour=3, minute=0, args=[app])
    scheduler.start()

    if not EMB_PATH.exists() or not META_PATH.exists():
        logging.warning("Index files not found, running initial scrape and index build...")
        asyncio.create_task(_nightly_job(app))
    else:
        asyncio.create_task(_load_resources(app))

    yield
    scheduler.shutdown()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def check_ready(request: Request, call_next: Callable):
    if not request.url.path == "/health" and not request.app.state.ready:
        raise HTTPException(status_code=503, detail="Service is initializing, please try again shortly.")
    return await call_next(request)


app.include_router(router)
