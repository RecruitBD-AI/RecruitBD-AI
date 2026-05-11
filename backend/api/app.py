import asyncio
import json
import logging

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from core.config import INDEX_DIR, SBERT_MODEL

from .routes import router

logging.basicConfig(format="%(levelname)s:\t%(message)s")

EMB_PATH = INDEX_DIR / "job_index_embeddings.npy"
TITLE_EMB_PATH = INDEX_DIR / "job_index_title_embeddings.npy"
META_PATH = INDEX_DIR / "job_index_metadata.json"


async def _load_resources(app: FastAPI):
    """Helper function to load the SBERT model and job index into memory."""
    try:
        print("Loading resources...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # SentenceTransformer is CPU-bound, run in thread pool so the event loop stays responsive
        model_task = asyncio.to_thread(lambda: SentenceTransformer(SBERT_MODEL, device=device))

        embeddings = asyncio.to_thread(np.load, EMB_PATH, mmap_mode="r")
        title_embeddings = asyncio.to_thread(np.load, TITLE_EMB_PATH, mmap_mode="r")

        metadata_task = asyncio.to_thread(_load_metadata)

        (
            app.state.model,
            app.state.job_embeddings,
            app.state.job_title_embeddings,
            app.state.job_metadata,
        ) = await asyncio.gather(model_task, embeddings, title_embeddings, metadata_task)

        app.state.ready = True
        print("Resources loaded successfully.")
    except Exception:
        logging.exception("Failed to load resources during startup")
        app.state.ready = False


def _load_metadata() -> list[dict]:
    """Helper function to load job metadata from disk."""
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load resources on startup."""
    # Initialize state variables
    app.state.model = None
    app.state.job_embeddings = None
    app.state.job_title_embeddings = None
    app.state.job_metadata = None
    app.state.ready = False

    if not EMB_PATH.exists() or not META_PATH.exists():
        logging.warning("Index files not found, running initial scrape and index build...")
    else:
        await _load_resources(app)

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
