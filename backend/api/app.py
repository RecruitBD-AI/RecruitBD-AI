import asyncio
import json
import logging

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from core.config import INDEX_DIR

from .routes import router

logging.basicConfig(format="%(levelname)s:\t%(message)s")

EMB_PATH = INDEX_DIR / "job_index_embeddings.npy"
META_PATH = INDEX_DIR / "job_index_metadata.json"


async def _load_resources(app: FastAPI):
    """Helper function to load the SBERT model and job index into memory."""
    loop = asyncio.get_running_loop()
    try:
        print("Loading SBERT model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # SentenceTransformer is CPU-bound, run in thread pool so the event loop stays responsive
        app.state.model = await loop.run_in_executor(
            None, lambda: SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
        )

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load resources on startup."""
    # Initialize state variables
    app.state.model = None
    app.state.job_embeddings = None
    app.state.job_metadata = None
    app.state.ready = False

    # Load resources in the background
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
