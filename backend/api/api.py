"""
api.py — RecruitBD-AI FastAPI Backend
--------------------------------------
Run with:  uvicorn api:app --reload --port 8080

Endpoints:
  POST /parse-cv       — upload PDF/DOCX, returns structured CV JSON
  POST /match-jobs     — accepts CV JSON body, returns ranked job matches
"""

import json
import os
import tempfile
import warnings
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from config import INDEX_DIR
from jobs.build_index import main as build_index

warnings.filterwarnings("ignore")

EMB_PATH = INDEX_DIR / "job_index_embeddings.npy"
META_PATH = INDEX_DIR / "job_index_metadata.json"

_model: SentenceTransformer | None = None
_job_embeddings: np.ndarray | None = None
_job_metadata: list[dict] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _job_embeddings, _job_metadata

    if not INDEX_DIR.exists():
        print("⚙️  Index not found. Building…")
        build_index()

    print("⚙️  Loading SBERT model (all-MiniLM-L6-v2)…")
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"⚙️  Loading job index from {INDEX_DIR}…")
    _job_embeddings = np.load(EMB_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        _job_metadata = json.load(f)

    print(f"✅  Ready — {len(_job_metadata):,} jobs indexed.")
    yield
    # Cleanup (nothing to do)


app = FastAPI(title="RecruitBD-AI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/parse-cv")
async def parse_cv(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accept a PDF or DOCX upload, parse it with Ollama, return structured JSON."""
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext not in ("pdf", "docx"):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=400, detail="File too large (max 10 MB).")

    # Write to temp file so the parsers can open it by path
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        from cv.cv_parse import extract_text, parse_cv

        text = extract_text(tmp_path)
        if not text or not text.strip():
            raise HTTPException(
                status_code=422, detail="Could not extract text from the file. Is it a scanned image PDF?"
            )

        cv_data = parse_cv(text)
        return cv_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CV parsing failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/match-jobs")
async def match_jobs(
    cv: dict[str, Any],
    top: int = Query(default=10, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Accept parsed CV JSON, return top-N job matches with scores."""
    if _model is None or _job_embeddings is None or _job_metadata is None:
        raise HTTPException(status_code=503, detail="Model not ready yet. Try again shortly.")

    try:
        from matcher.cv_matcher import match_cv_dict

        results = match_cv_dict(cv, _job_embeddings, _job_metadata, _model, top_n=top)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching failed: {str(e)}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "jobs_loaded": len(_job_metadata) if _job_metadata else 0,
        "model_loaded": _model is not None,
    }


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
