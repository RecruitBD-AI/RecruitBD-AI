import logging
import os
import shutil
import tempfile
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

router = APIRouter()


@router.post("/parse-cv")
async def parse_cv(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accept a PDF or DOCX upload, parse it with Ollama, return structured JSON."""
    filename = file.filename or ""
    extension = filename.rsplit(".", 1)[-1].lower()

    if extension not in {"pdf", "docx"}:
        logging.warning(f"Unsupported file type uploaded: {filename}")
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF or DOCX file.")

    with tempfile.NamedTemporaryFile(suffix=f".{extension}", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        from services import extract_text, parse_cv

        text = extract_text(tmp_path)
        if not text or not text.strip():
            logging.warning(f"Text extraction failed for file: {filename}")
            raise HTTPException(
                status_code=422, detail="Could not extract text from the file. Is it a scanned image PDF?"
            )

        cv_data = parse_cv(text)
        return cv_data
    except Exception as e:
        logging.error(f"Error parsing CV: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"CV parsing failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@router.post("/match-jobs")
async def match_jobs(
    request: Request, cv_data: dict[str, Any], top: int = Query(default=10, ge=1, le=100)
) -> list[dict[str, Any]]:
    """Accept parsed CV JSON, return top-N job matches with scores."""
    model = request.app.state.model
    job_embeddings = request.app.state.job_embeddings
    job_title_embeddings = request.app.state.job_title_embeddings
    job_metadata = request.app.state.job_metadata

    if model is None or job_embeddings is None or job_title_embeddings is None or job_metadata is None:
        logging.error("Model or job index not loaded.")
        raise HTTPException(status_code=503, detail="Job matching service is not available.")

    try:
        from services import match_cv_dict

        results = match_cv_dict(cv_data, job_embeddings, job_title_embeddings, job_metadata, model, top_n=top)
        return results
    except Exception as e:
        logging.error(f"Error matching jobs: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Job matching failed: {str(e)}")


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Health check endpoint to verify the service is running and dependencies are loaded."""
    model = getattr(request.app.state, "model", None)
    job_metadata = getattr(request.app.state, "job_metadata", None)
    ready = getattr(request.app.state, "ready", False)
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "jobs_indexed": len(job_metadata) if job_metadata else 0,
        "ready": ready,
    }
