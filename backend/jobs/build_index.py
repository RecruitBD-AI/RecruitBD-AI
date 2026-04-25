"""
build_index.py — Pre-encode all job descriptions and cache to disk
------------------------------------------------------------------
Run this ONCE. Saves job embeddings + metadata to disk.
cv_matcher.py will load these instead of re-encoding every run.

Usage:
    python build_index.py --jobs jobs.csv
    python build_index.py --jobs jobs.csv --output job_index
"""

import asyncio
import csv
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from config import DATA_DIR, INDEX_DIR
from jobs.jobs import main as scrape_jobs

CORE_SKILLS = {
    "python",
    "java",
    "javascript",
    "typescript",
    "c++",
    "c#",
    "sql",
    "dart",
    "flutter",
    "react",
    "node.js",
    "spring boot",
    "django",
    "fastapi",
    "flask",
    "machine learning",
    "deep learning",
    "nlp",
    "computer vision",
    "ai",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "mongodb",
    "postgresql",
    "git",
    "rest api",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "go",
    "kotlin",
    "swift",
    "php",
    "ruby",
    "scala",
    "redis",
    "firebase",
    "android",
    "ios",
    "jetpack compose",
    "selenium",
    "jenkins",
    "linux",
    "bash",
    "r",
    "matlab",
    "vue",
    "angular",
    "express",
    "next.js",
}

SENIORITY_MAP = {
    "intern": 0,
    "trainee": 0,
    "fresher": 0,
    "junior": 1,
    "jr": 1,
    "associate": 2,
    "executive": 2,
    "officer": 2,
    "senior": 3,
    "sr": 3,
    "lead": 4,
    "manager": 4,
    "head": 5,
    "director": 5,
    "gm": 5,
}


def detect_seniority(text):
    text = text.lower()
    best = -1
    for kw, lvl in SENIORITY_MAP.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", text):
            best = max(best, lvl)
    return best if best >= 0 else 2


def extract_skills_from_desc(text):
    if not text:
        return []
    text_lower = text.lower()
    found = []
    for skill in CORE_SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
            found.append(skill)
    return found


def extract_required_years(job_desc, job_edu):
    text = (job_desc + " " + job_edu).lower()
    if any(w in text for w in ["fresher", "no experience", "not required"]):
        return 0.0
    patterns = [
        r"(\d+)\s*\+\s*years?",
        r"(\d+)\s*to\s*\d+\s*years?",
        r"(?:at least|minimum|min\.?)\s*(\d+)\s*years?",
        r"(\d+)\s*years?\s*(?:of\s*)?(?:experience|exp)",
        r"experience\s*(?:of\s*)?(\d+)\s*years?",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    return 0.0


def build_job_text(job):
    """Build a clean text representation of a job for embedding."""
    parts = [
        job.get("job_title", ""),
        job.get("job_description", "")[:600],
        job.get("skills_required", ""),
        job.get("education_requirements", "")[:100],
    ]
    return " ".join(p for p in parts if p.strip())


def main(output_prefix: str = "job_index"):
    csv_files = list(DATA_DIR.glob("jobs_*.csv")) if DATA_DIR.exists() else []

    if not csv_files:
        print("No data found. Scraping job listings...")
        asyncio.run(scrape_jobs())
        csv_files = list(DATA_DIR.glob("jobs_*.csv"))

    if not csv_files:
        raise FileNotFoundError("Scraping completed but no CSV found in data/")

    jobs_path = max(csv_files, key=lambda f: f.stat().st_mtime)

    with open(jobs_path, "r", encoding="utf-8") as f:
        jobs = list(csv.DictReader(f))

    print(f"Jobs loaded: {len(jobs)}")
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build metadata for each job (pre-extracted fields)
    print("Extracting job metadata...")
    metadata = []
    texts = []

    for job in jobs:
        job_title = job.get("job_title", "")
        job_desc = job.get("job_description", "")
        job_add = job.get("additional_requirements", "")
        job_edu = job.get("education_requirements", "")

        meta = {
            "job_id": job.get("job_id"),
            "job_title": job_title,
            "company": job.get("company_name"),
            "location": job.get("location"),
            "salary_range": job.get("salary_range"),
            "deadline": job.get("deadline"),
            "skills_required": job.get("skills_required", ""),
            "education_requirements": job_edu,
            "job_description": job_desc[:800],
            # Pre-extracted fields
            "extracted_skills": extract_skills_from_desc(job_desc),
            "required_years": extract_required_years(job_desc + " " + job_add, job_edu),
            "seniority_level": detect_seniority(job_title + " " + job_desc[:200]),
        }
        metadata.append(meta)
        texts.append(build_job_text(job))

    # Encode all job texts in one batch (much faster than one by one)
    print(f"Encoding {len(texts)} job descriptions in batch...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # pre-normalize for faster cosine sim
    )

    # Save to INDEX_DIR
    INDEX_DIR.mkdir(exist_ok=True)
    emb_path = INDEX_DIR / f"{output_prefix}_embeddings.npy"
    meta_path = INDEX_DIR / f"{output_prefix}_metadata.json"

    np.save(emb_path, embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    print("\n✅ Index built successfully!")
    print(f"   Embeddings → {emb_path}  ({embeddings.shape})")
    print(f"   Metadata   → {meta_path}  ({len(metadata)} jobs)")
    print(f"\nNow run cv_matcher.py with --index {output_prefix}")


if __name__ == "__main__":
    main()
