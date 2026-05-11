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

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import DATA_DIR, INDEX_DIR, SBERT_MODEL

from .constants import (
    CORE_SKILL_PATTERNS,
    SENIORITY_MAP,
    SENIORITY_PATTERNS,
    canonicalize,
    detect_seniority,
    extract_known_skills_from_text,
)


def extract_skills_from_desc(text):
    """Extract known skills from job description using pattern matching."""
    if not text:
        return []
    text_lower = text.lower()
    found = []
    for skill, pattern in CORE_SKILL_PATTERNS.items():
        if pattern.search(text_lower):
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


def _parse_skill_list(raw: str) -> list[str]:
    """Split a comma/newline/semicolon-delimited skill string into parts."""
    if not raw or not raw.strip():
        return []
    parts = re.split(r"[,\n;|/]", raw)
    return [p.strip() for p in parts if p.strip() and 1 < len(p.strip()) < 40]


def build_job_text(job):
    """Build a clean text representation of a job for embedding."""
    parts = [
        job.get("job_title", ""),
        job.get("job_description", "")[:600],
        job.get("additional_requirements", "")[:300],
        job.get("skills_required", ""),
        job.get("education_requirements", "")[:100],
    ]
    return " ".join(p for p in parts if p.strip())


async def main(output_prefix: str = "job_index"):
    jobs_path = DATA_DIR / "jobs.csv"

    if not jobs_path.exists():
        from services import scrape_jobs

        print("Jobs CSV not found. Running scraper to fetch jobs...")
        await scrape_jobs()

    if not jobs_path.exists():
        return FileNotFoundError(f"Jobs CSV not found at {jobs_path}")

    with open(jobs_path, "r", encoding="utf-8") as f:
        jobs = list(csv.DictReader(f))

    print(f"Jobs loaded: {len(jobs)}")
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer(SBERT_MODEL)

    # Build metadata for each job (pre-extracted fields)
    print("Extracting job metadata...")
    metadata = []
    texts = []

    for job in jobs:
        job_title = job.get("job_title", "")
        job_desc = job.get("job_description", "")
        job_add = job.get("additional_requirements", "")
        job_edu = job.get("education_requirements", "")
        skills_required_raw = job.get("skills_required", "")

        # Extract skills from description + additional requirements
        desc_extracted = extract_skills_from_desc(job_desc + " " + job_add)

        # Parse structured skills_required field
        structured_skills = _parse_skill_list(skills_required_raw)

        # Also extract known skills from the skills_required text itself
        # (catches skills embedded in prose-style requirements)
        skills_from_text = extract_known_skills_from_text(skills_required_raw)

        # Merge all sources → canonicalize → deduplicate
        all_raw_skills = set(desc_extracted) | set(structured_skills) | skills_from_text
        canonical_skills = list({canonicalize(s) for s in all_raw_skills if s.strip()})

        meta = {
            "job_id": job.get("job_id"),
            "job_title": job_title,
            "job_workplace": job.get("job_workplace", ""),
            "job_nature": job.get("job_nature", ""),
            "company": job.get("company_name"),
            "location": job.get("location"),
            "salary_range": job.get("salary_range"),
            "deadline": job.get("deadline"),
            "skills_required": skills_required_raw,
            "education_requirements": job_edu,
            "additional_requirements": job_add[:500],
            "job_description": job_desc[:800],
            # Pre-extracted fields
            "extracted_skills": desc_extracted,
            "required_years": extract_required_years(job_desc + " " + job_add, job_edu),
            "seniority_level": detect_seniority(job_title + " " + job_desc[:200]),
            # Pre-canonicalized skills — merged from all sources
            "canonical_skills": canonical_skills,
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

    # Encode job titles for the title-matching scoring step
    all_job_titles = [m.get("job_title", "") or "unknown" for m in metadata]
    print(f"Encoding {len(all_job_titles)} job titles in batch...")
    title_embeddings = model.encode(
        all_job_titles,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Save to INDEX_DIR
    INDEX_DIR.mkdir(exist_ok=True)
    emb_path = INDEX_DIR / f"{output_prefix}_embeddings.npy"
    title_emb_path = INDEX_DIR / f"{output_prefix}_title_embeddings.npy"
    meta_path = INDEX_DIR / f"{output_prefix}_metadata.json"

    np.save(emb_path, embeddings)
    np.save(title_emb_path, title_embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    print("\n[SUCCESS] Index built successfully!")
    print(f"   Desc Embeddings  -> {emb_path}  ({embeddings.shape})")
    print(f"   Title Embeddings -> {title_emb_path}  ({title_embeddings.shape})")
    print(f"   Metadata         -> {meta_path}  ({len(metadata)} jobs)")
    print(f"\nNow run cv_matcher.py with --index {output_prefix}")


if __name__ == "__main__":
    asyncio.run(main())
