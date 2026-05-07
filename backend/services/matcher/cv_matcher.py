"""
cv_matcher.py — CV to Job Matching Engine (v6)
-----------------------------------------------
Loads precomputed job index from build_index.py for fast matching.
Only encodes the CV — job embeddings are loaded from disk instantly.

Run build_index.py ONCE first, then use this for every CV.



command: python cv_matcher5.py --cv cv-jsons/Raiyen_Zayed_Rakin_CV.json --index csv-encoder/job_index --top 10 --output matches.json
"""

import argparse
import json
import re
import warnings
from typing import Any

from core.config import SBERT_MODEL

import numpy as np
from sentence_transformers import SentenceTransformer

from .constants import (
    CORE_SKILL_PATTERNS,
    CORE_SKILLS,
    SENIORITY_MAP,
    SENIORITY_PATTERNS,
    SOFT_SKILLS,
    SYNONYM_MAP,
    SYNONYM_PATTERNS,
)

warnings.filterwarnings("ignore")


# ── WEIGHTS ───────────────────────────────────────────────────────────────────

WEIGHTS = {
    "skill": 0.35,
    "education": 0.20,
    "experience": 0.15,
    "semantic": 0.11,
    "title": 0.06,
    "skill_source": 0.05,
    "nature": 0.04,
    "workplace": 0.04,
}


def skill_weight(skill):
    s = skill.lower()
    if s in CORE_SKILLS:
        return 1.5
    if s in SOFT_SKILLS:
        return 0.3
    return 1.0


FIELD_RELEVANCE = {
    "software": [
        "computer science",
        "cse",
        "software engineering",
        "it",
        "mathematics",
        "physics",
        "electrical",
        "electronics",
    ],
    "developer": ["computer science", "cse", "software engineering", "it", "mathematics", "physics"],
    "engineer": [
        "computer science",
        "cse",
        "electrical",
        "mechanical",
        "civil",
        "electronics",
        "mathematics",
        "physics",
    ],
    "data": ["computer science", "cse", "statistics", "mathematics", "physics", "economics"],
    "ai": ["computer science", "cse", "mathematics", "statistics", "physics", "electrical"],
    "web": ["computer science", "cse", "it"],
    "mobile": ["computer science", "cse", "it"],
    "android": ["computer science", "cse", "it"],
    "ios": ["computer science", "cse", "it"],
    "flutter": ["computer science", "cse", "it"],
    "backend": ["computer science", "cse", "it"],
    "frontend": ["computer science", "cse", "it"],
    "devops": ["computer science", "cse", "it"],
    "network": ["computer science", "cse", "electrical", "it"],
    "business": ["business administration", "bba", "mba", "commerce"],
    "finance": ["finance", "accounting", "business administration", "economics"],
    "marketing": ["marketing", "business administration", "bba", "mba"],
    "account": ["accounting", "finance", "commerce"],
    "teacher": ["education", "english", "any discipline"],
    "doctor": ["medicine", "mbbs"],
    "nurse": ["nursing", "health science"],
}


def field_of_study_score(cv_education, job_title, job_desc):
    cv_fields = " ".join(e.get("degree", "") + " " + e.get("institution", "") for e in cv_education).lower()
    job_text = (job_title + " " + job_desc[:300]).lower()
    for keyword, fields in FIELD_RELEVANCE.items():
        if keyword in job_text:
            for field in fields:
                if field in cv_fields:
                    return 1.0
            return 0.4  # softened from 0.2 — many roles accept adjacent disciplines
    return 0.6


# ── DEGREE LEVELS ─────────────────────────────────────────────────────────────

DEGREE_LEVELS = {
    # Secondary (Level 1)
    "ssc": 1,
    "secondary": 1,
    "মাধ্যমিক": 1,
    "এসএসসি": 1,
    # Higher Secondary (Level 2)
    "hsc": 2,
    "higher secondary": 2,
    "উচ্চ মাধ্যমিক": 2,
    "এইচএসসি": 2,
    # Diploma (Level 3)
    "diploma": 3,
    "ডিপ্লোমা": 3,
    # Bachelor (Level 4)
    "bachelor": 4,
    "b.sc": 4,
    "bsc": 4,
    "honours": 4,
    "hons": 4,
    "স্নাতক": 4,
    "ব্যাচেলর": 4,
    "বিএসসি": 4,
    "অনার্স": 4,
    # Master (Level 5)
    "master": 5,
    "msc": 5,
    "mba": 5,
    "স্নাতকোত্তর": 5,
    "মাস্টার্স": 5,
    "এমএসসি": 5,
    "এমবিএ": 5,
    # PhD / Doctorate (Level 6)
    "phd": 6,
    "doctorate": 6,
    "পিএইচডি": 6,
    "ডক্টরেট": 6,
}


def get_degree_level(text):
    text = text.lower()
    for kw, lvl in sorted(DEGREE_LEVELS.items(), key=lambda x: -x[1]):
        if kw in text:
            return lvl
    return 0


# ── SENIORITY ─────────────────────────────────────────────────────────────────


def detect_seniority(text):
    text_lower = text.lower()
    best = -1
    for kw, lvl in SENIORITY_MAP.items():
        if SENIORITY_PATTERNS[kw].search(text_lower):
            best = max(best, lvl)
    return best if best >= 0 else 2


def cv_seniority(cv):
    exp = cv.get("experience", [])
    entries = exp if isinstance(exp, list) else exp.get("entries", [])
    years = 0
    for e in entries:
        start = e.get("start_date", "")
        end = e.get("end_date", "")
        sy = re.search(r"(20\d{2})", start)
        ey = re.search(r"(20\d{2})", end) if "present" not in end.lower() else None
        if sy:
            end_year = int(ey.group(1)) if ey else 2026
            years += max(0, end_year - int(sy.group(1)))

    if years >= 8:
        year_level = 4
    elif years >= 5:
        year_level = 3
    elif years >= 2:
        year_level = 2
    elif years >= 1:
        year_level = 1
    else:
        year_level = 0

    title_levels = [detect_seniority(e.get("title", "")) for e in entries if e.get("title")]
    title_level = max(title_levels) if title_levels else year_level
    return max(year_level, title_level), years


def seniority_penalty(cv_level, job_level):
    gap = abs(cv_level - job_level)
    if gap == 0:
        return 1.0
    if gap == 1:
        return 0.85
    if gap == 2:
        return 0.65
    return 0.40


# ── SKILL HELPERS ─────────────────────────────────────────────────────────────


def normalize_skill(s):
    return re.sub(r"[^a-z0-9\+\#\.]", " ", s.lower()).strip()


def canonicalize(skill):
    s = normalize_skill(skill)
    return SYNONYM_MAP.get(s, s)


def extract_skill_set(text):
    if not text:
        return set()
    parts = re.split(r"[,\n;|/]", text)
    return {canonicalize(p) for p in parts if p.strip() and 1 < len(p.strip()) < 40}


def extract_known_skills_from_text(text):
    """Extract known skills from free-form text using pre-compiled patterns."""
    if not text:
        return set()
    text_lower = text.lower()
    found = set()
    for skill, pattern in CORE_SKILL_PATTERNS.items():
        if pattern.search(text_lower):
            found.add(canonicalize(skill))
    for variant, pattern in SYNONYM_PATTERNS.items():
        if pattern.search(text_lower):
            found.add(SYNONYM_MAP.get(variant, variant))
    return found


def get_job_skills(meta):
    """Get job skills from precomputed metadata."""
    # Use pre-canonicalized skills from the index when available
    pre_canon = meta.get("canonical_skills")
    if pre_canon:
        return set(pre_canon)

    structured = extract_skill_set(meta.get("skills_required", ""))
    from_desc = {canonicalize(s) for s in meta.get("extracted_skills", [])}
    combined = structured | from_desc
    if combined:
        return combined

    # Fallback: derive skills from free text when index metadata did not carry skills.
    fallback_text = " ".join(
        [
            meta.get("job_title", ""),
            meta.get("job_description", ""),
            meta.get("skills_required", ""),
            meta.get("additional_requirements", ""),
        ]
    )
    return extract_known_skills_from_text(fallback_text)


def _normalize_cv_skills(cv_skills):
    normalized = []
    for s in cv_skills:
        if s is None:
            continue
        if isinstance(s, str):
            normalized.append(s)
            continue
        if isinstance(s, dict):
            # Common parser shapes: {"name": ...}, {"skill": ...}, {"title": ...}
            for key in ("name", "skill", "title", "value"):
                val = s.get(key)
                if isinstance(val, str) and val.strip():
                    normalized.append(val)
                    break
            continue
        normalized.append(s)
    return normalized


def _inferred_skill_score_without_job_skills(cv_skills, job_text, semantic_hint=0.0):
    """Estimate skill fit when job skill fields are missing from metadata."""
    normalized_cv_skills = _normalize_cv_skills(cv_skills)
    cv_set = extract_skill_set(", ".join(normalized_cv_skills))
    sem = float(np.clip(semantic_hint, 0.0, 1.0))

    if not cv_set:
        return min(max(0.2 + 0.4 * sem, 0.1), 0.6)

    text_norm = normalize_skill(job_text or "")
    weighted_total = sum(skill_weight(s) for s in cv_set)
    weighted_hits = 0.0

    for s in cv_set:
        s_norm = normalize_skill(s)
        if not s_norm:
            continue
        if re.search(r"\b" + re.escape(s_norm) + r"\b", text_norm):
            weighted_hits += skill_weight(s)

    lexical_overlap = weighted_hits / weighted_total if weighted_total else 0.0

    # Keep this conservative so semantic/title signals still lead ranking.
    inferred = 0.15 + (0.7 * lexical_overlap) + (0.15 * sem)
    return float(np.clip(inferred, 0.05, 0.95))


def synthesize_cv_summary(cv) -> str:
    parts = []
    summary = cv.get("summary", "").strip()
    if summary:
        parts.append(summary)
    exp = cv.get("experience", [])
    entries = exp if isinstance(exp, list) else exp.get("entries", [])
    for e in entries[:3]:
        title = e.get("title", "")
        company = e.get("company", "")
        desc = e.get("description", "")[:150]
        if title:
            parts.append(f"{title} at {company}. {desc}")
    skills = cv.get("skills", [])
    if skills:
        parts.append(f"Skills: {', '.join(skills[:10])}")
    return " ".join(parts)


def extract_skills_from_experience(cv):
    exp = cv.get("experience", [])
    entries = exp if isinstance(exp, list) else exp.get("entries", [])
    skills = set()
    for e in entries:
        desc = e.get("description", "").lower()
        for skill, pattern in CORE_SKILL_PATTERNS.items():
            if pattern.search(desc):
                skills.add(canonicalize(skill))
    return skills


# ── SCORING FUNCTIONS ─────────────────────────────────────────────────────────


def skill_score(cv_skills, job_skills, job_text="", semantic_hint=0.0):
    normalized_cv_skills = _normalize_cv_skills(cv_skills)
    cv_set = extract_skill_set(", ".join(normalized_cv_skills))
    if not job_skills:
        return _inferred_skill_score_without_job_skills(cv_skills, job_text, semantic_hint)

    # Coverage-based metric: what fraction of *job* requirements does the CV cover?
    # This avoids penalizing candidates who have extra skills beyond the job posting.
    weighted_intersection = sum(skill_weight(s) for s in cv_set & job_skills)
    weighted_job_total = sum(skill_weight(s) for s in job_skills)
    coverage = weighted_intersection / weighted_job_total if weighted_job_total else 0.0

    # Partial matching bonus for substring overlap (e.g. "react" matches "react native")
    partial = sum(1 for cs in cv_set for js in job_skills if cs != js and (cs in js or js in cs))
    partial_bonus = min(partial / max(len(job_skills), 1), 0.2)

    # If no direct or partial overlap at all, return 0 — no free points
    if coverage == 0.0 and partial_bonus == 0.0:
        return 0.0

    # Small breadth bonus — candidates with broader skill sets get a slight edge
    breadth_bonus = min(len(cv_set) / 30, 0.1)

    return min(coverage + partial_bonus + breadth_bonus, 1.0)


def education_score(cv_education, job_edu_raw, job_title, job_desc):
    if not job_edu_raw or not job_edu_raw.strip():
        level_score = 0.7
    else:
        cv_level = max((get_degree_level(e.get("degree", "")) for e in cv_education), default=0)
        job_level = get_degree_level(job_edu_raw)
        if job_level == 0:
            level_score = 0.7
        elif cv_level >= job_level:
            level_score = 1.0
        elif cv_level == job_level - 1:
            level_score = 0.5
        else:
            level_score = 0.1
    fos = field_of_study_score(cv_education, job_title, job_desc)
    return round(level_score * 0.6 + fos * 0.4, 4)


def experience_score(cv_years, required_years):
    if required_years == 0:
        return 1.0
    if cv_years >= required_years:
        return 1.0
    if cv_years >= required_years * 0.7:
        return 0.75
    if cv_years >= required_years * 0.4:
        return 0.4
    return 0.1


def skill_source_score(cv_exp_skills, job_skills):
    if not job_skills or not cv_exp_skills:
        return 0.3
    hits = cv_exp_skills & job_skills
    return min(len(hits) / max(len(job_skills), 1), 1.0)


def nature_score(cv_level, cv_years, job_nature):
    text = str(job_nature or "").strip().lower()
    if not text:
        return 0.6
    if "intern" in text:
        return 1.0 if cv_level <= 1 or cv_years < 2 else 0.3
    if "part time" in text:
        return 0.7 if cv_level <= 2 else 0.6
    if "freelance" in text:
        return 0.7
    if "contract" in text:
        return 0.75 if cv_level >= 1 else 0.7
    if "full time" in text:
        return 1.0 if cv_years >= 1 else 0.85
    return 0.6


CITY_TOKENS = {
    "dhaka",
    "chattogram",
    "chittagong",
    "khulna",
    "rajshahi",
    "sylhet",
    "barishal",
    "rangpur",
    "cumilla",
    "gazipur",
    "narayanganj",
    "bogura",
    "mymensingh",
    "cox",
    "bazar",
}


def _location_overlap(cv_location, job_location):
    cv_text = str(cv_location or "").lower()
    job_text = str(job_location or "").lower()
    if not cv_text or not job_text:
        return False
    if "anywhere in bangladesh" in job_text:
        return True

    cv_tokens = {t for t in re.findall(r"[a-z]+", cv_text) if t in CITY_TOKENS}
    job_tokens = {t for t in re.findall(r"[a-z]+", job_text) if t in CITY_TOKENS}
    return bool(cv_tokens & job_tokens)


def workplace_score(cv_location, job_location, job_workplace):
    text = str(job_workplace or "").strip().lower()
    if not text:
        return 0.6
    if "work from home" in text and "work at office" in text:
        return 0.85
    if "work from home" in text:
        return 0.9
    if "work at office" in text or "office" in text:
        if _location_overlap(cv_location, job_location):
            return 1.0
        return 0.55
    return 0.6


def build_title_scores(model, cv, job_metadata):
    exp = cv.get("experience", [])
    entries = exp if isinstance(exp, list) else exp.get("entries", [])
    past_titles = [e.get("title", "").strip() for e in entries if e.get("title", "").strip()]

    all_job_titles = [m.get("job_title", "") or "unknown" for m in job_metadata]
    job_title_embs = model.encode(
        all_job_titles,
        batch_size=128,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    if not past_titles:
        return np.full(len(job_metadata), 0.4)

    cv_title_embs = model.encode(past_titles, normalize_embeddings=True, convert_to_numpy=True)

    # Use max-similarity instead of mean: picks the best-matching past title
    # per job. This preserves signal for career-changers whose most recent role
    # may differ significantly from earlier ones.
    sim_matrix = job_title_embs @ cv_title_embs.T  # (N_jobs, N_cv_titles)
    return np.clip(sim_matrix.max(axis=1), 0, 1)


def score_jobs(cv, job_metadata, semantic_scores, title_scores_all, cv_level, cv_years, cv_exp_skills):
    results = []
    cv_location = cv.get("location", "")
    for i, meta in enumerate(job_metadata):
        job_title = meta.get("job_title", "")
        job_desc = meta.get("job_description", "")
        job_edu = meta.get("education_requirements", "")
        job_add = meta.get("additional_requirements", "")
        job_skills = get_job_skills(meta)

        s_sem = max(0.0, float(semantic_scores[i]))
        s_skill = skill_score(
            cv.get("skills", []),
            job_skills,
            job_text=f"{job_title} {job_desc} {job_add}",
            semantic_hint=s_sem,
        )
        s_edu = education_score(cv.get("education", []), job_edu, job_title, job_desc)
        s_exp = experience_score(cv_years, meta.get("required_years", 0))
        s_title = float(title_scores_all[i])
        s_source = skill_source_score(cv_exp_skills, job_skills)
        s_nature = nature_score(cv_level, cv_years, meta.get("job_nature", ""))
        s_workplace = workplace_score(cv_location, meta.get("location", ""), meta.get("job_workplace", ""))

        job_level = meta.get("seniority_level", 2)
        penalty = seniority_penalty(cv_level, job_level)

        raw = (
            s_skill * WEIGHTS["skill"]
            + s_edu * WEIGHTS["education"]
            + s_exp * WEIGHTS["experience"]
            + s_sem * WEIGHTS["semantic"]
            + s_title * WEIGHTS["title"]
            + s_source * WEIGHTS["skill_source"]
            + s_nature * WEIGHTS["nature"]
            + s_workplace * WEIGHTS["workplace"]
        )
        final = raw * penalty * 100

        results.append(
            {
                "job_id": meta.get("job_id"),
                "job_title": job_title,
                "company": meta.get("company"),
                "location": meta.get("location"),
                "salary_range": meta.get("salary_range"),
                "deadline": meta.get("deadline"),
                "final_score": round(final, 2),
                "breakdown": {
                    "skill_match": round(s_skill * 100, 1),
                    "education_match": round(s_edu * 100, 1),
                    "experience_match": round(s_exp * 100, 1),
                    "semantic_match": round(s_sem * 100, 1),
                    "title_match": round(s_title * 100, 1),
                    "skill_source": round(s_source * 100, 1),
                    "nature_match": round(s_nature * 100, 1),
                    "workplace_match": round(s_workplace * 100, 1),
                    "seniority_penalty": round(penalty, 2),
                },
            }
        )

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


# ── MAIN MATCH FUNCTION ───────────────────────────────────────────────────────


def match_cv_dict(
    cv: dict,
    job_embeddings,
    job_metadata: list,
    model,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """
    Match a pre-parsed CV dict against preloaded job index.
    Called by the FastAPI server — model/index already in memory.
    """
    cv_level, cv_years = cv_seniority(cv)
    cv_summary = synthesize_cv_summary(cv)
    cv_exp_skills = extract_skills_from_experience(cv)

    cv_emb = model.encode(cv_summary, normalize_embeddings=True, convert_to_numpy=True)
    semantic_scores = job_embeddings @ cv_emb

    title_scores_all = build_title_scores(model, cv, job_metadata)
    results = score_jobs(cv, job_metadata, semantic_scores, title_scores_all, cv_level, cv_years, cv_exp_skills)
    return results[:top_n]


def match(cv_path, index_prefix, top_n=10, output_path=None):
    # Load CV
    with open(cv_path, "r", encoding="utf-8") as f:
        cv = json.load(f)

    # Load precomputed index
    emb_path = f"{index_prefix}_embeddings.npy"
    meta_path = f"{index_prefix}_metadata.json"

    print(f"CV      : {cv.get('name')}")
    print("Loading job index...")
    job_embeddings = np.load(emb_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        job_metadata = json.load(f)
    print(f"Jobs    : {len(job_metadata)} loaded from index")

    print("Loading Sentence-BERT model...")
    model = SentenceTransformer(SBERT_MODEL)

    # Pre-compute CV values once
    cv_level, cv_years = cv_seniority(cv)
    cv_summary = synthesize_cv_summary(cv)
    cv_exp_skills = extract_skills_from_experience(cv)

    level_labels = {0: "Intern", 1: "Junior", 2: "Mid-level", 3: "Senior", 4: "Lead", 5: "Director"}
    print(f"Candidate: {cv_years} yrs | {level_labels.get(cv_level, 'Mid-level')}")
    print(f"Summary  : {'original' if cv.get('summary', '').strip() else 'synthesized'}")

    # Encode CV summary once
    cv_emb = model.encode(cv_summary, normalize_embeddings=True, convert_to_numpy=True)

    # Batch cosine similarity: CV vs all jobs at once (instant)
    print("Computing semantic similarities...")
    semantic_scores = job_embeddings @ cv_emb  # dot product on normalized = cosine sim

    print("Pre-encoding job titles in batch...")
    title_scores_all = build_title_scores(model, cv, job_metadata)

    print(f"Scoring {len(job_metadata)} jobs...")
    results = score_jobs(cv, job_metadata, semantic_scores, title_scores_all, cv_level, cv_years, cv_exp_skills)

    top_results = results[:top_n]

    print(f"\n{'=' * 65}")
    print(f"TOP {top_n} MATCHES FOR {cv.get('name', '').upper()}")
    print(f"{'=' * 65}")
    for i, r in enumerate(top_results, 1):
        b = r["breakdown"]
        print(f"\n#{i} [{r['final_score']}%] {r['job_title']}")
        print(f"    Company : {r['company']}")
        print(f"    Location: {r['location']}")
        print(f"    Salary  : {r['salary_range']}")
        print(
            f"    Skill={b['skill_match']}% | Edu={b['education_match']}% | "
            f"Exp={b['experience_match']}% | Semantic={b['semantic_match']}% | "
            f"Title={b['title_match']}% | Nature={b['nature_match']}% | "
            f"Workplace={b['workplace_match']}% | Penalty={b['seniority_penalty']}"
        )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(top_results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved -> {output_path}")

    return top_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV-Job Matcher v6 (indexed)")
    parser.add_argument("--cv", required=True, help="Parsed CV JSON path")
    parser.add_argument("--index", default="job_index", help="Index prefix from build_index.py")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    match(args.cv, args.index, args.top, args.output)
