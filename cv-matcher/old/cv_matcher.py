"""
cv_matcher.py — CV to Job Matching Engine
------------------------------------------
Scores each job against a parsed CV using 4 signals:
  1. Skill match      (40%) — Jaccard similarity on skills
  2. Education match  (25%) — degree level vs job requirement
  3. Experience match (20%) — CV years vs required years
  4. Semantic match   (15%) — Sentence-BERT on summary vs job description

Usage:
  python cv_matcher.py --cv parsed_cv.json --jobs jobs.csv --top 10
  python cv_matcher.py --cv parsed_cv.json --jobs jobs.csv --top 20 --output matches.json
"""

import re
import csv
import json
import argparse
from sentence_transformers import SentenceTransformer, util


# ── CONFIG ────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "skill":      0.50,
    "education":  0.20,
    "experience": 0.15,
    "semantic":   0.15,
}

DEGREE_LEVELS = {
    "ssc": 1, "secondary": 1,
    "hsc": 2, "higher secondary": 2, "a level": 2,
    "diploma": 3,
    "bachelor": 4, "b.sc": 4, "bsc": 4, "b.a": 4, "ba": 4,
    "honours": 4, "hons": 4,
    "master": 5, "m.sc": 5, "msc": 5, "mba": 5, "m.a": 5,
    "phd": 6, "doctorate": 6,
}


# ── SKILL MATCH ───────────────────────────────────────────────────────────────

def normalize_skill(s: str) -> str:
    return re.sub(r"[^a-z0-9\+\#]", " ", s.lower()).strip()

def extract_skill_set(text: str) -> set:
    """Split comma/newline separated skill text into a normalized set."""
    if not text:
        return set()
    parts = re.split(r"[,\n;|]", text)
    return {normalize_skill(p) for p in parts if p.strip() and len(p.strip()) > 1}

def skill_score(cv_skills: list, job_skills_raw: str, job_desc: str) -> float:
    cv_set = extract_skill_set(", ".join(cv_skills))
    job_set = extract_skill_set(job_skills_raw) | extract_skill_set(job_desc)

    if not job_set:
        return 0.5  # no skill info in job → neutral score

    intersection = cv_set & job_set
    union = cv_set | job_set

    if not union:
        return 0.0

    jaccard = len(intersection) / len(union)
    # Also reward partial string matches (e.g. "python" in "python django")
    partial_hits = sum(
        1 for cs in cv_set
        for js in job_set
        if cs in js or js in cs
    )
    partial_bonus = min(partial_hits / max(len(job_set), 1), 0.3)

    return min(jaccard + partial_bonus, 1.0)


# ── EDUCATION MATCH ───────────────────────────────────────────────────────────

def get_degree_level(text: str) -> int:
    """Return numeric degree level from text."""
    text_lower = text.lower()
    for keyword, level in sorted(DEGREE_LEVELS.items(), key=lambda x: -x[1]):
        if keyword in text_lower:
            return level
    return 0

def education_score(cv_education: list, job_edu_raw: str) -> float:
    if not job_edu_raw or not job_edu_raw.strip():
        return 0.8  # no requirement stated → mostly fine

    # Get highest CV degree
    cv_level = max(
        (get_degree_level(e.get("degree", "")) for e in cv_education),
        default=0
    )
    job_level = get_degree_level(job_edu_raw)

    if job_level == 0:
        return 0.8  # can't parse job requirement → neutral

    if cv_level >= job_level:
        return 1.0
    elif cv_level == job_level - 1:
        return 0.6  # one level below
    else:
        return 0.2  # significantly underqualified


# ── EXPERIENCE MATCH ──────────────────────────────────────────────────────────

def extract_required_years(exp_text: str) -> float:
    """Parse '2 to 5 years', 'At least 3 years', '1 year' etc."""
    if not exp_text:
        return 0.0
    exp_text = exp_text.lower()

    # "Na" or "freshers" → 0 years required
    if any(w in exp_text for w in ["na", "fresher", "no experience", "not required"]):
        return 0.0

    # "2 to 5 years" → take minimum (2)
    range_match = re.search(r"(\d+)\s*to\s*(\d+)", exp_text)
    if range_match:
        return float(range_match.group(1))

    # "at least 3 years" or "minimum 2 years"
    single_match = re.search(r"(\d+)\s*year", exp_text)
    if single_match:
        return float(single_match.group(1))

    return 0.0

def experience_score(cv_exp: dict, job_exp_raw: str) -> float:
    cv_years = cv_exp.get("total_years", 0) or 0
    required = extract_required_years(job_exp_raw)

    if required == 0:
        return 1.0  # no experience required → full score

    if cv_years >= required:
        return 1.0
    elif cv_years >= required * 0.6:
        return 0.7  # close enough
    elif cv_years > 0:
        return 0.4  # has some exp but not enough
    else:
        return 0.1  # no experience at all


# ── SEMANTIC MATCH ────────────────────────────────────────────────────────────

def semantic_score(cv_summary: str, job_desc: str, model: SentenceTransformer) -> float:
    if not cv_summary or not job_desc:
        return 0.5  # missing info → neutral

    # Truncate to avoid token limits
    cv_text  = cv_summary[:512]
    job_text = job_desc[:512]

    emb_cv  = model.encode(cv_text,  convert_to_tensor=True)
    emb_job = model.encode(job_text, convert_to_tensor=True)

    score = util.cos_sim(emb_cv, emb_job).item()
    # Cosine similarity can be negative, clamp to [0, 1]
    return max(0.0, float(score))


# ── MAIN SCORER ───────────────────────────────────────────────────────────────

def score_job(cv: dict, job: dict, model: SentenceTransformer) -> dict:
    s_skill = skill_score(
        cv.get("skills", []),
        job.get("skills_required", ""),
        job.get("job_description", "")
    )
    s_edu = education_score(
        cv.get("education", []),
        job.get("education_requirements", "")
    )
    cv_exp_raw = cv.get("experience", [])
    cv_exp = cv_exp_raw if isinstance(cv_exp_raw, dict) else {"total_years": 0, "entries": cv_exp_raw}
    s_exp = experience_score(cv_exp, job.get("experience", ""))
    s_sem = semantic_score(
        cv.get("summary", ""),
        job.get("job_description", "") or job.get("job_context", ""),
        model
    )

    final = (
        s_skill      * WEIGHTS["skill"] +
        s_edu        * WEIGHTS["education"] +
        s_exp        * WEIGHTS["experience"] +
        s_sem        * WEIGHTS["semantic"]
    ) * 100

    return {
        "job_id":       job.get("job_id"),
        "job_title":    job.get("job_title"),
        "company":      job.get("company_name"),
        "location":     job.get("location"),
        "salary_range": job.get("salary_range"),
        "deadline":     job.get("deadline"),
        "final_score":  round(final, 2),
        "breakdown": {
            "skill_match":      round(s_skill * 100, 1),
            "education_match":  round(s_edu   * 100, 1),
            "experience_match": round(s_exp   * 100, 1),
            "semantic_match":   round(s_sem   * 100, 1),
        }
    }


def match(cv_path: str, jobs_path: str, top_n: int = 10, output_path: str = None):
    # Load CV
    with open(cv_path, "r", encoding="utf-8") as f:
        cv = json.load(f)

    # Load jobs
    with open(jobs_path, "r", encoding="utf-8") as f:
        jobs = list(csv.DictReader(f))

    print(f"CV: {cv.get('name')}")
    print(f"Jobs loaded: {len(jobs)}")
    print(f"Loading Sentence-BERT model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Scoring {len(jobs)} jobs...\n")
    results = []
    for i, job in enumerate(jobs):
        scored = score_job(cv, job, model)
        results.append(scored)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(jobs)} scored...")

    # Sort by final score descending
    results.sort(key=lambda x: x["final_score"], reverse=True)
    top_results = results[:top_n]

    # Print top matches
    print(f"\n{'='*60}")
    print(f"TOP {top_n} JOB MATCHES FOR {cv.get('name', 'Candidate').upper()}")
    print(f"{'='*60}")
    for i, r in enumerate(top_results, 1):
        print(f"\n#{i} [{r['final_score']}%] {r['job_title']}")
        print(f"    Company  : {r['company']}")
        print(f"    Location : {r['location']}")
        print(f"    Salary   : {r['salary_range']}")
        print(f"    Breakdown: Skill={r['breakdown']['skill_match']}% | "
              f"Edu={r['breakdown']['education_match']}% | "
              f"Exp={r['breakdown']['experience_match']}% | "
              f"Semantic={r['breakdown']['semantic_match']}%")

    # Save output
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(top_results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {output_path}")

    return top_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV-Job Matcher")
    parser.add_argument("--cv",     required=True, help="Path to parsed CV JSON")
    parser.add_argument("--jobs",   required=True, help="Path to jobs CSV")
    parser.add_argument("--top",    type=int, default=10, help="Number of top matches to return")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    match(args.cv, args.jobs, args.top, args.output)