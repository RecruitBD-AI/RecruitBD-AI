"""
cv_matcher.py — CV to Job Matching Engine (v2)
-----------------------------------------------
Scores each job against a parsed CV using 6 signals:

  1. Skill match        (35%) — Weighted Jaccard + synonyms + partial match
  2. Education match    (20%) — Degree level (60%) + field of study (40%)
  3. Experience match   (15%) — CV years vs required years
  4. Semantic match     (15%) — Sentence-BERT on summary vs job description
  5. Job title match    (10%) — Sentence-BERT on past titles vs job title
  6. Skill source bonus  (5%) — Skills found in experience/projects vs job

Usage:
  python cv_matcher.py --cv parsed_cv.json --jobs jobs.csv --top 10
  python cv_matcher.py --cv parsed_cv.json --jobs jobs.csv --top 20 --output matches.json
"""

import re
import csv
import json
import argparse
from sentence_transformers import SentenceTransformer, util


# ── WEIGHTS ───────────────────────────────────────────────────────────────────

WEIGHTS = {
    "skill":        0.35,
    "education":    0.20,
    "experience":   0.15,
    "semantic":     0.15,
    "title":        0.10,
    "skill_source": 0.05,
}


# ── SKILL SYNONYMS ────────────────────────────────────────────────────────────

SKILL_SYNONYMS = [
    {"javascript", "js", "node.js", "nodejs", "node js"},
    {"typescript", "ts"},
    {"python", "py"},
    {"java", "core java"},
    {"c++", "cpp", "c plus plus"},
    {"c#", "csharp", "c sharp"},
    {"machine learning", "ml"},
    {"deep learning", "dl"},
    {"natural language processing", "nlp", "text mining"},
    {"computer vision", "cv", "image processing"},
    {"artificial intelligence", "ai"},
    {"sql", "mysql", "postgresql", "postgres", "mssql", "oracle sql"},
    {"mongodb", "mongo"},
    {"react", "react.js", "reactjs"},
    {"vue", "vue.js", "vuejs"},
    {"angular", "angularjs", "angular.js"},
    {"flutter", "dart flutter"},
    {"spring boot", "spring", "spring framework"},
    {"git", "github", "gitlab", "version control"},
    {"docker", "containerization"},
    {"kubernetes", "k8s"},
    {"aws", "amazon web services"},
    {"gcp", "google cloud", "google cloud platform"},
    {"azure", "microsoft azure"},
    {"rest", "rest api", "restful", "restful api"},
    {"html", "html5"},
    {"css", "css3", "scss", "sass"},
    {"linux", "ubuntu", "unix"},
    {"microsoft office", "ms office", "excel", "word", "powerpoint"},
]

def build_synonym_map() -> dict:
    mapping = {}
    for group in SKILL_SYNONYMS:
        canonical = sorted(group)[0]
        for variant in group:
            mapping[variant] = canonical
    return mapping

SYNONYM_MAP = build_synonym_map()


# ── SKILL WEIGHTS ─────────────────────────────────────────────────────────────

CORE_SKILLS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "dart",
    "flutter", "react", "node.js", "spring boot", "django", "fastapi",
    "machine learning", "deep learning", "nlp", "computer vision",
    "docker", "kubernetes", "aws", "azure", "gcp", "mongodb", "postgresql",
    "git", "rest api", "tensorflow", "pytorch", "scikit-learn",
}

SOFT_SKILLS = {
    "leadership", "communication", "teamwork", "collaboration", "problem solving",
    "time management", "critical thinking", "active listening", "proposal writing",
    "public presentations", "adaptability",
}

def skill_weight(skill: str) -> float:
    s = skill.lower()
    if s in CORE_SKILLS:  return 1.5
    if s in SOFT_SKILLS:  return 0.3
    return 1.0


# ── FIELD OF STUDY ────────────────────────────────────────────────────────────

FIELD_RELEVANCE = {
    "software":  ["computer science", "cse", "software engineering", "information technology", "it"],
    "engineer":  ["computer science", "cse", "electrical", "mechanical", "civil", "engineering"],
    "data":      ["computer science", "cse", "statistics", "mathematics", "data science"],
    "ai":        ["computer science", "cse", "mathematics", "statistics", "data science"],
    "ml":        ["computer science", "cse", "mathematics", "statistics"],
    "web":       ["computer science", "cse", "information technology", "it"],
    "mobile":    ["computer science", "cse", "information technology", "it"],
    "network":   ["computer science", "cse", "electrical", "information technology"],
    "developer": ["computer science", "cse", "software engineering", "information technology"],
    "business":  ["business administration", "bba", "mba", "commerce", "management"],
    "finance":   ["finance", "accounting", "business administration", "economics"],
    "marketing": ["marketing", "business administration", "bba", "mba"],
    "account":   ["accounting", "finance", "commerce", "business administration"],
    "teacher":   ["education", "english", "bengali", "any discipline"],
    "nurse":     ["nursing", "health science", "medicine"],
    "doctor":    ["medicine", "mbbs", "health science"],
    "pharmacy":  ["pharmacy", "health science"],
}

def field_of_study_score(cv_education: list, job_title: str, job_desc: str) -> float:
    cv_fields = " ".join(
        e.get("degree", "") + " " + e.get("institution", "")
        for e in cv_education
    ).lower()

    job_text = (job_title + " " + job_desc).lower()
    best_score = 0.5

    for keyword, relevant_fields in FIELD_RELEVANCE.items():
        if keyword in job_text:
            for field in relevant_fields:
                if field in cv_fields:
                    return 1.0
            best_score = min(best_score, 0.3)

    return best_score


# ── DEGREE LEVELS ─────────────────────────────────────────────────────────────

DEGREE_LEVELS = {
    "ssc": 1, "secondary": 1,
    "hsc": 2, "higher secondary": 2, "a level": 2,
    "diploma": 3,
    "bachelor": 4, "b.sc": 4, "bsc": 4, "b.a": 4, "honours": 4, "hons": 4,
    "master": 5, "m.sc": 5, "msc": 5, "mba": 5, "m.a": 5,
    "phd": 6, "doctorate": 6,
}

def get_degree_level(text: str) -> int:
    text_lower = text.lower()
    for keyword, level in sorted(DEGREE_LEVELS.items(), key=lambda x: -x[1]):
        if keyword in text_lower:
            return level
    return 0


# ── SKILL HELPERS ─────────────────────────────────────────────────────────────

def normalize_skill(s: str) -> str:
    return re.sub(r"[^a-z0-9\+\#\.]", " ", s.lower()).strip()

def canonicalize(skill: str) -> str:
    s = normalize_skill(skill)
    return SYNONYM_MAP.get(s, s)

def extract_skill_set(text: str) -> set:
    if not text:
        return set()
    parts = re.split(r"[,\n;|]", text)
    return {canonicalize(p) for p in parts if p.strip() and len(p.strip()) > 1}

def extract_skills_from_experience(cv: dict) -> set:
    """Pull core skills mentioned inside experience/project descriptions."""
    skills = set()
    exp = cv.get("experience", [])
    entries = exp if isinstance(exp, list) else exp.get("entries", [])
    for entry in entries:
        desc = entry.get("description", "").lower()
        for skill in CORE_SKILLS:
            if skill in desc:
                skills.add(canonicalize(skill))
    return skills


# ── SCORING FUNCTIONS ─────────────────────────────────────────────────────────

def skill_score(cv_skills: list, job_skills_raw: str, job_desc: str) -> float:
    cv_set  = extract_skill_set(", ".join(cv_skills))
    job_set = extract_skill_set(job_skills_raw) | extract_skill_set(job_desc)

    if not job_set:
        return 0.5

    weighted_intersection = sum(skill_weight(s) for s in cv_set & job_set)
    weighted_union        = sum(skill_weight(s) for s in cv_set | job_set)
    jaccard = weighted_intersection / weighted_union if weighted_union else 0.0

    partial_hits = sum(
        1 for cs in cv_set for js in job_set
        if cs != js and (cs in js or js in cs)
    )
    partial_bonus = min(partial_hits / max(len(job_set), 1), 0.2)

    return min(jaccard + partial_bonus, 1.0)


def skill_source_bonus(cv: dict, job_skills_raw: str, job_desc: str) -> float:
    exp_skills = extract_skills_from_experience(cv)
    job_set    = extract_skill_set(job_skills_raw) | extract_skill_set(job_desc)
    if not job_set or not exp_skills:
        return 0.5
    hits = exp_skills & job_set
    return min(len(hits) / max(len(job_set), 1), 1.0)


def education_score(cv_education: list, job_edu_raw: str, job_title: str, job_desc: str) -> float:
    # Degree level (60% of education score)
    if not job_edu_raw or not job_edu_raw.strip():
        level_score = 0.8
    else:
        cv_level  = max((get_degree_level(e.get("degree", "")) for e in cv_education), default=0)
        job_level = get_degree_level(job_edu_raw)
        if job_level == 0:
            level_score = 0.8
        elif cv_level >= job_level:
            level_score = 1.0
        elif cv_level == job_level - 1:
            level_score = 0.6
        else:
            level_score = 0.2

    # Field of study relevance (40% of education score)
    fos = field_of_study_score(cv_education, job_title, job_desc)

    return round(level_score * 0.6 + fos * 0.4, 4)


def extract_required_years(exp_text: str) -> float:
    if not exp_text:
        return 0.0
    exp_text = exp_text.lower()
    if any(w in exp_text for w in ["na", "fresher", "no experience", "not required"]):
        return 0.0
    m = re.search(r"(\d+)\s*to\s*(\d+)", exp_text)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)\s*year", exp_text)
    if m:
        return float(m.group(1))
    return 0.0

def experience_score(cv_exp: dict, job_exp_raw: str) -> float:
    cv_years = cv_exp.get("total_years", 0) or 0
    required = extract_required_years(job_exp_raw)
    if required == 0:   return 1.0
    if cv_years >= required:        return 1.0
    if cv_years >= required * 0.6:  return 0.7
    if cv_years > 0:                return 0.4
    return 0.1


def semantic_score(cv_summary: str, job_desc: str, model: SentenceTransformer) -> float:
    if not cv_summary or not job_desc:
        return 0.5
    e1 = model.encode(cv_summary[:512], convert_to_tensor=True)
    e2 = model.encode(job_desc[:512],   convert_to_tensor=True)
    return max(0.0, float(util.cos_sim(e1, e2).item()))


def title_score(cv: dict, job_title: str, model: SentenceTransformer) -> float:
    if not job_title:
        return 0.5

    exp = cv.get("experience", [])
    entries = exp if isinstance(exp, list) else exp.get("entries", [])
    past_titles = [e.get("title", "") for e in entries if e.get("title", "").strip()]

    if not past_titles:
        summary = cv.get("summary", "")
        if not summary:
            return 0.5
        past_titles = [summary[:200]]

    job_emb = model.encode(job_title, convert_to_tensor=True)
    scores = []
    for title in past_titles:
        t_emb = model.encode(title, convert_to_tensor=True)
        scores.append(max(0.0, float(util.cos_sim(t_emb, job_emb).item())))

    return max(scores) if scores else 0.5


# ── MAIN SCORER ───────────────────────────────────────────────────────────────

def score_job(cv: dict, job: dict, model: SentenceTransformer) -> dict:
    cv_exp_raw = cv.get("experience", [])
    cv_exp = cv_exp_raw if isinstance(cv_exp_raw, dict) else {"total_years": 0, "entries": cv_exp_raw}

    job_title = job.get("job_title", "")
    job_desc  = job.get("job_description", "") or job.get("job_context", "")

    s_skill  = skill_score(cv.get("skills", []), job.get("skills_required", ""), job_desc)
    s_edu    = education_score(cv.get("education", []), job.get("education_requirements", ""), job_title, job_desc)
    s_exp    = experience_score(cv_exp, job.get("experience", ""))
    s_sem    = semantic_score(cv.get("summary", ""), job_desc, model)
    s_title  = title_score(cv, job_title, model)
    s_source = skill_source_bonus(cv, job.get("skills_required", ""), job_desc)

    final = (
        s_skill  * WEIGHTS["skill"] +
        s_edu    * WEIGHTS["education"] +
        s_exp    * WEIGHTS["experience"] +
        s_sem    * WEIGHTS["semantic"] +
        s_title  * WEIGHTS["title"] +
        s_source * WEIGHTS["skill_source"]
    ) * 100

    return {
        "job_id":       job.get("job_id"),
        "job_title":    job_title,
        "company":      job.get("company_name"),
        "location":     job.get("location"),
        "salary_range": job.get("salary_range"),
        "deadline":     job.get("deadline"),
        "final_score":  round(final, 2),
        "breakdown": {
            "skill_match":        round(s_skill  * 100, 1),
            "education_match":    round(s_edu    * 100, 1),
            "experience_match":   round(s_exp    * 100, 1),
            "semantic_match":     round(s_sem    * 100, 1),
            "title_match":        round(s_title  * 100, 1),
            "skill_source_bonus": round(s_source * 100, 1),
        }
    }


def match(cv_path: str, jobs_path: str, top_n: int = 10, output_path: str = None):
    with open(cv_path, "r", encoding="utf-8") as f:
        cv = json.load(f)
    with open(jobs_path, "r", encoding="utf-8") as f:
        jobs = list(csv.DictReader(f))

    print(f"CV      : {cv.get('name')}")
    print(f"Jobs    : {len(jobs)}")
    print(f"Loading Sentence-BERT model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Scoring {len(jobs)} jobs...\n")
    results = []
    for i, job in enumerate(jobs):
        results.append(score_job(cv, job, model))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(jobs)} scored...")

    results.sort(key=lambda x: x["final_score"], reverse=True)
    top_results = results[:top_n]

    print(f"\n{'='*65}")
    print(f"TOP {top_n} JOB MATCHES FOR {cv.get('name', 'Candidate').upper()}")
    print(f"{'='*65}")
    for i, r in enumerate(top_results, 1):
        b = r["breakdown"]
        print(f"\n#{i} [{r['final_score']}%] {r['job_title']}")
        print(f"    Company  : {r['company']}")
        print(f"    Location : {r['location']}")
        print(f"    Salary   : {r['salary_range']}")
        print(f"    Skill={b['skill_match']}% | Edu={b['education_match']}% | Exp={b['experience_match']}%"
              f" | Semantic={b['semantic_match']}% | Title={b['title_match']}% | SrcBonus={b['skill_source_bonus']}%")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(top_results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved → {output_path}")

    return top_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV-Job Matcher v2")
    parser.add_argument("--cv",     required=True, help="Parsed CV JSON path")
    parser.add_argument("--jobs",   required=True, help="Jobs CSV path")
    parser.add_argument("--top",    type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    match(args.cv, args.jobs, args.top, args.output)