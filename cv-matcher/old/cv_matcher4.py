"""
cv_matcher.py — CV to Job Matching Engine (v4)
-----------------------------------------------
Key fixes over v3:
  - Extracts years of experience from job_description text (was always 100% before)
  - Extracts skills from job_description when skills_required is empty (was 50% neutral)
  - Builds CV summary from experience descriptions when summary field is empty
  - Seniority detection from job_description text, not just title

Signals:
  1. Skill match      (35%) — CV skills vs skills extracted from full job text
  2. Education match  (20%) — Degree level + field of study
  3. Experience match (15%) — Years extracted from job_description text
  4. Semantic match   (15%) — SBERT on synthesized CV summary vs job description
  5. Title match      (10%) — SBERT on past titles vs job title
  6. Skill source     (05%) — Skills from experience descriptions vs job

Usage:
  python cv_matcher.py --cv parsed_cv.json --jobs jobs.csv --top 10
  python cv_matcher.py --cv parsed_cv.json --jobs jobs.csv --top 20 --output matches.json
"""

import re
import csv
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

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
    {"javascript", "js", "node.js", "nodejs"},
    {"typescript", "ts"},
    {"python", "py"},
    {"java", "core java"},
    {"c++", "cpp"},
    {"c#", "csharp"},
    {"machine learning", "ml"},
    {"deep learning", "dl"},
    {"natural language processing", "nlp"},
    {"computer vision", "cv"},
    {"artificial intelligence", "ai"},
    {"sql", "mysql", "postgresql", "postgres", "oracle sql"},
    {"mongodb", "mongo"},
    {"react", "react.js", "reactjs"},
    {"vue", "vue.js", "vuejs"},
    {"angular", "angularjs"},
    {"flutter", "dart"},
    {"spring boot", "spring"},
    {"git", "github", "gitlab"},
    {"docker", "containerization"},
    {"kubernetes", "k8s"},
    {"aws", "amazon web services"},
    {"gcp", "google cloud"},
    {"azure", "microsoft azure"},
    {"rest", "rest api", "restful"},
    {"html", "html5"},
    {"css", "css3", "scss"},
    {"linux", "ubuntu", "unix"},
]

def build_synonym_map():
    mapping = {}
    for group in SKILL_SYNONYMS:
        canonical = sorted(group)[0]
        for variant in group:
            mapping[variant] = canonical
    return mapping

SYNONYM_MAP = build_synonym_map()

CORE_SKILLS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "dart",
    "flutter", "react", "node.js", "spring boot", "django", "fastapi", "flask",
    "machine learning", "deep learning", "nlp", "computer vision", "ai",
    "docker", "kubernetes", "aws", "azure", "gcp", "mongodb", "postgresql",
    "git", "rest api", "tensorflow", "pytorch", "scikit-learn", "go",
    "kotlin", "swift", "php", "ruby", "scala", "redis", "firebase",
    "android", "ios", "flutter", "jetpack compose", "spring boot",
    "selenium", "jenkins", "linux", "bash", "r", "matlab",
}

SOFT_SKILLS = {
    "leadership", "communication", "teamwork", "collaboration",
    "time management", "critical thinking", "active listening",
    "problem solving", "adaptability", "presentation",
}

def skill_weight(skill):
    s = skill.lower()
    if s in CORE_SKILLS:  return 1.5
    if s in SOFT_SKILLS:  return 0.3
    return 1.0


# ── FIELD OF STUDY ────────────────────────────────────────────────────────────

FIELD_RELEVANCE = {
    "software":   ["computer science", "cse", "software engineering", "it"],
    "developer":  ["computer science", "cse", "software engineering", "it"],
    "engineer":   ["computer science", "cse", "electrical", "mechanical", "civil"],
    "data":       ["computer science", "cse", "statistics", "mathematics"],
    "ai":         ["computer science", "cse", "mathematics", "statistics"],
    "web":        ["computer science", "cse", "it"],
    "mobile":     ["computer science", "cse", "it"],
    "android":    ["computer science", "cse", "it"],
    "ios":        ["computer science", "cse", "it"],
    "flutter":    ["computer science", "cse", "it"],
    "backend":    ["computer science", "cse", "it"],
    "frontend":   ["computer science", "cse", "it"],
    "fullstack":  ["computer science", "cse", "it"],
    "devops":     ["computer science", "cse", "it"],
    "network":    ["computer science", "cse", "electrical", "it"],
    "business":   ["business administration", "bba", "mba", "commerce"],
    "finance":    ["finance", "accounting", "business administration"],
    "marketing":  ["marketing", "business administration", "bba", "mba"],
    "account":    ["accounting", "finance", "commerce"],
    "teacher":    ["education", "english", "any discipline"],
    "doctor":     ["medicine", "mbbs"],
    "nurse":      ["nursing", "health science"],
}

def field_of_study_score(cv_education, job_title, job_desc):
    cv_fields = " ".join(
        e.get("degree", "") + " " + e.get("institution", "")
        for e in cv_education
    ).lower()
    job_text = (job_title + " " + job_desc[:300]).lower()
    for keyword, fields in FIELD_RELEVANCE.items():
        if keyword in job_text:
            for field in fields:
                if field in cv_fields:
                    return 1.0
            return 0.2  # keyword matched but field didn't
    return 0.6  # no keyword matched — neutral


# ── DEGREE LEVELS ─────────────────────────────────────────────────────────────

DEGREE_LEVELS = {
    "ssc": 1, "secondary": 1,
    "hsc": 2, "higher secondary": 2,
    "diploma": 3,
    "bachelor": 4, "b.sc": 4, "bsc": 4, "honours": 4, "hons": 4,
    "master": 5, "msc": 5, "mba": 5,
    "phd": 6, "doctorate": 6,
}

def get_degree_level(text):
    text = text.lower()
    for kw, lvl in sorted(DEGREE_LEVELS.items(), key=lambda x: -x[1]):
        if kw in text:
            return lvl
    return 0


# ── SENIORITY ─────────────────────────────────────────────────────────────────

SENIORITY_MAP = {
    "intern": 0, "trainee": 0, "fresher": 0,
    "junior": 1, "jr": 1,
    "associate": 2, "executive": 2, "officer": 2,
    "senior": 3, "sr": 3,
    "lead": 4, "manager": 4,
    "head": 5, "director": 5, "gm": 5,
}

def detect_seniority(text):
    text = text.lower()
    best = -1
    for kw, lvl in SENIORITY_MAP.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", text):
            best = max(best, lvl)
    return best if best >= 0 else 2

def cv_seniority(cv):
    exp = cv.get("experience", [])
    entries = exp if isinstance(exp, list) else exp.get("entries", [])

    # From start/end dates in experience entries
    years = 0
    for e in entries:
        start = e.get("start_date", "")
        end   = e.get("end_date", "")
        sy = re.search(r"(20\d{2})", start)
        ey = re.search(r"(20\d{2})", end) if "present" not in end.lower() else None
        if sy:
            end_year = int(ey.group(1)) if ey else 2026
            years += max(0, end_year - int(sy.group(1)))

    if years >= 8:   year_level = 4
    elif years >= 5: year_level = 3
    elif years >= 2: year_level = 2
    elif years >= 1: year_level = 1
    else:            year_level = 0

    title_levels = [detect_seniority(e.get("title", "")) for e in entries if e.get("title")]
    title_level  = max(title_levels) if title_levels else year_level
    return max(year_level, title_level), years

def seniority_penalty(cv_level, job_level):
    gap = abs(cv_level - job_level)
    if gap == 0:  return 1.0
    if gap == 1:  return 0.85
    if gap == 2:  return 0.65
    return 0.40


# ── TEXT-BASED SKILL EXTRACTION ───────────────────────────────────────────────

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

def extract_skills_from_desc(job_desc):
    """
    Scan job description for known core skills mentioned anywhere in the text.
    This handles jobs where skills_required is empty.
    """
    if not job_desc:
        return set()
    text = job_desc.lower()
    found = set()
    for skill in CORE_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text):
            found.add(canonicalize(skill))
    return found

def get_job_skills(job):
    """Combine skills_required field + skills extracted from description."""
    structured = extract_skill_set(job.get("skills_required", ""))
    from_desc  = extract_skills_from_desc(job.get("job_description", ""))
    return structured | from_desc


# ── CV SUMMARY SYNTHESIS ──────────────────────────────────────────────────────

def synthesize_cv_summary(cv):
    """
    Build a summary from experience descriptions if summary field is empty.
    This fixes the semantic_match always being 0.5 neutral.
    """
    summary = cv.get("summary", "").strip()
    if summary:
        return summary

    exp = cv.get("experience", [])
    entries = exp if isinstance(exp, list) else exp.get("entries", [])

    parts = []
    for e in entries[:3]:  # top 3 roles
        title   = e.get("title", "")
        company = e.get("company", "")
        desc    = e.get("description", "")[:150]
        if title:
            parts.append(f"{title} at {company}. {desc}")

    skills = cv.get("skills", [])
    if skills:
        parts.append(f"Skills: {', '.join(skills[:10])}")

    return " ".join(parts)


# ── EXPERIENCE YEARS FROM TEXT ────────────────────────────────────────────────

def extract_required_years(job_desc, job_edu):
    """
    Extract required years from job_description text since
    the experience field is empty for all jobs.
    """
    text = (job_desc + " " + job_edu).lower()

    # "fresher" or "no experience" = 0 years
    if any(w in text for w in ["fresher", "no experience", "not required", "experience not"]):
        return 0.0

    # "5+ years", "5-7 years", "at least 5 years", "minimum 5 years"
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

    return 0.0  # no requirement found = open to all


# ── SCORING FUNCTIONS ─────────────────────────────────────────────────────────

def skill_score(cv_skills, job_skills):
    cv_set = extract_skill_set(", ".join(cv_skills))
    if not job_skills:
        return 0.3  # penalize instead of neutral — empty skills = bad signal

    weighted_intersection = sum(skill_weight(s) for s in cv_set & job_skills)
    weighted_union        = sum(skill_weight(s) for s in cv_set | job_skills)
    jaccard = weighted_intersection / weighted_union if weighted_union else 0.0

    partial = sum(
        1 for cs in cv_set for js in job_skills
        if cs != js and (cs in js or js in cs)
    )
    partial_bonus = min(partial / max(len(job_skills), 1), 0.2)

    return min(jaccard + partial_bonus, 1.0)


def education_score(cv_education, job_edu_raw, job_title, job_desc):
    if not job_edu_raw or not job_edu_raw.strip():
        level_score = 0.7
    else:
        cv_level  = max((get_degree_level(e.get("degree", "")) for e in cv_education), default=0)
        job_level = get_degree_level(job_edu_raw)
        if job_level == 0:              level_score = 0.7
        elif cv_level >= job_level:     level_score = 1.0
        elif cv_level == job_level - 1: level_score = 0.5
        else:                           level_score = 0.1

    fos = field_of_study_score(cv_education, job_title, job_desc)
    return round(level_score * 0.6 + fos * 0.4, 4)


def experience_score(cv_years, required_years):
    if required_years == 0:              return 1.0
    if cv_years >= required_years:       return 1.0
    if cv_years >= required_years * 0.7: return 0.75
    if cv_years >= required_years * 0.4: return 0.4
    return 0.1


def semantic_score(cv_summary, job_desc, model):
    if not cv_summary or not job_desc:
        return 0.3  # penalize instead of neutral
    e1 = model.encode(cv_summary[:512], convert_to_tensor=True)
    e2 = model.encode(job_desc[:512],   convert_to_tensor=True)
    return max(0.0, float(util.cos_sim(e1, e2).item()))


def title_score(cv, job_title, model):
    if not job_title:
        return 0.5
    exp = cv.get("experience", [])
    entries = exp if isinstance(exp, list) else exp.get("entries", [])
    past_titles = [e.get("title", "") for e in entries if e.get("title", "").strip()]
    if not past_titles:
        return 0.4
    job_emb = model.encode(job_title, convert_to_tensor=True)
    scores  = [
        max(0.0, float(util.cos_sim(model.encode(t, convert_to_tensor=True), job_emb).item()))
        for t in past_titles if t.strip()
    ]
    return max(scores) if scores else 0.4


def skill_source_bonus(cv, job_skills):
    exp = cv.get("experience", [])
    entries = exp if isinstance(exp, list) else exp.get("entries", [])
    exp_skills = set()
    for e in entries:
        desc = e.get("description", "").lower()
        for skill in CORE_SKILLS:
            if re.search(r"\b" + re.escape(skill) + r"\b", desc):
                exp_skills.add(canonicalize(skill))
    if not job_skills or not exp_skills:
        return 0.3
    hits = exp_skills & job_skills
    return min(len(hits) / max(len(job_skills), 1), 1.0)


# ── MAIN SCORER ───────────────────────────────────────────────────────────────

def score_job(cv, job, model, cv_years, cv_level, cv_summary):
    job_title = job.get("job_title", "")
    job_desc  = job.get("job_description", "")
    job_edu   = job.get("education_requirements", "")

    job_skills     = get_job_skills(job)
    required_years = extract_required_years(job_desc, job_edu)
    job_level      = detect_seniority(job_title + " " + job_desc[:200])

    s_skill  = skill_score(cv.get("skills", []), job_skills)
    s_edu    = education_score(cv.get("education", []), job_edu, job_title, job_desc)
    s_exp    = experience_score(cv_years, required_years)
    s_sem    = semantic_score(cv_summary, job_desc, model)
    s_title  = title_score(cv, job_title, model)
    s_source = skill_source_bonus(cv, job_skills)

    raw = (
        s_skill  * WEIGHTS["skill"] +
        s_edu    * WEIGHTS["education"] +
        s_exp    * WEIGHTS["experience"] +
        s_sem    * WEIGHTS["semantic"] +
        s_title  * WEIGHTS["title"] +
        s_source * WEIGHTS["skill_source"]
    )

    penalty = seniority_penalty(cv_level, job_level)
    final   = raw * penalty * 100

    return {
        "job_id":       job.get("job_id"),
        "job_title":    job_title,
        "company":      job.get("company_name"),
        "location":     job.get("location"),
        "salary_range": job.get("salary_range"),
        "deadline":     job.get("deadline"),
        "final_score":  round(final, 2),
        "breakdown": {
            "skill_match":      round(s_skill  * 100, 1),
            "education_match":  round(s_edu    * 100, 1),
            "experience_match": round(s_exp    * 100, 1),
            "semantic_match":   round(s_sem    * 100, 1),
            "title_match":      round(s_title  * 100, 1),
            "skill_source":     round(s_source * 100, 1),
            "seniority_penalty":round(penalty,  2),
        }
    }


def match(cv_path, jobs_path, top_n=10, output_path=None):
    with open(cv_path, "r", encoding="utf-8") as f:
        cv = json.load(f)
    with open(jobs_path, "r", encoding="utf-8") as f:
        jobs = list(csv.DictReader(f))

    print(f"CV   : {cv.get('name')}")
    print(f"Jobs : {len(jobs)}")
    print(f"Loading Sentence-BERT model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Pre-compute CV-level values once
    cv_level, cv_years = cv_seniority(cv)
    cv_summary = synthesize_cv_summary(cv)

    level_labels = {0:"Intern", 1:"Junior", 2:"Mid-level", 3:"Senior", 4:"Lead", 5:"Director"}
    print(f"Candidate: {cv_years} yrs exp | Seniority: {level_labels.get(cv_level, 'Mid-level')}")
    print(f"Summary source: {'original' if cv.get('summary','').strip() else 'synthesized from experience'}")
    print(f"\nScoring {len(jobs)} jobs...")

    results = []
    for i, job in enumerate(jobs):
        results.append(score_job(cv, job, model, cv_years, cv_level, cv_summary))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(jobs)} scored...")

    results.sort(key=lambda x: x["final_score"], reverse=True)
    top_results = results[:top_n]

    print(f"\n{'='*65}")
    print(f"TOP {top_n} MATCHES FOR {cv.get('name','').upper()}")
    print(f"{'='*65}")
    for i, r in enumerate(top_results, 1):
        b = r["breakdown"]
        print(f"\n#{i} [{r['final_score']}%] {r['job_title']}")
        print(f"    Company : {r['company']}")
        print(f"    Location: {r['location']}")
        print(f"    Salary  : {r['salary_range']}")
        print(f"    Skill={b['skill_match']}% | Edu={b['education_match']}% | "
              f"Exp={b['experience_match']}% | Semantic={b['semantic_match']}% | "
              f"Title={b['title_match']}% | Penalty={b['seniority_penalty']}")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(top_results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved → {output_path}")

    return top_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV-Job Matcher v4")
    parser.add_argument("--cv",     required=True)
    parser.add_argument("--jobs",   required=True)
    parser.add_argument("--top",    type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    match(args.cv, args.jobs, args.top, args.output)