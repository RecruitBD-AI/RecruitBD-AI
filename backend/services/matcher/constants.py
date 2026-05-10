"""
constants.py — Shared constants and utilities for the matcher package
----------------------------------------------------------------------
Single source of truth for CORE_SKILLS, SENIORITY_MAP, SKILL_SYNONYMS,
and shared helper functions used by both build_index.py and cv_matcher.py.
"""

import re

# ── SKILL SYNONYMS ────────────────────────────────────────────────────────────
# Each group maps variants to a single canonical form.
# The *first* element of each tuple is the preferred canonical name.
# Groups should only contain true synonyms — not loosely related skills.

SKILL_SYNONYMS = [
    ("javascript", {"js", "node.js", "nodejs"}),
    ("typescript", {"ts"}),
    ("python", {"py"}),
    ("java", {"core java"}),
    ("c++", {"cpp"}),
    ("c#", {"csharp"}),
    ("machine learning", {"ml"}),
    ("deep learning", {"dl"}),
    ("natural language processing", {"nlp"}),
    ("computer vision", {"cv"}),
    ("artificial intelligence", {"ai"}),
    ("sql", {"mysql", "postgresql", "postgres", "oracle sql"}),
    ("mongodb", {"mongo"}),
    ("react", {"react.js", "reactjs"}),
    ("vue", {"vue.js", "vuejs"}),
    ("angular", {"angularjs"}),
    ("flutter", {"dart"}),
    ("spring boot", {"spring"}),
    ("git", {"github", "gitlab", "bitbucket"}),
    ("docker", {"containerization"}),
    ("kubernetes", {"k8s"}),
    ("aws", {"amazon web services"}),
    ("gcp", {"google cloud", "google cloud platform"}),
    ("azure", {"microsoft azure"}),
    ("rest api", {"rest", "restful"}),
    ("graphql", {"gql"}),
    ("ci cd", {"cicd", "continuous integration", "continuous delivery"}),
    ("microservices", {"microservice architecture"}),
    ("oop", {"object oriented programming"}),
    ("dsa", {"data structures and algorithms"}),
    ("test automation", {"unit testing", "automated testing"}),
    ("pytest", {"py test"}),
    ("pandas", {"python pandas"}),
    ("numpy", {"python numpy"}),
    ("power bi", {"powerbi", "bi dashboard"}),
    ("tableau", {"tableau bi"}),
    ("excel", {"ms excel", "microsoft excel"}),
    ("firebase", {"firestore"}),
    ("adobe premiere pro", {"premiere pro", "premiere"}),
    ("adobe photoshop", {"photoshop"}),
    ("adobe illustrator", {"illustrator"}),
    ("after effects", {"adobe after effects"}),
    # UI/UX is a discipline — Figma is a tool. Separated.
    ("ui ux", {"ux ui", "user interface", "user experience"}),
    ("seo", {"search engine optimization"}),
    ("sem", {"search engine marketing"}),
    ("digital marketing", {"social media marketing", "smm"}),
    ("content writing", {"copywriting"}),
    # "bd" removed — ambiguous in Bangladesh context
    ("sales", {"business development"}),
    ("customer support", {"customer service"}),
    ("crm", {"customer relationship management"}),
    ("accounting", {"bookkeeping", "accounts"}),
    # HR and recruitment are related but distinct roles — split
    ("recruitment", {"talent acquisition"}),
    ("project management", {"agile", "scrum"}),
    ("qa", {"quality assurance", "manual testing"}),
    ("electrical", {"electrical maintenance"}),
    ("autocad", {"cad"}),
    ("nursing", {"patient care"}),
    ("teaching", {"tutoring", "home tutor"}),
    ("video editing", {"video editor"}),
    ("graphic design", {"graphics design"}),
    ("wordpress", {"wp"}),
    ("shopify", {"shopify development"}),
    ("laravel", {"php laravel"}),
    ("react native", {"rn"}),
    ("kotlin", {"android kotlin"}),
    ("swift", {"ios swift"}),
    ("redis", {"redis cache"}),
    ("elasticsearch", {"elastic search"}),
    ("fastapi", {"python fastapi"}),
    ("flask", {"python flask"}),
    ("django", {"python django"}),
    ("next.js", {"nextjs"}),
    ("express", {"express.js", "expressjs"}),
    ("html", {"html5"}),
    ("css", {"css3", "scss"}),
    ("linux", {"ubuntu", "unix"}),
]

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
    "graphql",
    "microservices",
    "ci cd",
    "test automation",
    "pytest",
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
    "pandas",
    "numpy",
    "power bi",
    "tableau",
    "go",
    "kotlin",
    "swift",
    "php",
    "ruby",
    "scala",
    "redis",
    "firebase",
    "firestore",
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
    "react native",
    "wordpress",
    "shopify",
    "laravel",
    "postgres",
    "mysql",
    "json",
    "protobuf",
    "oop",
    "dsa",
    "excel",
    "project management",
    "agile",
    "scrum",
    "qa",
    "seo",
    "sem",
    "digital marketing",
    "smm",
    "content writing",
    "copywriting",
    "sales",
    "business development",
    "customer support",
    "customer service",
    "crm",
    "accounting",
    "bookkeeping",
    "recruitment",
    "hr",
    "graphic design",
    "video editing",
    "adobe photoshop",
    "adobe illustrator",
    "adobe premiere pro",
    "after effects",
    "figma",
    "ui ux",
    "autocad",
    "electrical",
    "nursing",
    "patient care",
    "teaching",
    "tutoring",
}

SOFT_SKILLS = {
    "leadership",
    "communication",
    "teamwork",
    "collaboration",
    "time management",
    "critical thinking",
    "active listening",
    "problem solving",
    "adaptability",
    "presentation",
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


# ── PRE-COMPILED REGEX PATTERNS ──────────────────────────────────────────────
# Build once at import time — eliminates repeated re.compile() in hot loops.


def _build_word_pattern(term: str) -> re.Pattern:
    """Compile a word-boundary pattern for a skill/keyword."""
    return re.compile(r"\b" + re.escape(term.lower()) + r"\b")


# Patterns for every core skill
CORE_SKILL_PATTERNS: dict[str, re.Pattern] = {
    skill: _build_word_pattern(skill) for skill in CORE_SKILLS
}

# Patterns for every synonym variant
SYNONYM_PATTERNS: dict[str, re.Pattern] = {}
for _canonical, _variants in SKILL_SYNONYMS:
    for _variant in _variants:
        if _variant not in SYNONYM_PATTERNS:
            SYNONYM_PATTERNS[_variant] = _build_word_pattern(_variant)
    # Also add the canonical itself so it can be detected in free text
    if _canonical not in SYNONYM_PATTERNS:
        SYNONYM_PATTERNS[_canonical] = _build_word_pattern(_canonical)

# Patterns for seniority keywords
SENIORITY_PATTERNS: dict[str, re.Pattern] = {
    kw: _build_word_pattern(kw) for kw in SENIORITY_MAP
}


def build_synonym_map() -> dict[str, str]:
    """Build variant → canonical mapping using explicit canonical forms."""
    mapping: dict[str, str] = {}
    for canonical, variants in SKILL_SYNONYMS:
        # The canonical form maps to itself
        mapping[canonical] = canonical
        for variant in variants:
            mapping[variant] = canonical
    return mapping


SYNONYM_MAP = build_synonym_map()


# ── SHARED UTILITY FUNCTIONS ─────────────────────────────────────────────────
# Used by both cv_matcher.py and build_index.py — defined here to avoid duplication.


def normalize_skill(s: str) -> str:
    """Normalize a skill string: lowercase, strip non-alphanum (keep +#.)."""
    return re.sub(r"[^a-z0-9\+\#\.]", " ", s.lower()).strip()


def canonicalize(skill: str) -> str:
    """Normalize and map a skill string to its canonical form."""
    s = normalize_skill(skill)
    return SYNONYM_MAP.get(s, s)


def detect_seniority(text: str) -> int:
    """Detect seniority level from text using keyword patterns.
    Returns 0-5 scale, defaults to 2 (mid-level) if nothing detected.
    """
    text_lower = text.lower()
    best = -1
    for kw, lvl in SENIORITY_MAP.items():
        if SENIORITY_PATTERNS[kw].search(text_lower):
            best = max(best, lvl)
    return best if best >= 0 else 2


def extract_known_skills_from_text(text: str) -> set[str]:
    """Extract known skills from free-form text using pre-compiled patterns."""
    if not text:
        return set()
    text_lower = text.lower()
    found: set[str] = set()
    for skill, pattern in CORE_SKILL_PATTERNS.items():
        if pattern.search(text_lower):
            found.add(canonicalize(skill))
    for variant, pattern in SYNONYM_PATTERNS.items():
        if pattern.search(text_lower):
            found.add(SYNONYM_MAP.get(variant, variant))
    return found
