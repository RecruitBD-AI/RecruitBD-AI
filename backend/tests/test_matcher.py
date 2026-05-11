from services.matcher.cv_matcher import (
    skill_weight,
    experience_score,
    get_degree_level,
    field_of_study_score,
    skill_score,
    education_score,
    extract_skill_set,
    _normalize_cv_skills,
    _build_unified_cv_skills,
    synthesize_cv_summary,
    seniority_penalty,
    nature_score,
    workplace_score,
    extract_skills_from_experience,
    skill_source_score,
)
from services.matcher.constants import canonicalize, extract_known_skills_from_text


# ── CANONICALIZATION ──────────────────────────────────────────────────────────


def test_skill_weight():
    """Test that core, soft, and unknown skills get the correct weights."""
    assert skill_weight("python") == 1.5
    assert skill_weight("react") == 1.5

    # Synonym variants should also resolve to core-skill weight
    assert skill_weight("nodejs") == 1.5  # canonicalizes to "javascript"
    assert skill_weight("cpp") == 1.5  # canonicalizes to "c++"

    # Soft skills
    assert skill_weight("communication") == 0.3

    # Unknown/Standard skills
    assert skill_weight("obscure_framework_123") == 1.0


def test_canonicalize():
    """Test that synonym canonicalization uses preferred forms."""
    assert canonicalize("nodejs") == "javascript"
    assert canonicalize("dart") == "flutter"
    assert canonicalize("containerization") == "docker"
    assert canonicalize("spring") == "spring boot"
    assert canonicalize("mysql") == "sql"
    assert canonicalize("postgresql") == "sql"

    # Canonical form maps to itself
    assert canonicalize("python") == "python"
    assert canonicalize("react") == "react"

    # Unknown skill passes through unchanged
    assert canonicalize("some_random_skill") == "some random skill"


def test_extract_skill_set():
    """Test skill extraction and canonicalization from delimited text."""
    result = extract_skill_set("Python, React.js, Node.js, Docker")
    assert "python" in result
    assert "react" in result  # react.js → react
    assert "javascript" in result  # node.js → javascript
    assert "docker" in result

    # Single-char items should be filtered out
    result2 = extract_skill_set("A, Python, B")
    assert "python" in result2
    assert "a" not in result2
    assert "b" not in result2

    # Empty input
    assert extract_skill_set("") == set()
    assert extract_skill_set(None) == set()


def test_extract_known_skills_from_text():
    """Test NLP skill extraction from free-form description text."""
    text = "Built REST APIs using Python and Django with PostgreSQL database"
    found = extract_known_skills_from_text(text)
    assert "python" in found
    assert "django" in found
    assert "rest api" in found
    assert "sql" in found  # postgresql → sql


# ── NORMALIZE CV SKILLS ───────────────────────────────────────────────────────


def test_normalize_cv_skills_strings():
    """Test that string skills pass through."""
    result = _normalize_cv_skills(["Python", "React", "Docker"])
    assert result == ["Python", "React", "Docker"]


def test_normalize_cv_skills_dicts():
    """Test that dict skills are extracted by common key names."""
    result = _normalize_cv_skills(
        [
            {"name": "Python"},
            {"skill": "React"},
            {"title": "Docker"},
        ]
    )
    assert result == ["Python", "React", "Docker"]


def test_normalize_cv_skills_mixed():
    """Test that None, ints, and dicts are handled safely."""
    result = _normalize_cv_skills([None, 42, "Python", {"name": "React"}, [1, 2]])
    assert "Python" in result
    assert "React" in result
    assert 42 not in result  # ints should be skipped


# ── EDUCATION SCORING ─────────────────────────────────────────────────────────


def test_get_degree_level():
    """Test that education text is properly converted to level integers."""
    assert get_degree_level("B.Sc in Computer Science") == 4
    assert get_degree_level("MSc in Data Science") == 5
    assert get_degree_level("HSC") == 2
    assert get_degree_level("Secondary School Certificate") == 1
    assert get_degree_level("PhD in Machine Learning") == 6
    assert get_degree_level("Diploma in IT") == 3
    assert get_degree_level("Some unknown text") == 0


def test_field_of_study_cse_vs_eee():
    """CSE should score higher than EEE for a Software Developer job."""
    cse_edu = [{"degree": "B.Sc in Computer Science and Engineering", "institution": "AIUB"}]
    eee_edu = [{"degree": "B.Sc in Electrical and Electronic Engineering", "institution": "AIUB"}]
    bba_edu = [{"degree": "BBA in Marketing", "institution": "DU"}]

    job_title = "Software Developer"
    job_desc = "We are looking for a software developer to join our team."

    cse_score = field_of_study_score(cse_edu, job_title, job_desc)
    eee_score = field_of_study_score(eee_edu, job_title, job_desc)
    bba_score = field_of_study_score(bba_edu, job_title, job_desc)

    # CSE is primary → 1.0, EEE is adjacent → 0.7, BBA is unrelated → 0.4
    assert cse_score == 1.0
    assert eee_score == 0.7
    assert bba_score == 0.4
    assert cse_score > eee_score > bba_score


def test_field_of_study_eee_for_electrical_job():
    """EEE should score highest for an Electrical Maintenance job."""
    cse_edu = [{"degree": "B.Sc in Computer Science and Engineering", "institution": "AIUB"}]
    eee_edu = [{"degree": "B.Sc in Electrical and Electronic Engineering", "institution": "AIUB"}]

    # Use "Electrical Maintenance" (not "Electrical Engineer") to trigger
    # the 'electrical' keyword specifically, not the broader 'engineer' keyword.
    job_title = "Electrical Maintenance Technician"
    job_desc = "Maintain electrical systems and wiring."

    eee_score = field_of_study_score(eee_edu, job_title, job_desc)
    cse_score = field_of_study_score(cse_edu, job_title, job_desc)

    assert eee_score == 1.0  # primary (electrical/electronics)
    assert cse_score == 0.7  # adjacent (computer science)


def test_education_score_level_matching():
    """Test combined degree-level + field-of-study scoring."""
    bsc_cse = [{"degree": "Bachelor of Computer Science", "institution": "AIUB"}]
    hsc = [{"degree": "Higher Secondary Certificate", "institution": "Some College"}]

    # BSc holder applying to job requiring BSc → full level match
    score_bsc = education_score(bsc_cse, "Bachelor degree required", "Software Developer", "")
    assert score_bsc > 0.8

    # HSC holder applying to job requiring BSc → penalized
    score_hsc = education_score(hsc, "Bachelor degree required", "Software Developer", "")
    assert score_hsc < score_bsc


# ── SKILL SCORING ─────────────────────────────────────────────────────────────


def test_skill_score_full_coverage():
    """CV covers all job skills → should score high."""
    cv_skills = {"python", "django", "sql", "docker"}
    job_skills = {"python", "django", "sql"}
    score = skill_score(cv_skills, job_skills)
    assert score > 0.9


def test_skill_score_partial_coverage():
    """CV covers some job skills → intermediate score."""
    cv_skills = {"python", "react"}
    job_skills = {"python", "django", "sql", "docker"}
    score = skill_score(cv_skills, job_skills)
    assert 0.2 < score < 0.7


def test_skill_score_zero_overlap():
    """No skill overlap at all → should return 0."""
    cv_skills = {"photoshop", "illustrator"}
    job_skills = {"python", "django", "sql"}
    score = skill_score(cv_skills, job_skills)
    assert score == 0.0


def test_skill_score_partial_string_match():
    """'react' in CV should get partial bonus for 'react native' in job (but not double-count)."""
    cv_skills = {"react", "javascript"}
    job_skills = {"react native", "javascript"}

    score = skill_score(cv_skills, job_skills)
    # javascript matches exactly, react partially matches react native
    assert score > 0.5


def test_skill_score_no_job_skills():
    """When job has no structured skills, fallback inference should still work."""
    cv_skills = {"python", "django"}
    score = skill_score(cv_skills, set(), job_text="Python Django web developer needed")
    assert score > 0.0  # Should infer something, not crash


# ── SENIORITY ─────────────────────────────────────────────────────────────────


def test_seniority_penalty():
    """Test that seniority gap reduces scores appropriately."""
    assert seniority_penalty(2, 2) == 1.0  # exact match
    assert seniority_penalty(2, 3) == 0.85  # 1 level gap
    assert seniority_penalty(0, 2) == 0.65  # 2 level gap
    assert seniority_penalty(0, 4) == 0.40  # 4 level gap (harsh)


# ── EXPERIENCE SCORING ────────────────────────────────────────────────────────


def test_experience_score():
    """Test the experience penalty/bonus logic."""
    assert experience_score(cv_years=2, required_years=0) == 1.0
    assert experience_score(cv_years=5, required_years=5) == 1.0
    assert experience_score(cv_years=7, required_years=5) == 1.0
    assert experience_score(cv_years=4, required_years=5) == 0.75
    assert experience_score(cv_years=2, required_years=5) == 0.40
    assert experience_score(cv_years=1, required_years=5) == 0.10


# ── NATURE & WORKPLACE ────────────────────────────────────────────────────────


def test_nature_score_internship():
    """Interns/freshers should score high for internship jobs."""
    assert nature_score(cv_level=0, cv_years=0, job_nature="Intern") == 1.0
    assert nature_score(cv_level=3, cv_years=5, job_nature="Intern") == 0.3


def test_nature_score_fulltime():
    """Full-time jobs should slightly penalize candidates with 0 experience."""
    assert nature_score(cv_level=2, cv_years=3, job_nature="Full Time") == 1.0
    assert nature_score(cv_level=0, cv_years=0, job_nature="Full Time") == 0.85


def test_workplace_score_remote():
    """Remote jobs should score well for everyone."""
    score = workplace_score("Dhaka", "Dhaka", "Work From Home")
    assert score == 0.9


def test_workplace_score_office_location_match():
    """Office job in same city should score 1.0."""
    score = workplace_score("Dhaka", "Dhaka", "Work at Office")
    assert score == 1.0


def test_workplace_score_office_location_mismatch():
    """Office job in different city should penalize."""
    score = workplace_score("Sylhet", "Dhaka", "Work at Office")
    assert score == 0.55


# ── SKILL SOURCE ──────────────────────────────────────────────────────────────


def test_skill_source_score():
    """Skills proven in experience should score higher."""
    exp_skills = {"python", "django", "sql"}
    job_skills = {"python", "django", "react", "sql"}
    score = skill_source_score(exp_skills, job_skills)
    assert score == 0.75  # 3/4 hits

    # No experience skills → fallback
    assert skill_source_score(set(), job_skills) == 0.3


# ── EXPERIENCE SKILL EXTRACTION ──────────────────────────────────────────────


def test_extract_skills_from_experience():
    """Should extract skills from both description text and tech arrays."""
    cv = {
        "experience": [
            {
                "title": "Backend Developer",
                "description": "Built REST APIs using Python and Django",
                "tech": ["Docker", "PostgreSQL"],
            }
        ]
    }
    skills = extract_skills_from_experience(cv)
    assert "python" in skills
    assert "django" in skills
    assert "rest api" in skills
    assert "docker" in skills
    assert "sql" in skills  # postgresql → sql via canonicalization


# ── UNIFIED CV SKILLS ─────────────────────────────────────────────────────────


def test_build_unified_cv_skills():
    """Should merge top-level skills, experience tech, and NLP-extracted skills."""
    cv = {
        "skills": ["React", "TypeScript"],
        "experience": [
            {
                "title": "Dev",
                "description": "Used Python and Django for backend",
                "tech": ["Docker"],
            }
        ],
    }
    exp_skills = extract_skills_from_experience(cv)
    unified = _build_unified_cv_skills(cv, exp_skills)

    assert "react" in unified  # from top-level skills
    assert "typescript" in unified  # from top-level skills
    assert "docker" in unified  # from tech array
    assert "python" in unified  # from description NLP
    assert "django" in unified  # from description NLP


# ── CV SUMMARY ────────────────────────────────────────────────────────────────


def test_synthesize_cv_summary_fresh_graduate():
    """Fresh graduates with no experience should still produce a rich summary."""
    cv = {
        "summary": "Aspiring CSE student",
        "experience": [],
        "education": [
            {"degree": "B.Sc in Computer Science", "institution": "AIUB"},
        ],
        "projects": ["Sign Language Recognition using YOLOv11"],
        "skills": ["Python", "Java", "SQL"],
    }
    summary = synthesize_cv_summary(cv)
    assert "Aspiring CSE student" in summary
    assert "B.Sc in Computer Science" in summary
    assert "Sign Language Recognition" in summary
    assert "Python" in summary


def test_synthesize_cv_summary_experienced():
    """Experienced candidates should include job titles and companies."""
    cv = {
        "summary": "",
        "experience": [
            {"title": "Software Engineer", "company": "Google", "description": "Built APIs"},
        ],
        "education": [{"degree": "MSc in CS", "institution": "MIT"}],
        "skills": ["Go", "Kubernetes"],
    }
    summary = synthesize_cv_summary(cv)
    assert "Software Engineer" in summary
    assert "Google" in summary
    assert "MSc in CS" in summary
