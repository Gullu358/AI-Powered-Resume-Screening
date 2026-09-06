import re
from dataclasses import asdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.schemas import ResumeResult
from app.services.parser import ParsedResume, SKILL_VOCABULARY


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "for", "from", "in", "is",
    "of", "on", "or", "that", "the", "to", "with", "will", "you", "your",
}


def _jd_terms(job_description: str) -> list[str]:
    terms = re.findall(r"[a-zA-Z][a-zA-Z+#.-]{1,}", job_description.lower())
    return list(dict.fromkeys(term for term in terms if term not in STOP_WORDS and len(term) > 2))


def _cosine(job_description: str, resume_text: str) -> float:
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform([job_description, resume_text])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except ValueError:
        return 0.0


def _explanation(score: float, matched: list[str], missing: list[str], cosine: float) -> str:
    strength = "strong" if score >= 75 else "moderate" if score >= 50 else "limited"
    matched_text = ", ".join(matched[:4]) or "no required skills"
    missing_text = ", ".join(missing[:4]) or "none of the listed skills"
    return (
        f"This resume shows a {strength} fit, with evidence of {matched_text}. "
        f"Text similarity is {cosine:.0%}; the main gaps are {missing_text}."
    )


def score_resume(job_description: str, resume: ParsedResume, resume_id: str) -> ResumeResult:
    jd_lower = job_description.lower()
    required_skills = [skill for skill in SKILL_VOCABULARY if re.search(r"(?<!\w)" + re.escape(skill) + r"(?!\w)", jd_lower)]
    matched_skills = [skill for skill in required_skills if skill in resume.skills]
    missing_skills = [skill for skill in required_skills if skill not in resume.skills]
    coverage = len(matched_skills) / len(required_skills) if required_skills else 0.0
    cosine = _cosine(job_description, resume.text)
    score = round((coverage * 0.45 + cosine * 0.55) * 100, 2)
    resume_terms = set(_jd_terms(resume.text))
    matched_keywords = [term for term in _jd_terms(job_description) if term in resume_terms]
    return ResumeResult(
        resume_id=resume_id,
        filename=resume.filename,
        total_skills=len(resume.skills),
        total_required_skills=len(required_skills),
        match_score=score,
        cosine_similarity=round(cosine, 4),
        skill_coverage=round(coverage, 4),
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        matched_keywords=matched_keywords,
        extracted_skills=resume.skills,
        experience=resume.experience,
        explanation=_explanation(score, matched_skills, missing_skills, cosine),
    )


def assign_dense_ranks(results: list[ResumeResult]) -> list[ResumeResult]:
    current_score: float | None = None
    current_rank = 0
    for result in results:
        if current_score is None or result.match_score != current_score:
            current_rank += 1
            current_score = result.match_score
        result.rank = current_rank
    return results
