from app.models.schemas import ResumeResult
from app.services.matcher import assign_dense_ranks, score_resume
from app.services.parser import parse_resume
from app.services.rag import answer_question


def test_parse_text_resume_extracts_skills_and_experience() -> None:
    resume = parse_resume("candidate.txt", b"Python developer with 3 years of experience using FastAPI, SQL and Docker.")

    assert resume.skills == ["python", "sql", "fastapi", "docker"]
    assert resume.experience == "3 years"


def test_matcher_returns_high_score_for_relevant_resume() -> None:
    resume = parse_resume("candidate.txt", b"Python, FastAPI, SQL, Docker and machine learning. 3 years experience.")

    result = score_resume("Python FastAPI SQL Docker machine learning", resume, "candidate-1")

    assert result.match_score >= 80
    assert set(result.matched_skills) == {"python", "sql", "fastapi", "docker", "machine learning"}
    assert result.missing_skills == []
    assert result.total_skills == 5
    assert result.total_required_skills == 5


def test_rag_fallback_returns_resume_evidence_without_ollama() -> None:
    answer, sources = answer_question("Python developer. Built FastAPI services for three years.", "Which framework was used?")

    assert "FastAPI" in answer
    assert sources


def test_dense_ranking_assigns_same_rank_to_ties() -> None:
    results = [
        ResumeResult(resume_id="a", filename="a.txt", match_score=90, cosine_similarity=1, skill_coverage=1, matched_skills=[], missing_skills=[], matched_keywords=[], extracted_skills=[], explanation=""),
        ResumeResult(resume_id="b", filename="b.txt", match_score=90, cosine_similarity=1, skill_coverage=1, matched_skills=[], missing_skills=[], matched_keywords=[], extracted_skills=[], explanation=""),
        ResumeResult(resume_id="c", filename="c.txt", match_score=70, cosine_similarity=1, skill_coverage=1, matched_skills=[], missing_skills=[], matched_keywords=[], extracted_skills=[], explanation=""),
    ]

    assign_dense_ranks(results)

    assert [result.rank for result in results] == [1, 1, 2]
