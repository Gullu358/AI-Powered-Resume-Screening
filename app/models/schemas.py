from pydantic import BaseModel, Field


class ResumeResult(BaseModel):
    resume_id: str
    filename: str
    rank: int = Field(default=0, ge=0)
    match_score: float = Field(ge=0, le=100)
    cosine_similarity: float = Field(ge=0, le=1)
    skill_coverage: float = Field(ge=0, le=1)
    total_skills: int = Field(default=0, ge=0)
    total_required_skills: int = Field(default=0, ge=0)
    matched_skills: list[str]
    missing_skills: list[str]
    matched_keywords: list[str]
    extracted_skills: list[str]
    experience: str | None = None
    explanation: str


class ScreeningResponse(BaseModel):
    job_description: str
    results: list[ResumeResult]


class ChatRequest(BaseModel):
    resume_id: str
    question: str = Field(min_length=2, max_length=1000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
