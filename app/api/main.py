from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.models.schemas import ChatRequest, ChatResponse, ScreeningResponse
from app.services.matcher import assign_dense_ranks, score_resume
from app.services.llm_service import improve_explanation
from app.services.parser import parse_resume
from app.services.rag import answer_question

app = FastAPI(title="Smart Resume Screening API", version="1.0.0")
resume_store: dict[str, str] = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/screen", response_model=ScreeningResponse)
async def screen_resumes(
    job_description: str = Form(..., min_length=10),
    resumes: list[UploadFile] = File(...),
) -> ScreeningResponse:
    results = []
    for upload in resumes:
        try:
            parsed = parse_resume(upload.filename or "resume", await upload.read())
            resume_id = str(uuid4())
            result = score_resume(job_description, parsed, resume_id)
            result.explanation = improve_explanation(job_description, parsed.text, result.explanation)
            resume_store[resume_id] = parsed.text
            results.append(result)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=422, detail=f"Could not process {upload.filename}: {error}") from error
    results.sort(key=lambda item: item.match_score, reverse=True)
    assign_dense_ranks(results)
    return ScreeningResponse(job_description=job_description, results=results)


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    text = resume_store.get(request.resume_id)
    if text is None:
        raise HTTPException(status_code=404, detail="Resume context not found. Screen the resume first.")
    answer, sources = answer_question(text, request.question)
    return ChatResponse(answer=answer, sources=sources)
