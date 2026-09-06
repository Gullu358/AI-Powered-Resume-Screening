import os


def _openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _chat(prompt: str) -> str | None:
    client = _openai_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def improve_explanation(job_description: str, resume_text: str, fallback: str) -> str:
    prompt = (
        "You are a careful resume screener. In exactly 2 concise sentences, explain the fit "
        "using only the supplied job description and resume. Do not invent facts.\n\n"
        f"JOB DESCRIPTION:\n{job_description[:6000]}\n\nRESUME:\n{resume_text[:8000]}"
    )
    return _chat(prompt) or fallback


def enhance_resume(job_description: str, resume_text: str, matched_skills: list[str], missing_skills: list[str]) -> str:
    fallback = (
        f"Prioritize evidence for {', '.join(missing_skills[:5]) or 'the role requirements'}. "
        f"Add measurable results and project details around {', '.join(matched_skills[:5]) or 'your strongest relevant skills'}. "
        "Place the most relevant skills near the top and mirror important job-description keywords naturally."
    )
    prompt = (
        "Act as a practical resume coach. In fewer than 100 words, tell the candidate exactly "
        "what to change to better fit the job description. Mention missing skills only as areas "
        "to address, never invent experience, and use clear actionable language.\n\n"
        f"JOB DESCRIPTION:\n{job_description[:6000]}\n\nRESUME:\n{resume_text[:8000]}\n\n"
        f"MATCHED SKILLS: {', '.join(matched_skills)}\nMISSING SKILLS: {', '.join(missing_skills)}"
    )
    return (_chat(prompt) or fallback)[:900]


def answer_with_openai(question: str, context: str) -> str | None:
    prompt = (
        "Answer the question only from the resume excerpts below. If the answer is not present, "
        "say that it is not mentioned in the resume. Keep the answer brief.\n\n"
        f"EXCERPTS:\n{context}\n\nQUESTION: {question}"
    )
    return _chat(prompt)
