import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.llm_service import answer_with_openai


def _chunks(text: str, size: int = 700, overlap: int = 100) -> list[str]:
    words = text.split()
    return [" ".join(words[start:start + size]) for start in range(0, len(words), max(1, size - overlap))]


def answer_question(text: str, question: str, top_k: int = 3) -> tuple[str, list[str]]:
    chunks = [chunk for chunk in _chunks(text) if chunk.strip()]
    if not chunks:
        return "No readable resume context is available.", []
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        chunk_matrix = vectorizer.fit_transform(chunks)
        question_vector = vectorizer.transform([question])
        scores = cosine_similarity(question_vector, chunk_matrix).ravel()
        selected = [index for index in scores.argsort()[::-1][:top_k] if scores[index] > 0]
    except ValueError:
        selected = list(range(min(top_k, len(chunks))))
    context = "\n\n".join(chunks[index] for index in selected) or chunks[0]
    answer = answer_with_openai(question, context)
    if answer:
        return answer, [f"Resume excerpt {index + 1}" for index in selected] or ["Resume excerpt 1"]
    question_terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]+", question.lower()))
    sentences = re.split(r"(?<=[.!?])\s+", context)
    relevant = [sentence for sentence in sentences if question_terms.intersection(set(sentence.lower().split()))]
    return (" ".join(relevant[:2]) or context[:500]), ["Resume text"]
