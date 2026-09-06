import io
import re
from dataclasses import dataclass

from app.core.config import settings


@dataclass
class ParsedResume:
    filename: str
    text: str
    skills: list[str]
    experience: str | None


SKILL_VOCABULARY = [
    "python", "java", "javascript", "typescript", "sql", "nosql", "mongodb",
    "postgresql", "mysql", "fastapi", "flask", "django", "streamlit", "react",
    "html", "css", "git", "docker", "kubernetes", "aws", "azure", "gcp",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "machine learning", "deep learning", "nlp", "computer vision", "llm",
    "generative ai", "langchain", "rag", "rest api", "graphql", "spark",
    "power bi", "tableau", "excel", "linux", "fastapi", "api testing",
]


def _extract_skills(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.lower())
    return [skill for skill in SKILL_VOCABULARY if re.search(r"(?<!\w)" + re.escape(skill) + r"(?!\w)", normalized)]


def _extract_experience(text: str) -> str | None:
    patterns = [
        r"(?:over\s+|more\s+than\s+)?(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)?",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"{match.group(1)} years"
    return None


def extract_text(filename: str, content: bytes) -> str:
    extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if extension in {"txt", "md"}:
        return content.decode("utf-8", errors="ignore")
    if extension == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if extension == "docx":
        from docx import Document

        document = Document(io.BytesIO(content))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    if extension in {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}:
        from PIL import Image
        import pytesseract

        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        return pytesseract.image_to_string(Image.open(io.BytesIO(content)))
    raise ValueError(f"Unsupported resume format: .{extension or 'unknown'}")


def parse_resume(filename: str, content: bytes) -> ParsedResume:
    text = extract_text(filename, content).strip()
    if not text:
        raise ValueError(f"No readable text found in {filename}")
    return ParsedResume(filename=filename, text=text, skills=_extract_skills(text), experience=_extract_experience(text))
