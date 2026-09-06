import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    tesseract_cmd: str | None = os.getenv("TESSERACT_CMD")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))


settings = Settings()
