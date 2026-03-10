"""
Resume Parser — extracts and cleans text from PDF files.
Handles encoding issues, removes visual noise, limits context length.
"""
import re
import io
from typing import Optional

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

MAX_CHARS = 6000  # safe limit for LLM context (≈ 1500 tokens)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from PDF bytes using pdfplumber."""
    if not PDF_AVAILABLE:
        raise ImportError("pdfplumber not installed. Run: pip install pdfplumber")

    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return "\n".join(text_parts)


def clean_text(raw: str) -> str:
    """
    Normalize and clean extracted text:
    - collapse whitespace
    - remove control characters
    - strip repeated punctuation
    - limit to MAX_CHARS
    """
    if not raw:
        return ""

    # Normalize unicode whitespace
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Remove control characters except newlines and tabs
    text = re.sub(r"[^\S\n\t ]+", " ", text)

    # Collapse multiple blank lines → max two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    # Strip lines that are just punctuation/numbers (table borders, page numbers)
    lines = text.splitlines()
    clean_lines = [
        ln for ln in lines
        if len(re.sub(r"[\W\d]", "", ln).strip()) > 1  # keep lines with ≥2 real letters
        or ln.strip() == ""
    ]
    text = "\n".join(clean_lines)

    # Trim to limit
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n\n[... текст обрезан для экономии контекста ...]"

    return text.strip()


def parse_resume(
    file_bytes: Optional[bytes] = None,
    raw_text: Optional[str] = None,
) -> str:
    """
    Main entry point.
    Accepts either a PDF (bytes) or plain text fallback.
    Returns cleaned, LLM-ready string.
    """
    if file_bytes:
        try:
            raw = extract_text_from_pdf(file_bytes)
        except Exception as e:
            # Fallback: try raw text if PDF extraction fails
            if raw_text:
                return clean_text(raw_text)
            raise RuntimeError(f"PDF parsing failed: {e}") from e
    elif raw_text:
        raw = raw_text
    else:
        return ""

    return clean_text(raw)
