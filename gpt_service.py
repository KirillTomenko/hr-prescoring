"""
GPT Service — sends prompts to OpenAI models and parses structured JSON responses.
Supports multiple models for side-by-side comparison.
"""
import json
import re
import os
from typing import Optional

from openai import OpenAI

from models import GPTAssessment

# ──────────────────────────────────────────────────────────────────────────────
# Available models (displayed in UI)
# ──────────────────────────────────────────────────────────────────────────────
AVAILABLE_MODELS = {
    "GPT-4o mini  ⚡ (быстро/дёшево)": "gpt-4o-mini",
    "GPT-4o  🧠 (умнее/дороже)": "gpt-4o",
    "GPT-3.5 Turbo  💨 (классика)": "gpt-3.5-turbo",
}

SYSTEM_PROMPT = """
Ты — опытный HR-аналитик и технический рекрутер.
Твоя задача: оценить соответствие резюме кандидата описанию вакансии.

ПРАВИЛА:
1. Отвечай СТРОГО в формате JSON — никакого дополнительного текста, никаких markdown-блоков.
2. Оценивай объективно: не завышай и не занижай скор.
3. Используй русский язык для всех текстовых полей.
4. score — целое число от 0 до 100, где:
   0–49  = кандидат не подходит
   50–74 = частично подходит, есть пробелы
   75–100 = хороший/отличный кандидат
5. Каждый список должен содержать от 2 до 5 конкретных пунктов.

ОБЯЗАТЕЛЬНЫЙ ФОРМАТ ОТВЕТА (только JSON, без ```):
{
  "score": <int 0-100>,
  "strengths": ["сильная сторона 1", "сильная сторона 2"],
  "weaknesses": ["слабая сторона 1", "слабая сторона 2"],
  "missing_skills": ["пропущенный навык 1", "пропущенный навык 2"],
  "summary": "Краткий вывод 2-3 предложения о соответствии кандидата вакансии."
}
""".strip()


def _build_user_prompt(vacancy: str, resume: str) -> str:
    return f"""
## ВАКАНСИЯ:
{vacancy}

## РЕЗЮМЕ КАНДИДАТА:
{resume}

Оцени соответствие кандидата вакансии и верни результат строго в JSON.
""".strip()


def _repair_json(raw: str) -> str:
    """Best-effort cleanup of LLM JSON output."""
    # Remove markdown code fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Escape raw newlines inside string values
    # (replace literal \n inside JSON strings with \\n)
    def escape_newlines(m):
        return m.group(0).replace("\n", "\\n")

    raw = re.sub(r'"(?:[^"\\]|\\.)*"', escape_newlines, raw, flags=re.DOTALL)

    # Remove trailing commas before } or ]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    return raw


def _parse_response(raw_text: str, model_name: str) -> GPTAssessment:
    """Parse raw model output into GPTAssessment."""
    repaired = _repair_json(raw_text)

    try:
        data = json.loads(repaired)
    except json.JSONDecodeError:
        # Last resort: try to extract JSON with regex
        match = re.search(r"\{.*\}", repaired, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            raise ValueError(f"Could not parse JSON from model response:\n{raw_text}")

    return GPTAssessment(
        score=int(data.get("score", 0)),
        strengths=data.get("strengths", []),
        weaknesses=data.get("weaknesses", []),
        missing_skills=data.get("missing_skills", []),
        summary=data.get("summary", ""),
        model_name=model_name,
        raw_response=raw_text,
    )


def assess_candidate(
    vacancy_text: str,
    resume_text: str,
    model_key: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.2,
) -> GPTAssessment:
    """
    Send vacancy + resume to a single OpenAI model and return structured assessment.

    Args:
        vacancy_text: Job description.
        resume_text: Cleaned resume text.
        model_key: OpenAI model identifier (e.g. 'gpt-4o-mini').
        api_key: OpenAI API key (falls back to OPENAI_API_KEY env var).
        temperature: LLM temperature (low = more deterministic).
    """
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    user_prompt = _build_user_prompt(vacancy_text, resume_text)

    response = client.chat.completions.create(
        model=model_key,
        temperature=temperature,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_text = response.choices[0].message.content or ""
    return _parse_response(raw_text, model_name=model_key)


def compare_models(
    vacancy_text: str,
    resume_text: str,
    model_keys: list[str],
    api_key: Optional[str] = None,
    temperature: float = 0.2,
) -> dict[str, GPTAssessment]:
    """
    Run the same assessment through multiple models.
    Returns dict: model_key → GPTAssessment.
    """
    results: dict[str, GPTAssessment] = {}
    for model_key in model_keys:
        try:
            results[model_key] = assess_candidate(
                vacancy_text=vacancy_text,
                resume_text=resume_text,
                model_key=model_key,
                api_key=api_key,
                temperature=temperature,
            )
        except Exception as e:
            # Store error placeholder so other models still run
            results[model_key] = GPTAssessment(
                score=0,
                strengths=[],
                weaknesses=[],
                missing_skills=[],
                summary=f"❌ Ошибка модели {model_key}: {e}",
                model_name=model_key,
            )
    return results
