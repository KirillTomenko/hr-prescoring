"""
Data models for HR Pre-Scoring System.
Defines structures for candidate assessments, history, and model comparisons.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class GPTAssessment:
    """Structured response from AI model.

    JSON contract (must match SYSTEM_PROMPT in gpt_service.py):
        score          – int 0-100
        strengths      – list[str]  сильные стороны
        weaknesses     – list[str]  слабые стороны
        missing_skills – list[str]  пропущенные навыки
        summary        – str        итоговый вывод
    """
    score: int                          # 0–100
    strengths: list[str]
    weaknesses: list[str]
    missing_skills: list[str]
    summary: str
    model_name: str = ""
    raw_response: str = ""


@dataclass
class ScoringWeights:
    """Weights for composite scoring (must sum to 1.0)."""
    hard_skills: float = 0.60
    experience: float = 0.25
    soft_skills: float = 0.15

    def validate(self) -> bool:
        total = round(self.hard_skills + self.experience + self.soft_skills, 2)
        return total == 1.0

    def as_dict(self) -> dict:
        return {
            "hard_skills": self.hard_skills,
            "experience": self.experience,
            "soft_skills": self.soft_skills,
        }


@dataclass
class HeuristicScores:
    """Keyword-based heuristic sub-scores."""
    hard_skills: float = 0.0
    experience: float = 0.0
    soft_skills: float = 0.0
    composite: float = 0.0


@dataclass
class CandidateResult:
    """Full assessment result for one candidate."""
    candidate_name: str
    vacancy_title: str
    gpt_assessment: GPTAssessment
    heuristic_scores: HeuristicScores
    weights: ScoringWeights
    final_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    model_name: str = ""

    def verdict(self) -> str:
        if self.final_score >= 75:
            return "✅ Отличный кандидат"
        elif self.final_score >= 50:
            return "⚠️ Частично подходит"
        else:
            return "❌ Не подходит"

    def verdict_color(self) -> str:
        if self.final_score >= 75:
            return "green"
        elif self.final_score >= 50:
            return "orange"
        return "red"

    def to_history_row(self) -> dict:
        return {
            "Кандидат": self.candidate_name,
            "Вакансия": self.vacancy_title,
            "Модель": self.model_name,
            "Итог": round(self.final_score, 1),
            "GPT-оценка": self.gpt_assessment.score,
            "Hard Skills": round(self.heuristic_scores.hard_skills, 1),
            "Опыт": round(self.heuristic_scores.experience, 1),
            "Soft Skills": round(self.heuristic_scores.soft_skills, 1),
            "Вердикт": self.verdict(),
            "Время": self.timestamp,
        }


@dataclass
class ModelComparisonResult:
    """Side-by-side comparison of assessments from multiple models."""
    candidate_name: str
    vacancy_title: str
    assessments: dict[str, GPTAssessment] = field(default_factory=dict)  # model_name -> assessment
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
