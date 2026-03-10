"""
Heuristic Scoring Engine
Keyword-based analysis that runs locally (no API calls).
Complements GPT scores with an objective cross-check.
"""
import re
from models import HeuristicScores, ScoringWeights

# ──────────────────────────────────────────────────────────────────────────────
# Keyword dictionaries
# ──────────────────────────────────────────────────────────────────────────────

HARD_SKILL_KEYWORDS = [
    # Languages
    "python", "java", "javascript", "typescript", "go", "golang", "rust",
    "c++", "c#", "kotlin", "swift", "scala", "php", "ruby", "r",
    # Web / Frontend
    "react", "vue", "angular", "next.js", "nuxt", "html", "css", "sass",
    # Backend / Infra
    "django", "flask", "fastapi", "spring", "node.js", "express",
    "docker", "kubernetes", "k8s", "terraform", "ansible", "ci/cd",
    # Data / ML
    "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "spark",
    # Cloud
    "aws", "azure", "gcp", "google cloud",
    # Other
    "git", "linux", "bash", "rest api", "graphql", "grpc", "kafka", "rabbitmq",
]

EXPERIENCE_KEYWORDS = [
    "разработ", "реализ", "внедр", "проектировани", "архитектур",
    "руководил", "лидировал", "запустил", "оптимизировал", "мигрировал",
    "developed", "implemented", "designed", "led", "built", "deployed",
    "maintained", "optimized", "migrated", "launched", "delivered",
    "год опыта", "лет опыта", "years of experience", "year of experience",
]

SOFT_SKILL_KEYWORDS = [
    "командн", "коммуникаци", "ответственн", "инициативн", "самостоятельн",
    "адаптивн", "обучаемост", "лидерств", "менторств", "презентаци",
    "teamwork", "communication", "leadership", "mentoring", "initiative",
    "adaptable", "proactive", "self-motivated", "ownership", "accountability",
]


def _keyword_hit_rate(text: str, keywords: list[str], cap: int) -> float:
    """
    Count unique keyword hits in text, return rate scaled to cap.
    Returns float in [0, 100].
    """
    lower = text.lower()
    hits = sum(1 for kw in keywords if re.search(re.escape(kw), lower))
    return min(hits / cap, 1.0) * 100


def compute_heuristic_scores(
    resume_text: str,
    vacancy_text: str,
    weights: ScoringWeights,
    hard_cap: int = 12,
    exp_cap: int = 8,
    soft_cap: int = 6,
) -> HeuristicScores:
    """
    Compute keyword-based sub-scores and weighted composite.

    Args:
        resume_text: Cleaned resume.
        vacancy_text: Job description (used to boost caps if vacancy mentions many keywords).
        weights: ScoringWeights instance.
        hard_cap: Max expected hard-skill keyword matches for 100%.
        exp_cap: Max expected experience keyword matches.
        soft_cap: Max expected soft-skill keyword matches.
    """
    combined = resume_text + " " + vacancy_text  # bias toward vacancy-relevant terms

    hard = _keyword_hit_rate(combined, HARD_SKILL_KEYWORDS, hard_cap)
    exp = _keyword_hit_rate(resume_text, EXPERIENCE_KEYWORDS, exp_cap)
    soft = _keyword_hit_rate(resume_text, SOFT_SKILL_KEYWORDS, soft_cap)

    composite = (
        hard * weights.hard_skills
        + exp * weights.experience
        + soft * weights.soft_skills
    )

    return HeuristicScores(
        hard_skills=round(hard, 1),
        experience=round(exp, 1),
        soft_skills=round(soft, 1),
        composite=round(composite, 1),
    )


def compute_final_score(gpt_score: int, heuristic: HeuristicScores) -> float:
    """
    Blend GPT score with heuristic composite.
    Gives GPT 70% weight (semantic understanding) and heuristic 30% (factual check).
    """
    blended = gpt_score * 0.70 + heuristic.composite * 0.30
    return round(min(blended, 100), 1)
