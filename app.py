"""
HR Pre-Scoring System — Streamlit UI
=====================================
Features:
  • Upload vacancy + PDF resume
  • Dynamic weight sliders (auto-normalize to 100%)
  • Single-model assessment
  • Multi-model comparison (side-by-side)
  • Heuristic keyword analysis
  • Persistent history with export
  • Visual score gauges and charts
"""
import os
import json
import time

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

from parser import parse_resume
from gpt_service import assess_candidate, compare_models, AVAILABLE_MODELS
from scoring import compute_heuristic_scores, compute_final_score
from models import ScoringWeights, CandidateResult, HeuristicScores
from history import append_result, load_history, clear_history, export_json

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Pre-Scoring AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* General */
    .main { background: #0f1117; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }

    /* Score badge */
    .score-badge {
        display: inline-block;
        padding: 12px 28px;
        border-radius: 50px;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -1px;
        text-align: center;
    }
    .score-green  { background: #1a3a2a; color: #4ade80; border: 2px solid #4ade80; }
    .score-orange { background: #3a2a0a; color: #fb923c; border: 2px solid #fb923c; }
    .score-red    { background: #3a0a0a; color: #f87171; border: 2px solid #f87171; }

    /* Tag pills */
    .tag-green  { display:inline-block; background:#1a3a2a; color:#4ade80;
                  border-radius:20px; padding:3px 12px; margin:3px; font-size:.85rem; }
    .tag-red    { display:inline-block; background:#3a0a0a; color:#f87171;
                  border-radius:20px; padding:3px 12px; margin:3px; font-size:.85rem; }
    .tag-orange { display:inline-block; background:#3a2a0a; color:#fb923c;
                  border-radius:20px; padding:3px 12px; margin:3px; font-size:.85rem; }
    .tag-blue   { display:inline-block; background:#0a1a3a; color:#60a5fa;
                  border-radius:20px; padding:3px 12px; margin:3px; font-size:.85rem; }

    /* Info box */
    .info-box {
        background: #1e2030; border-left: 4px solid #6366f1;
        border-radius: 8px; padding: 14px 18px; margin: 10px 0;
        font-size: .9rem; line-height: 1.6;
        color: #e2e8f0 !important;
    }

    /* Divider */
    .fancy-divider {
        height: 2px; background: linear-gradient(90deg, #6366f1, #a855f7, transparent);
        margin: 24px 0; border: none;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def score_color(score: float) -> str:
    if score >= 75:
        return "green"
    elif score >= 50:
        return "orange"
    return "red"


def render_score_badge(score: float) -> None:
    color = score_color(score)
    st.markdown(
        f'<div class="score-badge score-{color}">{score}</div>',
        unsafe_allow_html=True,
    )


def render_tags(items: list[str], color: str = "blue") -> None:
    if not items:
        st.markdown("*нет данных*")
        return
    html = "".join(f'<span class="tag-{color}">{item}</span>' for item in items)
    st.markdown(html, unsafe_allow_html=True)


def gauge_chart(score: float, title: str, height: int = 220) -> go.Figure:
    color = {"green": "#4ade80", "orange": "#fb923c", "red": "#f87171"}[score_color(score)]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 14, "color": "#9ca3af"}},
        number={"font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#374151"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1e2030",
            "bordercolor": "#374151",
            "steps": [
                {"range": [0, 50],  "color": "#1a1a2e"},
                {"range": [50, 75], "color": "#1a2030"},
                {"range": [75, 100],"color": "#1a2a20"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#0f1117",
        font_color="#d1d5db",
    )
    return fig


def radar_chart(heuristic: HeuristicScores) -> go.Figure:
    categories = ["Hard Skills", "Опыт", "Soft Skills"]
    values = [heuristic.hard_skills, heuristic.experience, heuristic.soft_skills]
    values_closed = values + [values[0]]
    cats_closed = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor="rgba(99,102,241,0.25)",
        line_color="#6366f1",
        name="Кандидат",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#6b7280"),
            bgcolor="#1e2030",
            angularaxis=dict(color="#9ca3af"),
        ),
        showlegend=False,
        paper_bgcolor="#0f1117",
        height=280,
        margin=dict(l=30, r=30, t=30, b=30),
    )
    return fig


def model_comparison_bar(comparison: dict) -> go.Figure:
    """Bar chart comparing scores across models."""
    models = list(comparison.keys())
    scores = [comparison[m].score for m in models]
    colors = [{"green": "#4ade80", "orange": "#fb923c", "red": "#f87171"}[score_color(s)] for s in scores]

    fig = go.Figure(go.Bar(
        x=models,
        y=scores,
        marker_color=colors,
        text=[f"{s}" for s in scores],
        textposition="outside",
        textfont=dict(size=16, color="#d1d5db"),
    ))
    fig.add_hline(y=75, line_dash="dash", line_color="#4ade80",
                  annotation_text="Отлично ≥75", annotation_position="right")
    fig.add_hline(y=50, line_dash="dash", line_color="#fb923c",
                  annotation_text="Частично ≥50", annotation_position="right")
    fig.update_layout(
        yaxis=dict(range=[0, 110], title="Score", color="#9ca3af"),
        xaxis=dict(color="#9ca3af"),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1e2030",
        height=320,
        margin=dict(l=40, r=80, t=20, b=40),
        font_color="#d1d5db",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 HR Pre-Scoring AI")
    st.caption("Автоматическая оценка кандидатов с помощью GPT")
    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.sidebar.warning("⚠️ Добавь OPENAI_API_KEY в файл .env")
    

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Weights ──────────────────────────────────────────────────────────────
    st.markdown("### ⚖️ Веса критериев")
    st.caption("Сумма автоматически нормализуется к 100%")

    raw_hard = st.slider("🔧 Hard Skills", 0, 100, 60, 5)
    raw_exp  = st.slider("📂 Опыт работы",  0, 100, 25, 5)
    raw_soft = st.slider("🤝 Soft Skills",  0, 100, 15, 5)

    total_raw = raw_hard + raw_exp + raw_soft
    if total_raw == 0:
        total_raw = 1  # avoid division by zero

    w_hard = raw_hard / total_raw
    w_exp  = raw_exp  / total_raw
    w_soft = raw_soft / total_raw

    weights = ScoringWeights(hard_skills=w_hard, experience=w_exp, soft_skills=w_soft)

    st.markdown(
        f"""
        <div class="info-box" style="color:#e2e8f0">
        ✅ Нормализовано:<br>
        🔧 Hard: <b>{w_hard*100:.0f}%</b> &nbsp;
        📂 Опыт: <b>{w_exp*100:.0f}%</b> &nbsp;
        🤝 Soft: <b>{w_soft*100:.0f}%</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Model selection ───────────────────────────────────────────────────────
    st.markdown("### 🤖 Модели GPT")
    mode = st.radio(
        "Режим работы",
        ["Одна модель", "Сравнение моделей"],
        help="Сравнение запускает несколько моделей параллельно",
    )

    if mode == "Одна модель":
        selected_label = st.selectbox("Выбери модель", list(AVAILABLE_MODELS.keys()))
        selected_model = AVAILABLE_MODELS[selected_label]
        compare_mode = False
    else:
        compare_labels = st.multiselect(
            "Выбери модели для сравнения",
            list(AVAILABLE_MODELS.keys()),
            default=list(AVAILABLE_MODELS.keys())[:2],
        )
        selected_models = [AVAILABLE_MODELS[l] for l in compare_labels]
        compare_mode = True

    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.2, 0.05,
                             help="Ниже = детерминированнее, Выше = креативнее")

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    st.markdown("### ℹ️ О проекте")
    st.caption(
        "Система прескоринга объединяет GPT-анализ "
        "(70%) и эвристику по ключевым словам (30%) "
        "для объективной оценки кандидатов."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main area — tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_assess, tab_compare, tab_history = st.tabs([
    "🎯 Оценка кандидата",
    "🔬 Сравнение моделей",
    "📋 История оценок",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Assess candidate
# ══════════════════════════════════════════════════════════════════════════════
with tab_assess:
    st.markdown("## 🎯 Оценка кандидата")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 📋 Вакансия")
        vacancy_text = st.text_area(
            "Описание вакансии",
            height=260,
            placeholder=(
                "Вставьте полное описание вакансии: требования, стек технологий, "
                "обязанности, уровень позиции…"
            ),
        )

    with col_right:
        st.markdown("### 👤 Кандидат")
        candidate_name = st.text_input("Имя кандидата", placeholder="Иванов Иван")
        vacancy_title  = st.text_input("Название позиции", placeholder="Python Backend Developer")

        uploaded_pdf = st.file_uploader("📎 Резюме (PDF)", type=["pdf"])
        manual_text  = st.text_area(
            "Или вставьте резюме текстом",
            height=130,
            placeholder="Если PDF недоступен — вставьте текст резюме вручную…",
        )

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Assess button ─────────────────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        assess_btn = st.button("🚀 Оценить кандидата", type="primary", width='stretch')

    if assess_btn:
        # Validation
        if not api_key:
            st.error("❌ Введи OpenAI API Key в боковой панели")
            st.stop()
        if not vacancy_text.strip():
            st.error("❌ Заполни описание вакансии")
            st.stop()
        if not candidate_name.strip():
            st.error("❌ Введи имя кандидата")
            st.stop()
        if not uploaded_pdf and not manual_text.strip():
            st.error("❌ Загрузи PDF или вставь текст резюме")
            st.stop()

        # Parse resume
        with st.spinner("📄 Читаю резюме…"):
            file_bytes = uploaded_pdf.read() if uploaded_pdf else None
            resume_text = parse_resume(
                file_bytes=file_bytes,
                raw_text=manual_text if not file_bytes else None,
            )

        if not resume_text.strip():
            st.error("❌ Не удалось извлечь текст из резюме")
            st.stop()

        # Heuristic scoring (fast, local)
        heuristic = compute_heuristic_scores(resume_text, vacancy_text, weights)

        if compare_mode:
            # ── Multi-model path ──────────────────────────────────────────────
            if not selected_models:
                st.warning("Выбери хотя бы одну модель для сравнения")
                st.stop()

            with st.spinner(f"🤖 Запрашиваю {len(selected_models)} модели…"):
                comparison = compare_models(
                    vacancy_text=vacancy_text,
                    resume_text=resume_text,
                    model_keys=selected_models,
                    api_key=api_key,
                    temperature=temperature,
                )

            st.success("✅ Готово! Результаты ниже.")

            # Store in session for compare tab
            st.session_state["last_comparison"] = {
                "candidate_name": candidate_name,
                "vacancy_title": vacancy_title,
                "comparison": comparison,
                "heuristic": heuristic,
            }

            # Quick preview
            st.markdown("### 📊 Быстрый обзор по моделям")
            fig = model_comparison_bar(comparison)
            st.plotly_chart(fig, width='stretch', key="compare_bar_preview")

            st.info("👉 Перейди на вкладку **🔬 Сравнение моделей** для детального анализа")

        else:
            # ── Single-model path ─────────────────────────────────────────────
            with st.spinner(f"🤖 Анализирую через {selected_model}…"):
                assessment = assess_candidate(
                    vacancy_text=vacancy_text,
                    resume_text=resume_text,
                    model_key=selected_model,
                    api_key=api_key,
                    temperature=temperature,
                )

            final_score = compute_final_score(assessment.score, heuristic)

            result = CandidateResult(
                candidate_name=candidate_name,
                vacancy_title=vacancy_title or "—",
                gpt_assessment=assessment,
                heuristic_scores=heuristic,
                weights=weights,
                final_score=final_score,
                model_name=selected_model,
            )

            # ── Results layout ────────────────────────────────────────────────
            st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
            st.markdown(f"## Результаты: {candidate_name}")

            top_left, top_mid, top_right = st.columns([1.2, 1, 1])

            with top_left:
                st.markdown("#### 🏆 Итоговый балл")
                render_score_badge(final_score)
                st.markdown(f"**{result.verdict()}**")
                st.caption(f"Модель: `{selected_model}` · GPT: {assessment.score} · Эвристика: {heuristic.composite}")

            with top_mid:
                st.plotly_chart(gauge_chart(assessment.score, "GPT Score"), width='stretch', key="gauge_gpt")

            with top_right:
                st.plotly_chart(gauge_chart(heuristic.composite, "Heuristic Score"), width='stretch', key="gauge_heuristic")

            # Radar
            st.markdown("#### 📡 Профиль по категориям")
            col_radar, col_details = st.columns([1, 1.5])

            with col_radar:
                st.plotly_chart(radar_chart(heuristic), width='stretch', key="radar_assess")

            with col_details:
                metric_cols = st.columns(3)
                metric_cols[0].metric("🔧 Hard Skills", f"{heuristic.hard_skills:.0f}/100")
                metric_cols[1].metric("📂 Опыт", f"{heuristic.experience:.0f}/100")
                metric_cols[2].metric("🤝 Soft Skills", f"{heuristic.soft_skills:.0f}/100")

                st.markdown('<br>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-box" style="color:#e2e8f0">{assessment.summary}</div>', unsafe_allow_html=True)

            # Strengths / Weaknesses / Missing
            st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("#### ✅ Сильные стороны")
                render_tags(assessment.strengths, "green")

            with c2:
                st.markdown("#### ⚠️ Слабые стороны")
                render_tags(assessment.weaknesses, "orange")

            with c3:
                st.markdown("#### ❌ Отсутствующие навыки")
                render_tags(assessment.missing_skills, "red")

            # Save to history
            append_result(result.to_history_row())
            st.success("💾 Результат сохранён в историю")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Model comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("## 🔬 Сравнение моделей")

    if "last_comparison" not in st.session_state:
        st.info("Пока нет данных. Запусти оценку в режиме **Сравнение моделей** на вкладке «Оценка кандидата».")
    else:
        ctx = st.session_state["last_comparison"]
        comparison: dict = ctx["comparison"]
        heuristic: HeuristicScores = ctx["heuristic"]

        st.markdown(f"**Кандидат:** {ctx['candidate_name']} · **Вакансия:** {ctx['vacancy_title']}")

        # ── Bar chart ──────────────────────────────────────────────────────────
        st.markdown("### 📊 Score по моделям")
        st.plotly_chart(model_comparison_bar(comparison), width='stretch', key="compare_bar_main")

        # ── Side-by-side cards ─────────────────────────────────────────────────
        st.markdown("### 📋 Детальное сравнение")
        cols = st.columns(len(comparison))

        for idx, (model_key, assessment) in enumerate(comparison.items()):
            with cols[idx]:
                final = compute_final_score(assessment.score, heuristic)
                color = score_color(final)
                st.markdown(f"#### `{model_key}`")
                render_score_badge(final)
                st.caption(f"GPT raw: {assessment.score}")

                st.markdown("**✅ Сильные:**")
                render_tags(assessment.strengths, "green")

                st.markdown("**⚠️ Слабые:**")
                render_tags(assessment.weaknesses, "orange")

                st.markdown("**❌ Отсутствуют:**")
                render_tags(assessment.missing_skills, "red")

                st.markdown(f'<div class="info-box" style="color:#e2e8f0">{assessment.summary}</div>',
                            unsafe_allow_html=True)

        # ── Radar overlay ──────────────────────────────────────────────────────
        st.markdown("### 📡 Эвристический профиль (общий)")
        col_r, col_s = st.columns([1, 1.5])
        with col_r:
            st.plotly_chart(radar_chart(heuristic), width='stretch', key="radar_compare")
        with col_s:
            st.markdown("Эвристика одинакова для всех моделей — она основана на анализе ключевых слов резюме и не зависит от выбранной LLM.")
            mcols = st.columns(3)
            mcols[0].metric("Hard Skills", f"{heuristic.hard_skills:.0f}")
            mcols[1].metric("Опыт", f"{heuristic.experience:.0f}")
            mcols[2].metric("Soft Skills", f"{heuristic.soft_skills:.0f}")

        # ── Export comparison ──────────────────────────────────────────────────
        comparison_export = {
            model: {
                "score": a.score,
                "strengths": a.strengths,
                "weaknesses": a.weaknesses,
                "missing_skills": a.missing_skills,
                "summary": a.summary,
            }
            for model, a in comparison.items()
        }
        st.download_button(
            "⬇️ Скачать сравнение (JSON)",
            data=json.dumps(comparison_export, ensure_ascii=False, indent=2),
            file_name=f"comparison_{ctx['candidate_name']}.json",
            mime="application/json",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: History
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("## 📋 История оценок")

    history = load_history()

    if not history:
        st.info("История пуста. Оцени первого кандидата!")
    else:
        df = pd.DataFrame(history)

        # ── Summary KPIs ───────────────────────────────────────────────────────
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("👥 Всего кандидатов", len(df))
        kpi_cols[1].metric("✅ Отличных (≥75)", int((df["Итог"] >= 75).sum()))
        kpi_cols[2].metric("⚠️ Частично (50–74)", int(((df["Итог"] >= 50) & (df["Итог"] < 75)).sum()))
        kpi_cols[3].metric("❌ Не подходят (<50)", int((df["Итог"] < 50).sum()))

        # ── Distribution chart ─────────────────────────────────────────────────
        fig_dist = px.histogram(
            df, x="Итог", nbins=10, title="Распределение итоговых баллов",
            color_discrete_sequence=["#6366f1"],
            template="plotly_dark",
        )
        fig_dist.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1e2030",
                                height=240, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_dist, width='stretch', key="history_dist")

        # ── Sortable table ─────────────────────────────────────────────────────
        display_cols = ["Кандидат", "Вакансия", "Модель", "Итог",
                        "GPT-оценка", "Hard Skills", "Опыт", "Soft Skills",
                        "Вердикт", "Время"]
        existing_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[existing_cols].sort_values("Итог", ascending=False),
            width='stretch',
            hide_index=True,
        )

        # ── Actions ───────────────────────────────────────────────────────────
        col_dl, col_clear, _ = st.columns([1, 1, 3])
        with col_dl:
            st.download_button(
                "⬇️ Скачать историю (JSON)",
                data=export_json(),
                file_name="prescoring_history.json",
                mime="application/json",
            )
        with col_clear:
            if st.button("🗑️ Очистить историю"):
                clear_history()
                st.rerun()
