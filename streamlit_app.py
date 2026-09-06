from html import escape
from uuid import uuid4

import streamlit as st

from app.models.schemas import ResumeResult
from app.services.matcher import assign_dense_ranks, score_resume
from app.services.parser import parse_resume
from app.services.rag import answer_question


st.set_page_config(page_title="AI-Powered Resume Screening", page_icon="AI", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#16242b; --muted:#5f7078; --paper:#f5f6f4; --panel:#ffffff; --line:#dfe6e2; --navy:#122a3a; --teal:#087f72; --green:#cdeedb; --green-ink:#125d3d; --gray:#e9edef; --red:#ffe0d8; --red-ink:#9e432e; --amber:#ffe8b5; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); font-size:16px; }
    .stApp { background:var(--paper); }
    [data-testid="stHeader"] { background:transparent; }
    h1, h2, h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:0; color:var(--ink); }
    h1 { font-size:2.6rem !important; margin:1.2rem 0 .15rem; }
    h2 { font-size:1.65rem !important; }
    h3 { font-size:1.25rem !important; }
    .topbar { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--line); padding:12px 0 14px; }
    .brand { font:700 1.1rem 'Space Grotesk'; color:var(--navy); }
    .brand-mark { display:inline-block; background:var(--teal); color:white; border-radius:6px; padding:7px 9px; margin-right:9px; }
    .eyebrow { color:var(--teal); font-size:.8rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
    .subtitle { color:var(--muted); font-size:1.1rem; margin-bottom:1.8rem; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:22px; min-height:280px; }
    .panel-title { color:var(--navy); font:600 1.18rem 'Space Grotesk'; margin-bottom:5px; }
    .panel-hint { color:var(--muted); font-size:.93rem; margin-bottom:16px; }
    .metric-card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:20px 22px; min-height:135px; }
    .metric-label { color:var(--muted); font-size:.84rem; font-weight:700; text-transform:uppercase; letter-spacing:.09em; }
    .metric-value { color:var(--navy); font:700 2.55rem 'Space Grotesk'; margin-top:10px; }
    .metric-sub { color:var(--muted); font-size:1rem; margin-top:3px; }
    .score-good { color:var(--teal); } .score-mid { color:#a66b00; } .score-low { color:#c6533c; }
    .section-label { color:var(--navy); font-size:.87rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin:12px 0 7px; }
    .skill-chip { display:inline-block; border-radius:5px; padding:7px 11px; margin:0 6px 7px 0; font-size:.95rem; font-weight:600; }
    .skill-chip.extracted { background:#69777f; color:#ffffff; }
    .skill-chip.matched { background:#69b88d; color:#ffffff; }
    .skill-chip.missing { background:#59666e; color:#ffffff; }
    .badge { display:inline-block; border-radius:999px; padding:7px 13px; font-size:1rem; font-weight:700; }
    .badge.very-low { background:#ffd9d0; color:#a53b28; } .badge.low { background:var(--amber); color:#855500; }
    .badge.average { background:#dce9f4; color:#24577a; } .badge.high { background:var(--green); color:var(--green-ink); }
    .resume-preview { background:#fbfcfb; border:1px solid var(--line); border-radius:7px; padding:16px; white-space:pre-wrap; max-height:280px; overflow:auto; color:#32434a; font-size:.95rem; line-height:1.55; }
    .summary-box { background:#eef8f2; border-left:4px solid var(--teal); border-radius:6px; padding:15px 17px; color:#214b3c; line-height:1.55; font-size:1rem; }
    div[data-testid="stFileUploader"] { background:#fbfcfb; border:1px dashed #7ab6a5; border-radius:8px; padding:7px; }
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"],
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] { display:none !important; }
    div[data-testid="stTextArea"] textarea { font-size:1rem; }
    .stButton > button { border-radius:6px; font-weight:700; min-height:42px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _score_band(score: float) -> tuple[str, str]:
    if score < 25:
        return "Very Low", "very-low"
    if score < 50:
        return "Low", "low"
    if score <= 75:
        return "Average", "average"
    return "High", "high"


def _score_class(score: float) -> str:
    return "score-good" if score > 75 else "score-mid" if score >= 50 else "score-low"


def _chips(values: list[str], style: str) -> str:
    if not values:
        return '<span style="color:#718087">None identified</span>'
    return " ".join(f'<span class="skill-chip {style}">{escape(value)}</span>' for value in values)


def _extracted_skill_chips(extracted: list[str], matched: list[str]) -> str:
    if not extracted:
        return '<span style="color:#718087">No skills identified</span>'
    matched_set = set(matched)
    return " ".join(
        f'<span class="skill-chip {"matched" if skill in matched_set else "extracted"}">{escape(skill)}</span>'
        for skill in extracted
    )


def _file_names(names: list[str]) -> str:
    if not names:
        return '<span style="color:#718087">No CV uploaded yet</span>'
    return " ".join(f'<span class="skill-chip extracted">{escape(name)}</span>' for name in names)


def _render_result(result: ResumeResult, parsed_text: str) -> None:
    band, band_class = _score_band(result.match_score)
    st.markdown(f"### Rank {result.rank} · {escape(result.filename)}")
    st.caption(f"Experience detected: {result.experience or 'Not specified'}")
    score_col, band_col = st.columns([1, 3])
    with score_col:
        st.markdown(
            f'<div class="metric-value {_score_class(result.match_score)}">{result.match_score:.1f}<span style="font-size:1rem"> / 100</span></div>',
            unsafe_allow_html=True,
        )
    with band_col:
        st.markdown(
            f'<div class="section-label">Shortlist signal</div><span class="badge {band_class}">{band}</span><div class="metric-sub">Based on overall skill and keyword match</div>',
            unsafe_allow_html=True,
        )
    st.progress(min(result.match_score / 100, 1.0))

    st.markdown('<div class="section-label">All skills found in this CV</div>', unsafe_allow_html=True)
    st.markdown(_extracted_skill_chips(result.extracted_skills, result.matched_skills), unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-label">Matched demanded skills</div>', unsafe_allow_html=True)
        st.markdown(_chips(result.matched_skills, "matched"), unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-label">Missing demanded skills</div>', unsafe_allow_html=True)
        st.markdown(_chips(result.missing_skills, "missing"), unsafe_allow_html=True)
    st.caption(f"TF-IDF similarity: {result.cosine_similarity:.0%}  |  Skill coverage: {result.skill_coverage:.0%}")
    st.caption(f"Total CV skills: {result.total_skills}  |  Total demanded skills: {result.total_required_skills}")
    st.markdown('<div class="section-label">Resume preview</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="resume-preview">{escape(parsed_text[:12000])}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">What to change for a better fit</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="summary-box">{escape(result.explanation)}</div>', unsafe_allow_html=True)
    with st.expander("Ask a question about this resume"):
        question = st.text_input("Question", key=f"question_{result.resume_id}", placeholder="Does this candidate have deployment experience?")
        if st.button("Ask resume", key=f"ask_{result.resume_id}") and question.strip():
            answer, sources = answer_question(parsed_text, question)
            st.success(answer)
            st.caption("Sources: " + ", ".join(sources))
    st.divider()


st.markdown(
    '<div class="topbar"><div class="brand"><span class="brand-mark">AI</span>Resume Screening</div><div style="color:#5f7078;font-size:.95rem">Fast, evidence-based candidate matching</div></div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="eyebrow">Candidate intelligence workspace</div>', unsafe_allow_html=True)
st.title("AI-Powered Resume Screening")
st.markdown('<div class="subtitle">Compare a resume with a role, understand the evidence, and improve the candidate profile with local AI assistance.</div>', unsafe_allow_html=True)

upload_col, jd_col = st.columns(2, gap="large")
with upload_col:
    st.markdown('<div class="panel-title">Upload CVs</div><div class="panel-hint">Upload one or more resumes for comparison.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Uploaded CVs</div>', unsafe_allow_html=True)
    st.markdown(_file_names(st.session_state.get("uploaded_names", [])), unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Choose CV files",
        type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    current_names = [uploaded_file.name for uploaded_file in uploaded_files or []]
    if current_names != st.session_state.get("uploaded_names", []):
        st.session_state.uploaded_names = current_names
        st.rerun()
    if uploaded_files:
        st.markdown('<div class="section-label">Resume preview</div>', unsafe_allow_html=True)
        for uploaded_file in uploaded_files:
            try:
                preview = parse_resume(uploaded_file.name, uploaded_file.getvalue()).text
                with st.expander(uploaded_file.name, expanded=False):
                    st.markdown(f'<div class="resume-preview">{escape(preview[:5000])}</div>', unsafe_allow_html=True)
            except Exception as error:
                st.warning(f"Could not preview {uploaded_file.name}: {error}")
with jd_col:
    st.markdown('<div class="panel-title">Job description</div><div class="panel-hint">Paste the role requirements and must-have skills.</div>', unsafe_allow_html=True)
    if st.session_state.get("submitted_job_description"):
        st.markdown('<div class="section-label">Submitted job description</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="resume-preview">{escape(st.session_state.submitted_job_description)}</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Paste job description",
        height=170,
        placeholder="Example: Python developer with FastAPI, SQL, Docker and machine learning experience.",
        label_visibility="collapsed",
    )
    send_jd = st.button("Send job description", key="send_jd", use_container_width=True)

if send_jd:
    if not job_description.strip():
        st.error("Paste a job description before sending it.")
    else:
        st.session_state.submitted_job_description = job_description.strip()
        st.rerun()

st.session_state.job_description = job_description
active_job_description = st.session_state.get("submitted_job_description", job_description)
st.markdown("<br>", unsafe_allow_html=True)
screen_clicked = st.button("Screen resumes", type="primary", use_container_width=True)

if screen_clicked:
    if not active_job_description.strip():
        st.error("Paste and send a job description before screening.")
    elif not uploaded_files:
        st.error("Upload at least one CV before screening.")
    else:
        results: list[tuple[ResumeResult, str]] = []
        errors: list[str] = []
        with st.spinner("Reading CVs and comparing skills..."):
            for uploaded_file in uploaded_files:
                try:
                    parsed = parse_resume(uploaded_file.name, uploaded_file.getvalue())
                    results.append((score_resume(active_job_description, parsed, str(uuid4())), parsed.text))
                except Exception as error:
                    errors.append(f"{uploaded_file.name}: {error}")
        results.sort(key=lambda item: item[0].match_score, reverse=True)
        assign_dense_ranks([result for result, _ in results])
        st.session_state.screening_results = results
        st.session_state.screening_errors = errors

for error in st.session_state.get("screening_errors", []):
    st.error(error)

screening_results = st.session_state.get("screening_results", [])

if screening_results:
    st.markdown("## Screening overview")
    best_score = screening_results[0][0].match_score
    high_count = sum(1 for result, _ in screening_results if _score_band(result.match_score)[0] == "High")
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.markdown(f'<div class="metric-card"><div class="metric-label">Resumes screened</div><div class="metric-value">{len(screening_results)}</div><div class="metric-sub">Candidate files reviewed</div></div>', unsafe_allow_html=True)
    metric_two.markdown(f'<div class="metric-card"><div class="metric-label">Best match</div><div class="metric-value {_score_class(best_score)}">{best_score:.1f}</div><div class="metric-sub">Highest relevance score</div></div>', unsafe_allow_html=True)
    metric_three.markdown(f'<div class="metric-card"><div class="metric-label">Shortlist signals</div><div class="metric-value">{high_count}</div><div class="metric-sub" style="font-size:1.12rem;font-weight:700">High match candidates</div></div>', unsafe_allow_html=True)
    st.markdown("## Candidate results")
    for result, text in screening_results:
        with st.container(border=True):
            _render_result(result, text)
