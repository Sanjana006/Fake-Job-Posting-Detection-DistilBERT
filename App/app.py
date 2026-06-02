import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Job Detector · DistilBERT",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset & base ─────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
}

/* ── Full-page animated gradient ──────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #0f2027 30%, #1a3a4a 60%, #0d2137 100%);
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite;
    min-height: 100vh;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Hide Streamlit chrome ────────────────────────────────── */
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Remove Streamlit's default top padding ───────────────── */
.block-container {
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* ── Remove gap between markdown blocks ──────────────────── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stVerticalBlock"] > div {
    gap: 0 !important;
}

/* ── Section title above form ────────────────────────────── */
.form-section-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(168, 216, 234, 0.7);
    margin-bottom: 12px;
    margin-top: 4px;
}

/* ── Form container glassmorphism ────────────────────────── */
[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.055) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 20px !important;
    padding: 32px 36px !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08) !important;
}

/* ── Input & textarea backgrounds ────────────────────────── */
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background: rgba(255, 255, 255, 0.07) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
    transition: all 0.25s ease !important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="textarea"] > div:focus-within {
    background: rgba(255, 255, 255, 0.12) !important;
    border-color: #4fc3f7 !important;
    box-shadow: 0 0 0 3px rgba(79, 195, 247, 0.15) !important;
}

/* ── Input text color ─────────────────────────────────────── */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    color: #e8f4f8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    font-weight: 400 !important;
    caret-color: #4fc3f7 !important;
}
div[data-baseweb="textarea"] textarea::placeholder,
div[data-baseweb="input"] input::placeholder {
    color: rgba(255,255,255,0.28) !important;
}

/* ── Field labels ─────────────────────────────────────────── */
label[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] p {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: rgba(168, 216, 234, 0.9) !important;
    letter-spacing: 0.4px !important;
    text-transform: uppercase !important;
    margin-bottom: 5px !important;
}

/* ── Submit button ────────────────────────────────────────── */
[data-testid="stForm"] [data-testid="stButton"] > button,
[data-testid="stFormSubmitButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #1565c0 0%, #0d47a1 50%, #01579b 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 0 !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 24px rgba(13, 71, 161, 0.5) !important;
    margin-top: 4px !important;
}
[data-testid="stForm"] [data-testid="stButton"] > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, #1976d2 0%, #1565c0 50%, #0d47a1 100%) !important;
    box-shadow: 0 8px 32px rgba(13, 71, 161, 0.65) !important;
    transform: translateY(-2px) !important;
}

/* ── Column gap tightening ────────────────────────────────── */
[data-testid="stColumns"] {
    gap: 16px !important;
}

/* ── Result containers ────────────────────────────────────── */
[data-testid="stSuccess"] {
    background: linear-gradient(135deg, rgba(27,94,32,0.55), rgba(46,125,50,0.35)) !important;
    border: 1px solid rgba(76,175,80,0.5) !important;
    border-left: 4px solid #4caf50 !important;
    border-radius: 14px !important;
    backdrop-filter: blur(12px) !important;
    color: #c8e6c9 !important;
}
[data-testid="stError"] {
    background: linear-gradient(135deg, rgba(183,28,28,0.55), rgba(198,40,40,0.35)) !important;
    border: 1px solid rgba(244,67,54,0.5) !important;
    border-left: 4px solid #f44336 !important;
    border-radius: 14px !important;
    backdrop-filter: blur(12px) !important;
    color: #ffcdd2 !important;
}
[data-testid="stSuccess"] p,
[data-testid="stError"] p {
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* ── Progress bar ─────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: rgba(255,255,255,0.1) !important;
    border-radius: 100px !important;
}
[data-testid="stProgress"] > div > div > div {
    border-radius: 100px !important;
}

/* ── Metric cards ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    padding: 18px 22px !important;
    text-align: center !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stMetricLabel"] p {
    color: rgba(168, 216, 234, 0.7) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #a8d8ea !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
}

/* ── Divider ──────────────────────────────────────────────── */
hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 0.5rem 0 !important;
}

/* ── Warning style ────────────────────────────────────────── */
[data-testid="stAlert"] {
    background: rgba(255, 152, 0, 0.15) !important;
    border: 1px solid rgba(255,152,0,0.35) !important;
    border-radius: 12px !important;
    color: #ffe082 !important;
}
</style>
""", unsafe_allow_html=True)


HF_MODEL = "your-username/fraud-model-repo"


# ── Model loading (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "fraud_distilbert_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(MODEL_PATH):
        target = MODEL_PATH
    else:
        target = HF_MODEL
        
    mdl = DistilBertForSequenceClassification.from_pretrained(target)
    tok = DistilBertTokenizerFast.from_pretrained(target)
    mdl.to(device)
    mdl.eval()
    return mdl, tok, device


def format_job_text(job_title, company_profile, description, requirements, benefits):
    return f"""
    Job Title: {job_title}

    Company Profile:
    {company_profile}

    Job Description:
    {description}

    Requirements:
    {requirements}

    Benefits:
    {benefits}
    """


def predict_fraud(text, mdl, tok, device):
    inputs = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = mdl(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    label = "fraud" if pred == 1 else "genuine"
    confidence = round(probs[0][pred].item() * 100, 2)
    return label, confidence


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 48px 0 28px 0;">
    <div style="
        display:inline-flex; align-items:center; gap:8px;
        background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.15);
        border-radius:50px; padding:8px 22px; font-size:14px;
        color:#a8d8ea; font-weight:500; letter-spacing:0.5px;
        margin-bottom:22px; backdrop-filter:blur(8px);
    ">
        🤖 &nbsp; Powered by DistilBERT · Fine-tuned on 17K+ Jobs
    </div>
    <h1 style="
        font-size:3.8rem; font-weight:900; color:#fff;
        line-height:1.1; letter-spacing:-1.5px; margin-bottom:16px;
    ">
        Fake Job Posting<br>
        <span style="
            background:linear-gradient(90deg,#4fc3f7,#81d4fa,#29b6f6);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            background-clip:text;
        ">Detector</span>
    </h1>
    <p style="
        font-size:1.15rem; color:rgba(255,255,255,0.55);
        font-weight:400; line-height:1.6; max-width:520px; margin:0 auto;
    ">
        Paste any job listing below — our AI model will instantly tell you
        whether it's <strong style="color:#81d4fa;">genuine</strong> or a
        <strong style="color:#ef9a9a;">scam</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Stats row ──────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Model Accuracy", "98.2%")
with c2:
    st.metric("Jobs Trained On", "17K+")
with c3:
    st.metric("Inference Time", "< 1s")

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# ── Form ───────────────────────────────────────────────────────────────────────
st.markdown('<p class="form-section-title">📋 &nbsp; Enter Job Details</p>', unsafe_allow_html=True)

with st.form("job_form", clear_on_submit=False):
    job_title = st.text_input(
        "Job Title",
        placeholder="e.g. Senior Data Scientist at Acme Corp",
    )

    col1, col2 = st.columns(2)
    with col1:
        company_profile = st.text_area(
            "Company Profile",
            placeholder="Brief company description, industry, size…",
            height=130,
        )
    with col2:
        description = st.text_area(
            "Job Description",
            placeholder="Roles, responsibilities, day-to-day tasks…",
            height=130,
        )

    col3, col4 = st.columns(2)
    with col3:
        requirements = st.text_area(
            "Requirements",
            placeholder="Skills, qualifications, experience needed…",
            height=120,
        )
    with col4:
        benefits = st.text_area(
            "Benefits",
            placeholder="Salary range, perks, remote options…",
            height=120,
        )

    submitted = st.form_submit_button("🔍  Analyse Job Posting", use_container_width=True)

# ── Result ─────────────────────────────────────────────────────────────────────
if submitted:
    if not job_title.strip():
        st.warning("⚠️  Please enter at least a **Job Title** before analysing.")
    else:
        with st.spinner("Running DistilBERT inference…"):
            mdl, tok, device = load_model()
            combined = format_job_text(job_title, company_profile, description, requirements, benefits)
            label, confidence = predict_fraud(combined, mdl, tok, device)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        if label == "genuine":
            st.success(f"✅  **This Job is Genuine** — no significant fraud signals detected.")
            bar_color = "#4caf50"
        else:
            st.error(f"🚨  **This Job is Fraudulent** — strong scam indicators found. Proceed with caution!")
            bar_color = "#f44336"

        # Confidence bar via HTML
        st.markdown(f"""
        <div style="
            margin-top: 16px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 14px;
            padding: 20px 26px;
            backdrop-filter: blur(10px);
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <span style="font-size:13px; font-weight:700; color:rgba(168,216,234,0.75);
                             letter-spacing:1px; text-transform:uppercase;">
                    Model Confidence
                </span>
                <span style="font-size:1.6rem; font-weight:900; color:#fff;">
                    {confidence}%
                </span>
            </div>
            <div style="background:rgba(255,255,255,0.08); border-radius:100px; height:10px; overflow:hidden;">
                <div style="
                    width:{int(confidence)}%;
                    height:100%;
                    border-radius:100px;
                    background:linear-gradient(90deg, {bar_color}aa, {bar_color});
                    box-shadow: 0 0 12px {bar_color}66;
                    transition: width 0.8s ease;
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)