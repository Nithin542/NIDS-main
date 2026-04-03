import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from dotenv import load_dotenv
import pandas as pd
import json

load_dotenv()
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import numpy as np
import warnings
from openai import OpenAI
from datetime import datetime

warnings.filterwarnings("ignore")
pd.set_option("styler.render.max_elements", 5_000_000)

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="NetShield NIDS — Cloud Security Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# ENTERPRISE CSS — GLASSMORPHISM + ANIMATED GRADIENTS
# ======================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---- Global ---- */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 40%, #0f172a 100%);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit header and footer for cleaner look */
header[data-testid="stHeader"] { background: transparent; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.15);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c7d2fe !important;
}

/* ---- Glassmorphism Cards ---- */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover {
    background: rgba(255, 255, 255, 0.07);
    border-color: rgba(99, 102, 241, 0.3);
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3), 0 0 30px rgba(99, 102, 241, 0.1);
}

/* ---- KPI Metric Cards ---- */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 16px 0; }
.kpi-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-card:hover { transform: translateY(-3px); background: rgba(255,255,255,0.06); }
.kpi-card.info::before    { background: linear-gradient(90deg, #6366f1, #818cf8); }
.kpi-card.danger::before  { background: linear-gradient(90deg, #ef4444, #f87171); }
.kpi-card.safe::before    { background: linear-gradient(90deg, #10b981, #34d399); }
.kpi-card.warn::before    { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.kpi-label {
    font-size: 0.72rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    font-weight: 600;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.kpi-value.info   { color: #818cf8; }
.kpi-value.danger { color: #f87171; }
.kpi-value.safe   { color: #34d399; }
.kpi-value.warn   { color: #fbbf24; }

/* ---- Mode Selection Cards ---- */
.mode-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 20px;
    padding: 40px 32px;
    text-align: center;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
}
.mode-card:hover {
    background: rgba(99, 102, 241, 0.08);
    border-color: rgba(99, 102, 241, 0.35);
    transform: translateY(-8px);
    box-shadow: 0 24px 48px rgba(0, 0, 0, 0.25), 0 0 40px rgba(99, 102, 241, 0.12);
}
.mode-icon { font-size: 3.5rem; margin-bottom: 16px; filter: drop-shadow(0 4px 8px rgba(99, 102, 241, 0.3)); }
.mode-title { font-size: 1.5rem; font-weight: 700; color: #e2e8f0; margin-bottom: 10px; }
.mode-desc  { font-size: 0.88rem; color: #94a3b8; line-height: 1.6; }

/* ---- Upload Hint ---- */
.upload-hint {
    background: rgba(99, 102, 241, 0.06);
    border: 1px dashed rgba(99, 102, 241, 0.25);
    border-radius: 12px;
    padding: 18px 22px;
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.7;
}

/* ---- Section Headers ---- */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 12px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.section-header h3 { margin: 0; font-weight: 700; color: #e2e8f0; font-size: 1.15rem; }

/* ---- Animated Floating Particles (Home Page) ---- */
@keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 0.3; }
    50% { transform: translateY(-20px) rotate(180deg); opacity: 0.6; }
}
.particles {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}
.particle {
    position: absolute;
    width: 4px; height: 4px;
    background: rgba(99, 102, 241, 0.3);
    border-radius: 50%;
    animation: float 6s ease-in-out infinite;
}
.particle:nth-child(1) { left: 10%; top: 20%; animation-delay: 0s; animation-duration: 8s; }
.particle:nth-child(2) { left: 25%; top: 60%; animation-delay: 1s; animation-duration: 6s; }
.particle:nth-child(3) { left: 50%; top: 30%; animation-delay: 2s; animation-duration: 7s; }
.particle:nth-child(4) { left: 70%; top: 70%; animation-delay: 0.5s; animation-duration: 9s; }
.particle:nth-child(5) { left: 85%; top: 15%; animation-delay: 1.5s; animation-duration: 5s; }
.particle:nth-child(6) { left: 40%; top: 85%; animation-delay: 3s; animation-duration: 10s; }
.particle:nth-child(7) { left: 60%; top: 45%; animation-delay: 2.5s; animation-duration: 7.5s; }
.particle:nth-child(8) { left: 15%; top: 90%; animation-delay: 4s; animation-duration: 6.5s; }

/* ---- Threat Level Indicator ---- */
.threat-gauge {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.gauge-bar-bg {
    width: 100%;
    height: 10px;
    background: rgba(255,255,255,0.06);
    border-radius: 5px;
    overflow: hidden;
    margin: 12px 0 8px 0;
}
.gauge-bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
}
.gauge-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    color: #94a3b8;
}
.gauge-level {
    font-size: 1.4rem;
    font-weight: 800;
    margin-top: 4px;
}

/* ---- Animated Counter ---- */
@keyframes countUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.kpi-value { animation: countUp 0.6s ease-out; }

/* ---- Glow Effect on Attack Detection ---- */
@keyframes dangerGlow {
    0%, 100% { box-shadow: 0 0 5px rgba(248,113,113,0.1); }
    50% { box-shadow: 0 0 20px rgba(248,113,113,0.3), 0 0 40px rgba(248,113,113,0.1); }
}
.kpi-card.danger:has(.kpi-value:not(:empty)) {
    animation: dangerGlow 3s ease-in-out infinite;
}

/* ---- Hero Banner ---- */
.hero {
    text-align: center;
    padding: 48px 24px 32px 24px;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #6366f1, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
    letter-spacing: -1px;
}
.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.65;
}

/* ---- Status Bar ---- */
.status-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 18px;
    border-radius: 10px;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.2);
    margin-bottom: 16px;
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #10b981;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
}
.status-text { font-size: 0.82rem; color: #6ee7b7; font-weight: 500; }

/* ---- Threat Severity Badge ---- */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-critical { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
.badge-normal   { background: rgba(16, 185, 129, 0.12); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.25); }

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.3) !important;
    transform: translateY(-2px) !important;
}

/* ---- Dataframe ---- */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ---- Plotly Charts ---- */
.js-plotly-plot { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# SESSION STATE DEFAULTS
# ======================================================
if 'mode' not in st.session_state:
    st.session_state.mode = 'home'
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.le = None
    st.session_state.feature_names = None
    st.session_state.explainer = None

# ======================================================
# HELPERS
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "nids.db")

def load_model():
    if st.session_state.model_loaded:
        return True, ""
    try:
        with st.spinner("⏳ Initializing NetShield AI Engine…"):
            st.session_state.model        = joblib.load(os.path.join(BASE_DIR, 'nids_model.pkl'))
            st.session_state.scaler       = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
            st.session_state.le           = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))
            raw_fn                        = joblib.load(os.path.join(BASE_DIR, 'feature_names.pkl'))
            st.session_state.feature_names = [f for f in raw_fn if f != 'Attack Type']
            try:
                st.session_state.explainer = shap.TreeExplainer(st.session_state.model)
            except Exception:
                st.session_state.explainer = None
        st.session_state.model_loaded = True
        return True, ""
    except Exception as e:
        return False, str(e)

def get_live_data(limit=500):
    try:
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            try:
                db_url = st.secrets["DATABASE_URL"]
            except (KeyError, FileNotFoundError):
                db_url = f"sqlite:///{os.path.join(BASE_DIR, 'nids.db')}"

        if db_url.startswith("postgres://"):
             db_url = db_url.replace("postgres://", "postgresql://", 1)

        engine = create_engine(db_url)
        df = pd.read_sql_query(f"SELECT * FROM logs ORDER BY timestamp DESC LIMIT {limit}", engine)

        with engine.begin() as conn:
            true_total = conn.execute(text("SELECT COUNT(*) FROM logs")).scalar()
            true_attacks = conn.execute(text("SELECT COUNT(*) FROM logs WHERE attack_type != 'Normal Traffic'")).scalar()

        return df, true_total, true_attacks
    except Exception as e:
        print(f"Error fetching live data: {e}")
        return pd.DataFrame(), 0, 0

def kpi_card(label, value, css_class):
    st.markdown(
        f'<div class="kpi-card {css_class}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {css_class}">{value}</div></div>',
        unsafe_allow_html=True
    )

def shap_chart(X_array, row_label=""):
    explainer    = st.session_state.explainer
    feature_names = st.session_state.feature_names
    if explainer is None:
        st.warning("SHAP Explainer not available for this model type.")
        return None
    with st.spinner("Computing SHAP explanations…"):
        shap_vals = explainer.shap_values(X_array)
        if isinstance(shap_vals, list):
            class_idx = int(np.argmax([np.sum(np.abs(sv)) for sv in shap_vals]))
            sv = shap_vals[class_idx][0]
        elif len(shap_vals.shape) == 3:
            class_idx = int(np.argmax(np.sum(np.abs(shap_vals[0]), axis=0)))
            sv = shap_vals[0, :, class_idx]
        else:
            sv = shap_vals[0]

        n   = min(len(feature_names), len(sv))
        sdf = pd.DataFrame({'Feature': feature_names[:n], 'SHAP': sv[:n], 'Raw': X_array[0][:n]})
        sdf['Abs']   = sdf['SHAP'].abs()
        top_features = sdf.sort_values('Abs', ascending=False).head(15)
        sdf_plot = top_features.sort_values('Abs', ascending=True)
        sdf_plot['Color'] = sdf_plot['SHAP'].apply(lambda x: '#f87171' if x > 0 else '#818cf8')
        sdf_plot['Text']  = sdf_plot['Raw'].apply(lambda x: f"Val: {x:.2f}")

        fig = go.Figure(go.Bar(
            x=sdf_plot['SHAP'], y=sdf_plot['Feature'], orientation='h',
            marker_color=sdf_plot['Color'], text=sdf_plot['Text'], textposition='inside',
            marker=dict(line=dict(width=0)),
        ))
        fig.update_layout(
            title=dict(
                text=f"SHAP Feature Contributions{' — ' + row_label if row_label else ''}",
                font=dict(size=15, color='#e2e8f0')
            ),
            xaxis_title="SHAP Value  (🔴 Attack ← → 🔵 Normal)",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='Inter'), height=500,
            margin=dict(l=200, r=30, t=60, b=60),
            xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.04)')
        )
        st.plotly_chart(fig, use_container_width=True)
        return top_features[['Feature', 'SHAP', 'Raw']].to_dict('records')

# ======================================================
# AI-POWERED EXPLANATION (Groq)
# ======================================================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "YOUR_API_KEY_HERE")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

def ai_explain(attack_type, confidence, shap_data, src_ip="", dst_ip="", btn_key="ai_btn"):
    if st.button("🤖 Generate AI Threat Intelligence Report", key=btn_key, use_container_width=True):
        feature_lines = "\n".join(
            f"  • {r['Feature']}: SHAP={r['SHAP']:.4f}, Scaled Value={r['Raw']:.3f}"
            for r in shap_data
        )
        prompt = f"""You are a senior cybersecurity analyst reviewing a Network Intrusion Detection System (NIDS) alert.

The ML model classified a network flow as: **{attack_type}** with {confidence:.1f}% confidence.
{f'Source IP: {src_ip} → Destination IP: {dst_ip}' if src_ip else ''}

Below are the top SHAP feature contributions (positive SHAP = pushes toward this attack class, negative = pushes toward normal):
{feature_lines}

Please provide:
1. **Plain-English Summary** — What happened in simple terms a non-technical manager could understand.
2. **Technical Analysis** — Which network features are abnormal and why they indicate this specific attack type.
3. **Risk Assessment** — How severe is this? (Critical / High / Medium / Low)
4. **Recommended Actions** — What should the security team do right now?

Keep it concise but insightful. Use bullet points."""

        try:
            client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
            with st.spinner("🧠 AI engine analyzing threat patterns…"):
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a cybersecurity expert who explains ML-based intrusion detection results in clear, actionable language."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.4,
                    max_tokens=800
                )
            explanation = response.choices[0].message.content
            st.markdown("---")
            st.markdown("#### 🤖 AI Threat Intelligence Report")
            st.markdown(f"""<div class="glass-card" style="border-left: 3px solid #6366f1;">
                {explanation.replace(chr(10), '<br>')}
            </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"AI explanation error: {e}")

def stop_button():
    if st.button("⏹ Return to Home", key="stop_btn"):
        st.session_state.mode = 'home'
        st.rerun()

# ======================================================
# SIDEBAR — NAVIGATION & SYSTEM INFO
# ======================================================
with st.sidebar:
    st.markdown("# 🛡️ NetShield")
    st.markdown("<p style='color:#94a3b8; font-size:0.82rem; margin-top:-10px;'>Cloud Security Intelligence Platform</p>", unsafe_allow_html=True)
    st.markdown("---")

    nav = st.radio(
        "Navigation",
        ["🏠 Home", "📡 Live Monitor", "📂 CSV Analysis"],
        index=["home", "live", "csv"].index(st.session_state.mode) if st.session_state.mode in ["home", "live", "csv"] else 0,
        label_visibility="collapsed"
    )
    nav_map = {"🏠 Home": "home", "📡 Live Monitor": "live", "📂 CSV Analysis": "csv"}
    selected_mode = nav_map[nav]

    if selected_mode != st.session_state.mode:
        if selected_mode != "home":
            ok, err = load_model()
            if not ok:
                st.error(f"Model load failed: {err}")
            else:
                st.session_state.mode = selected_mode
                st.rerun()
        else:
            st.session_state.mode = selected_mode
            st.rerun()

    st.markdown("---")
    st.markdown("#### System Status")

    # Engine status indicator
    engine_status = "🟢 Online" if st.session_state.model_loaded else "🟡 Standby"
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03); border-radius:10px; padding:12px; margin-bottom:8px;">
        <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">AI Engine</div>
        <div style="font-size:0.95rem; font-weight:600;">{engine_status}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03); border-radius:10px; padding:12px; margin-bottom:8px;">
        <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">Architecture</div>
        <div style="font-size:0.95rem; font-weight:600;">HybridFormer v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03); border-radius:10px; padding:12px; margin-bottom:8px;">
        <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">Timestamp</div>
        <div style="font-size:0.95rem; font-weight:600;">{datetime.now().strftime('%H:%M:%S')}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='text-align:center; color:#475569; font-size:0.72rem;'>NetShield NIDS v2.0<br>Powered by HybridFormer + SHAP + Groq LLM</p>", unsafe_allow_html=True)

# ======================================================
# HOME — HERO + MODE SELECTOR
# ======================================================
if st.session_state.mode == 'home':
    # Animated floating particles
    st.markdown("""
    <div class="particles">
        <div class="particle"></div><div class="particle"></div>
        <div class="particle"></div><div class="particle"></div>
        <div class="particle"></div><div class="particle"></div>
        <div class="particle"></div><div class="particle"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <div class="hero-title">NetShield NIDS</div>
        <div class="hero-sub">
            Enterprise-grade cloud network security powered by HybridFormer AI,
            explainable SHAP analytics, and LLM-driven threat intelligence.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    h1, h2 = st.columns(2, gap="large")

    with h1:
        st.markdown("""
<div class="mode-card">
<div class="mode-icon">📡</div>
<div class="mode-title">Live Cloud Monitor</div>
<div class="mode-desc">
Stream real-time network telemetry from your AWS EC2 cloud sensor.<br>
Auto-refresh every 5 seconds with threat highlighting and per-flow XAI deep dives.
</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶  Launch Live Monitor", use_container_width=True, key="btn_live"):
            ok, err = load_model()
            if ok:
                st.session_state.mode = 'live'
                st.rerun()
            else:
                st.error(f"Failed to load model: {err}")

    with h2:
        st.markdown("""
<div class="mode-card">
<div class="mode-icon">📂</div>
<div class="mode-title">CSV Forensic Analysis</div>
<div class="mode-desc">
Upload historical CICIDS-format traffic captures for batch classification.<br>
Get attack predictions, confidence scores, and granular SHAP explanations.
</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶  Open CSV Analysis", use_container_width=True, key="btn_csv"):
            ok, err = load_model()
            if ok:
                st.session_state.mode = 'csv'
                st.rerun()
            else:
                st.error(f"Failed to load model: {err}")

    # ---- Feature Highlights ----
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h3>🔒 Platform Capabilities</h3></div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("🧠", "HybridFormer AI", "CNN + Transformer + Graph Neural Network with attention-based fusion."),
        ("📊", "SHAP Explainability", "Game-theoretic feature attribution for mathematical transparency."),
        ("🤖", "LLM Intelligence", "Groq-powered plain-English threat reports with actionable remediation."),
        ("☁️", "Cloud-Native", "AWS EC2 sensor → Neon PostgreSQL → Streamlit SOC Dashboard.")
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; min-height:180px;">
                <div style="font-size:2.2rem; margin-bottom:10px;">{icon}</div>
                <div style="font-weight:700; font-size:0.95rem; margin-bottom:8px; color:#c7d2fe;">{title}</div>
                <div style="font-size:0.8rem; color:#94a3b8; line-height:1.55;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ======================================================
# LIVE MONITOR
# ======================================================
elif st.session_state.mode == 'live':
    st.markdown('<div class="section-header"><h3>📡 Live Cloud Network Monitor</h3></div>', unsafe_allow_html=True)

    # Status Bar
    st.markdown("""
    <div class="status-bar">
        <div class="status-dot"></div>
        <div class="status-text">Engine Active — Monitoring cloud sensor traffic in real-time</div>
    </div>
    """, unsafe_allow_html=True)

    c_ref1, c_ref2, c_ref3 = st.columns([1, 1, 2])
    with c_ref1:
        auto_refresh = st.checkbox("Auto-Refresh (5s)", value=True)
    with c_ref2:
        if st.button("🔄 Manual Refresh"):
            st.rerun()
    with c_ref3:
        if st.button("🗑️ Clear Traffic Logs"):
            try:
                db_url = os.environ.get("DATABASE_URL")
                if not db_url: db_url = st.secrets["DATABASE_URL"]
                if db_url.startswith("postgres://"): db_url = db_url.replace("postgres://", "postgresql://", 1)
                engine = create_engine(db_url)
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM logs"))
                st.success("✅ Database wiped clean!")
                time.sleep(1.5)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear logs: {e}")

    df, true_total, true_attacks = get_live_data()

    if df.empty and true_total == 0:
        st.info("⏳ No traffic logs yet. Make sure the NIDS engine is running on your AWS EC2 instance.")
    else:
        total_flows = true_total
        attacks     = true_attacks
        normals     = total_flows - attacks
        atk_pct     = attacks / total_flows * 100 if total_flows else 0

        # KPI Grid
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card info">
                <div class="kpi-label">Processed Flows</div>
                <div class="kpi-value info">{total_flows:,}</div>
            </div>
            <div class="kpi-card danger">
                <div class="kpi-label">Malicious Flows</div>
                <div class="kpi-value danger">{attacks:,}</div>
            </div>
            <div class="kpi-card safe">
                <div class="kpi-label">Normal Traffic</div>
                <div class="kpi-value safe">{normals:,}</div>
            </div>
            <div class="kpi-card warn">
                <div class="kpi-label">Attack Rate</div>
                <div class="kpi-value warn">{atk_pct:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Threat Level Gauge + Donut Chart Row ----
        gauge_col, donut_col = st.columns([1, 1])

        with gauge_col:
            # Dynamic Threat Level Gauge
            if atk_pct == 0:
                level, level_color, level_text = 'LOW', '#34d399', 'All Clear'
            elif atk_pct < 5:
                level, level_color, level_text = 'MODERATE', '#fbbf24', 'Elevated Activity'
            elif atk_pct < 20:
                level, level_color, level_text = 'HIGH', '#fb923c', 'Active Threats'
            else:
                level, level_color, level_text = 'CRITICAL', '#f87171', 'Under Attack'

            bar_width = max(atk_pct, 2)  # minimum visual width
            st.markdown(f"""
            <div class="threat-gauge">
                <div class="gauge-label">Threat Level</div>
                <div class="gauge-level" style="color:{level_color};">{level}</div>
                <div class="gauge-bar-bg">
                    <div class="gauge-bar-fill" style="width:{min(bar_width, 100)}%; background: linear-gradient(90deg, {level_color}, {level_color}88);"></div>
                </div>
                <div style="font-size:0.78rem; color:#64748b;">{level_text} — {atk_pct:.1f}% attack rate</div>
            </div>
            """, unsafe_allow_html=True)

        with donut_col:
            # Attack vs Normal Donut Chart
            fig_donut = go.Figure(go.Pie(
                labels=['Normal', 'Malicious'],
                values=[normals, attacks],
                hole=0.65,
                marker=dict(colors=['#34d399', '#f87171'], line=dict(width=0)),
                textinfo='percent',
                textfont=dict(size=13, color='#e2e8f0'),
                hoverinfo='label+value+percent'
            ))
            fig_donut.update_layout(
                height=220,
                showlegend=True,
                legend=dict(font=dict(color='#94a3b8', size=11), orientation='h', y=-0.1, x=0.5, xanchor='center'),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=10, b=30, l=10, r=10),
                annotations=[dict(text=f'<b>{total_flows:,}</b><br><span style="font-size:10px;color:#64748b">Total</span>',
                                  x=0.5, y=0.5, font=dict(size=18, color='#e2e8f0'), showarrow=False)]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # ---- Top Talkers (Source IPs) ----
        if 'src_ip' in df.columns and len(df) > 0:
            talkers_col, types_col = st.columns(2)

            with talkers_col:
                st.markdown('<div class="section-header"><h3>🌐 Top Source IPs</h3></div>', unsafe_allow_html=True)
                top_ips = df['src_ip'].value_counts().head(8).reset_index()
                top_ips.columns = ['IP', 'Flows']
                fig_ips = go.Figure(go.Bar(
                    x=top_ips['Flows'], y=top_ips['IP'], orientation='h',
                    marker=dict(color='#818cf8', line=dict(width=0)),
                    text=top_ips['Flows'], textposition='inside'
                ))
                fig_ips.update_layout(
                    height=280,
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8', family='Inter', size=11),
                    margin=dict(l=120, r=20, t=10, b=30),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', title=''),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.04)', title='', autorange='reversed')
                )
                st.plotly_chart(fig_ips, use_container_width=True)

            with types_col:
                st.markdown('<div class="section-header"><h3>⚡ Attack Type Breakdown</h3></div>', unsafe_allow_html=True)
                type_counts = df['attack_type'].value_counts().reset_index()
                type_counts.columns = ['Type', 'Count']
                type_colors = ['#34d399' if t == 'Normal Traffic' else '#f87171' for t in type_counts['Type']]
                fig_types = go.Figure(go.Bar(
                    x=type_counts['Count'], y=type_counts['Type'], orientation='h',
                    marker=dict(color=type_colors, line=dict(width=0)),
                    text=type_counts['Count'], textposition='inside'
                ))
                fig_types.update_layout(
                    height=280,
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8', family='Inter', size=11),
                    margin=dict(l=120, r=20, t=10, b=30),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', title=''),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.04)', title='', autorange='reversed')
                )
                st.plotly_chart(fig_types, use_container_width=True)

        # Traffic Timeline Chart
        if 'timestamp' in df.columns and len(df) > 1:
            st.markdown('<div class="section-header"><h3>📈 Traffic Timeline</h3></div>', unsafe_allow_html=True)
            timeline_df = df.copy()
            timeline_df['timestamp'] = pd.to_datetime(timeline_df['timestamp'], errors='coerce')
            timeline_df = timeline_df.dropna(subset=['timestamp'])
            timeline_df['is_attack'] = (timeline_df['attack_type'] != 'Normal Traffic').astype(int)
            timeline_df = timeline_df.sort_values('timestamp')

            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(
                x=timeline_df['timestamp'], y=timeline_df['confidence'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=timeline_df['is_attack'].map({0: '#34d399', 1: '#f87171'}),
                    opacity=0.7,
                    line=dict(width=0)
                ),
                text=timeline_df['attack_type'],
                hovertemplate='%{text}<br>Confidence: %{y:.2%}<br>Time: %{x}<extra></extra>'
            ))
            fig_timeline.update_layout(
                height=280,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8', family='Inter'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.04)', title=''),
                yaxis=dict(gridcolor='rgba(255,255,255,0.04)', title='Confidence', tickformat='.0%'),
                margin=dict(l=50, r=20, t=20, b=40),
                showlegend=False
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

        # Traffic Table
        st.markdown('<div class="section-header"><h3>📋 Recent Traffic Stream</h3></div>', unsafe_allow_html=True)
        display_df = df.drop(columns=['features_json'], errors='ignore')

        def hl_live(row):
            return ['background-color:rgba(248,113,113,0.12)']*len(row) \
                   if row['attack_type'] != 'Normal Traffic' else ['']*len(row)

        st.dataframe(display_df.style.apply(hl_live, axis=1), use_container_width=True, height=320)

        # XAI Section
        st.markdown('<div class="section-header"><h3>🧠 Explainable AI — Threat Deep Dive</h3></div>', unsafe_allow_html=True)
        attack_logs = df[df['attack_type'] != 'Normal Traffic']
        if attack_logs.empty:
            st.success("✅ No attacks detected in the current monitoring window. All traffic is classified as benign.")
        else:
            opts = attack_logs.apply(
                lambda x: f"ID {x['id']} | {x['src_ip']} → {x['dst_ip']} | {x['attack_type']} ({x['timestamp']})",
                axis=1
            ).tolist()
            sel = st.selectbox("Select a malicious flow to investigate:", opts, key="live_xai")
            if sel:
                sid = int(sel.split(" | ")[0].replace("ID ", ""))
                row = attack_logs[attack_logs['id'] == sid].iloc[0]
                st.markdown(f"""
                <div class="glass-card" style="border-left: 3px solid #f87171; margin-bottom: 16px;">
                    <strong>Flow:</strong> <code>{row['src_ip']} → {row['dst_ip']}</code> &nbsp;|&nbsp;
                    <strong>Classification:</strong> <span class="badge badge-critical">{row['attack_type']}</span> &nbsp;|&nbsp;
                    <strong>Confidence:</strong> <code>{row['confidence']*100:.1f}%</code>
                </div>
                """, unsafe_allow_html=True)

                if 'features_json' in row and pd.notnull(row['features_json']):
                    try:
                        X = np.array(json.loads(row['features_json'])).reshape(1, -1)
                        top_feats = shap_chart(X, row['attack_type'])
                        if top_feats:
                            ai_explain(
                                attack_type=row['attack_type'],
                                confidence=row['confidence'] * 100,
                                shap_data=top_feats,
                                src_ip=row['src_ip'],
                                dst_ip=row['dst_ip'],
                                btn_key=f"ai_live_{sid}"
                            )
                    except Exception as e:
                        st.error(f"XAI error: {e}")
                else:
                    st.warning("No feature vector stored for this entry. Re-run the updated engine.")

    if auto_refresh:
        time.sleep(5)
        st.rerun()

# ======================================================
# CSV BATCH ANALYSIS
# ======================================================
elif st.session_state.mode == 'csv':
    st.markdown('<div class="section-header"><h3>📂 CSV Forensic Analysis</h3></div>', unsafe_allow_html=True)

    st.markdown("""
<div class="upload-hint">
<strong>📎 Upload Instructions</strong><br><br>
Upload a CSV of network flows to classify each row using the HybridFormer AI engine.<br><br>
<strong>Accepted formats:</strong><br>
&nbsp;• <strong>CICIDS-style</strong>: original column names like <code>Destination Port</code>, <code>Flow Duration</code><br>
&nbsp;• <strong>Pre-processed</strong>: columns already matching the trained model's feature schema
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop your CSV file here", type=["csv"], key="csv_upload")

    if uploaded is not None:
        model         = st.session_state.model
        scaler        = st.session_state.scaler
        le            = st.session_state.le
        feature_names = st.session_state.feature_names
        explainer     = st.session_state.explainer

        try:
            raw_df = pd.read_csv(uploaded)
            raw_df.columns = raw_df.columns.str.strip()
            raw_df.replace([np.inf, -np.inf], np.nan, inplace=True)

            st.success(f"✅ Loaded **{uploaded.name}** — `{len(raw_df):,}` rows × `{len(raw_df.columns)}` columns")

            with st.expander("🔍 Column Diagnostics", expanded=False):
                matched = [f for f in feature_names if f in raw_df.columns]
                missing = [f for f in feature_names if f not in raw_df.columns]
                st.markdown(f"**Model expects:** `{len(feature_names)}` features")
                st.markdown(f"**Matched in CSV:** `{len(matched)}`  ✅")
                if missing:
                    st.warning(f"**Not found ({len(missing)}):** {missing[:10]}{'…' if len(missing)>10 else ''}")
                else:
                    st.success("All feature columns found — predictions should be accurate.")

            matched = [f for f in feature_names if f in raw_df.columns]
            if len(matched) == len(feature_names):
                features_df = raw_df[feature_names].copy()
                mode_label  = "pre-processed"
            else:
                rows_out = []
                for _, r in raw_df.iterrows():
                    fd = {}
                    for col in feature_names:
                        raw_val = r.get(col, None)
                        val = float(raw_val) if (raw_val is not None and not pd.isna(raw_val)) else 0.0
                        if col == 'Packet Length Variance':
                            std_v = r.get('Packet Length Std', 0)
                            val = float(std_v if not pd.isna(std_v) else 0) ** 2
                        fd[col] = val
                    rows_out.append(fd)
                features_df = pd.DataFrame(rows_out)
                mode_label  = f"partial match ({len(matched)}/{len(feature_names)} features)"

            features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
            st.caption(f"Feature mode: **{mode_label}**")

            with st.spinner(f"🔄 Classifying {len(features_df):,} rows with HybridFormer…"):
                X_scaled = scaler.transform(features_df)
                probs    = model.predict_proba(X_scaled)
                pred_idx = np.argmax(probs, axis=1)
                labels   = le.inverse_transform(pred_idx)
                confs    = probs[np.arange(len(probs)), pred_idx]

            results = raw_df.copy()
            results['Prediction']   = labels
            results['Confidence %'] = (confs * 100).round(2)
            results['Status']       = results['Prediction'].apply(
                lambda x: "⚠️ Attack" if x != 'Normal Traffic' else "🟢 Normal"
            )

            # KPIs
            st.markdown("---")
            total  = len(results)
            n_atk  = int((results['Prediction'] != 'Normal Traffic').sum())
            n_norm = total - n_atk
            atk_pct = n_atk / total * 100 if total else 0

            st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-card info">
                    <div class="kpi-label">Total Rows</div>
                    <div class="kpi-value info">{total:,}</div>
                </div>
                <div class="kpi-card danger">
                    <div class="kpi-label">Attacks Found</div>
                    <div class="kpi-value danger">{n_atk:,}</div>
                </div>
                <div class="kpi-card safe">
                    <div class="kpi-label">Normal</div>
                    <div class="kpi-value safe">{n_norm:,}</div>
                </div>
                <div class="kpi-card warn">
                    <div class="kpi-label">Attack Rate</div>
                    <div class="kpi-value warn">{atk_pct:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Distribution chart
            st.markdown('<div class="section-header"><h3>📊 Attack Distribution</h3></div>', unsafe_allow_html=True)
            ac = results['Prediction'].value_counts().reset_index()
            ac.columns = ['Type', 'Count']
            colors = ['#34d399' if t == 'Normal Traffic' else '#f87171' for t in ac['Type']]
            fig = go.Figure(go.Bar(
                x=ac['Type'], y=ac['Count'],
                marker_color=colors, text=ac['Count'], textposition='outside',
                marker=dict(line=dict(width=0))
            ))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8', family='Inter'), xaxis_title="", yaxis_title="Count",
                margin=dict(t=20, b=40),
                xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.04)')
            )
            st.plotly_chart(fig, use_container_width=True)

            # Full table
            st.markdown('<div class="section-header"><h3>📋 Row-by-Row Predictions</h3></div>', unsafe_allow_html=True)
            show_cols = ['Status', 'Prediction', 'Confidence %']
            for opt in ['src_ip', 'dst_ip', 'Label', 'Attack Type']:
                if opt in results.columns:
                    show_cols.insert(0, opt)

            def hl_csv(row):
                return ['background-color:rgba(248,113,113,0.10)']*len(row) \
                       if row.get('Prediction', '') != 'Normal Traffic' else ['']*len(row)

            preview = results[show_cols].head(1000)
            if len(results) > 1000:
                st.caption(f"Showing first 1,000 of {len(results):,} rows — download the full CSV below.")
            st.dataframe(preview.style.apply(hl_csv, axis=1),
                         use_container_width=True, height=350)

            st.download_button(
                "⬇️ Download Full Results CSV",
                data=results.to_csv(index=False).encode(),
                file_name=f"nids_results_{uploaded.name}",
                mime="text/csv"
            )

            # Per-row XAI
            st.markdown('<div class="section-header"><h3>🧠 Row-Level XAI — Explain Any Prediction</h3></div>', unsafe_allow_html=True)
            atk_rows = results[results['Prediction'] != 'Normal Traffic']

            if atk_rows.empty:
                st.success("🎉 No attacks found in this CSV!")
            else:
                row_opts = atk_rows.apply(
                    lambda r: f"Row {r.name} | {r['Prediction']} ({r['Confidence %']:.1f}%)",
                    axis=1
                ).tolist()
                sel_row = st.selectbox("Pick a row to investigate:", row_opts, key="csv_xai")
                if sel_row:
                    ridx  = int(sel_row.split(" | ")[0].replace("Row ", ""))
                    plabel = results.loc[ridx, 'Prediction']
                    pconf  = results.loc[ridx, 'Confidence %']
                    st.markdown(f"""
                    <div class="glass-card" style="border-left: 3px solid #f87171; margin-bottom: 16px;">
                        <strong>Row {ridx}</strong> &nbsp;|&nbsp;
                        <span class="badge badge-critical">{plabel}</span>
                        &nbsp;|&nbsp; Confidence: <code>{pconf:.1f}%</code>
                    </div>
                    """, unsafe_allow_html=True)
                    try:
                        top_feats = shap_chart(X_scaled[ridx].reshape(1, -1), f"Row {ridx} — {plabel}")
                        if top_feats:
                            ai_explain(
                                attack_type=plabel,
                                confidence=pconf,
                                shap_data=top_feats,
                                btn_key=f"ai_csv_{ridx}"
                            )
                    except Exception as e:
                        st.error(f"XAI error: {e}")

        except Exception as e:
            st.error(f"❌ Failed to process CSV: {e}")
