import streamlit as st
import sqlite3
import pandas as pd
import json
import joblib
import shap
import plotly.graph_objects as go
import time
import os
import numpy as np
import warnings
from openai import OpenAI

warnings.filterwarnings("ignore")
pd.set_option("styler.render.max_elements", 5_000_000)

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="NetShield NIDS", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
.stApp { background-color: #0E1117; color: #FAFAFA; font-family: 'Inter', sans-serif; }
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    border-radius: 12px;
    padding: 22px 20px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-4px); }
.metric-value { font-size: 2.4rem; font-weight: 700; margin: 8px 0; }
.metric-label { font-size: 0.85rem; color: #A0AEC0; text-transform: uppercase; letter-spacing: 1px; }
.danger { color: #FC8181; }
.safe   { color: #68D391; }
.info   { color: #63B3ED; }
.warn   { color: #F6E05E; }
.mode-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.mode-card:hover { transform: translateY(-6px); border-color: rgba(255,255,255,0.25); }
.mode-icon { font-size: 3rem; margin-bottom: 12px; }
.mode-title { font-size: 1.4rem; font-weight: 700; margin-bottom: 6px; }
.mode-desc  { font-size: 0.9rem; color: #A0AEC0; }
.upload-hint {
    background: rgba(99,179,237,0.07);
    border: 1px dashed rgba(99,179,237,0.35);
    border-radius: 10px;
    padding: 14px 18px;
    color: #A0AEC0;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# SESSION STATE DEFAULTS
# ======================================================
if 'mode' not in st.session_state:
    st.session_state.mode = 'home'        # 'home' | 'live' | 'csv'
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
    """Load model into session state (called only when a mode is selected)."""
    if st.session_state.model_loaded:
        return True, ""
    try:
        with st.spinner("⏳ Loading NIDS model… (first load may take ~20s)"):
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
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query(f"SELECT * FROM logs ORDER BY timestamp DESC LIMIT {limit}", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def metric_card(label, value, css_class):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value {css_class}">{value}</div></div>',
        unsafe_allow_html=True
    )

def shap_chart(X_array, row_label=""):
    """Compute SHAP values, render chart, and return top features for AI explanation."""
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
        sdf_plot['Color'] = sdf_plot['SHAP'].apply(lambda x: '#FC8181' if x > 0 else '#63B3ED')
        sdf_plot['Text']  = sdf_plot['Raw'].apply(lambda x: f"Val: {x:.2f}")

        fig = go.Figure(go.Bar(
            x=sdf_plot['SHAP'], y=sdf_plot['Feature'], orientation='h',
            marker_color=sdf_plot['Color'], text=sdf_plot['Text'], textposition='inside'
        ))
        fig.update_layout(
            title=f"Feature Contributions (SHAP){' — ' + row_label if row_label else ''}",
            xaxis_title="SHAP  (🔴 → Attack   🔵 → Normal)",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FAFAFA'), height=480,
            margin=dict(l=200, r=20, t=50, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Return top features for AI explanation
        return top_features[['Feature', 'SHAP', 'Raw']].to_dict('records')

# ======================================================
# AI-POWERED EXPLANATION (Groq)
# ======================================================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "YOUR_API_KEY_HERE")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

def ai_explain(attack_type, confidence, shap_data, src_ip="", dst_ip="", btn_key="ai_btn"):
    """Show an 'Ask AI' button; on click, call Groq API."""
    if st.button("🤖 Ask AI to Explain This Threat", key=btn_key):
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
            with st.spinner("🧠 AI is analyzing the threat…"):
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
            st.markdown("#### 🤖 AI Threat Analysis")
            st.markdown(explanation)
        except Exception as e:
            st.error(f"AI explanation error: {e}")

def stop_button():
    if st.button("⏹ Stop & Return to Home", key="stop_btn"):
        st.session_state.mode = 'home'
        st.rerun()

# ======================================================
# HEADER
# ======================================================
st.title("🛡️ NetShield NIDS Engine")
st.markdown("Network Intrusion Detection System — powered by Machine Learning & Explainable AI")
st.markdown("---")

# ======================================================
# HOME — MODE SELECTOR
# ======================================================
if st.session_state.mode == 'home':
    st.markdown("### Choose a Mode")
    st.markdown("<br>", unsafe_allow_html=True)

    h1, h2 = st.columns(2, gap="large")

    with h1:
        st.markdown("""
<div class="mode-card">
<div class="mode-icon">📡</div>
<div class="mode-title">Live Monitor</div>
<div class="mode-desc">
  Stream real-time network traffic through the NIDS engine.<br>
  Auto-refreshes every 5 seconds and highlights active threats.
</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶ Start Live Monitor", use_container_width=True, key="btn_live"):
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
<div class="mode-title">CSV Batch Analysis</div>
<div class="mode-desc">
  Upload a CICIDS-format CSV and classify every row.<br>
  Get attack type predictions, confidence scores, and per-row SHAP explanations.
</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶ Open CSV Analysis", use_container_width=True, key="btn_csv"):
            ok, err = load_model()
            if ok:
                st.session_state.mode = 'csv'
                st.rerun()
            else:
                st.error(f"Failed to load model: {err}")

# ======================================================
# LIVE MONITOR
# ======================================================
elif st.session_state.mode == 'live':
    col_title, col_stop = st.columns([5, 1])
    with col_title:
        st.subheader("📡 Live Network Monitor")
    with col_stop:
        stop_button()

    c_ref1, c_ref2 = st.columns([1, 5])
    with c_ref1:
        auto_refresh = st.checkbox("Auto-Refresh (5s)", value=True)
    with c_ref2:
        if st.button("Manual Refresh 🔄"):
            st.rerun()

    df = get_live_data()

    if df.empty:
        st.info("No traffic logs yet. Make sure the NIDS engine is running with sudo.")
    else:
        total_flows = len(df)
        attacks     = int((df['attack_type'] != 'Normal Traffic').sum())
        normals     = total_flows - attacks
        atk_pct     = attacks / total_flows * 100 if total_flows else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Processed Flows", total_flows, "info")
        with c2: metric_card("Malicious Flows", attacks, "danger")
        with c3: metric_card("Normal Traffic",  normals, "safe")
        with c4: metric_card("Attack Rate", f"{atk_pct:.1f}%", "warn")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Recent Traffic Stream")
        display_df = df.drop(columns=['features_json'], errors='ignore')

        def hl_live(row):
            return ['background-color:rgba(252,129,129,0.15)']*len(row) \
                   if row['attack_type'] != 'Normal Traffic' else ['']*len(row)

        st.dataframe(display_df.style.apply(hl_live, axis=1), use_container_width=True, height=300)

        st.markdown("---")
        st.markdown("#### 🧠 Explainable AI — Threat Deep Dive")
        attack_logs = df[df['attack_type'] != 'Normal Traffic']
        if attack_logs.empty:
            st.success("✅ No attacks detected in the recent window.")
        else:
            opts = attack_logs.apply(
                lambda x: f"ID {x['id']} | {x['src_ip']} → {x['dst_ip']} | {x['attack_type']} ({x['timestamp']})",
                axis=1
            ).tolist()
            sel = st.selectbox("Select a malicious flow to explain:", opts, key="live_xai")
            if sel:
                sid = int(sel.split(" | ")[0].replace("ID ", ""))
                row = attack_logs[attack_logs['id'] == sid].iloc[0]
                st.markdown(
                    f"**Flow:** `{row['src_ip']} → {row['dst_ip']}` &nbsp;|&nbsp; "
                    f"**Type:** <span style='color:#FC8181'>**{row['attack_type']}**</span> &nbsp;|&nbsp; "
                    f"**Confidence:** `{row['confidence']*100:.2f}%`",
                    unsafe_allow_html=True
                )
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
    col_title, col_stop = st.columns([5, 1])
    with col_title:
        st.subheader("📂 CSV Batch Analysis")
    with col_stop:
        stop_button()

    st.markdown("""
<div class="upload-hint">
Upload a CSV of network flows to classify each row using the same NIDS model as live traffic.<br><br>
<b>Accepted formats:</b><br>
&nbsp;• <b>CICIDS-style</b>: original column names like <code>Destination Port</code>, <code>Flow Duration</code> …<br>
&nbsp;• <b>Pre-processed</b>: columns already matching the trained model's feature names.
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

            # Strip whitespace from ALL column names (CICIDS CSVs often have leading spaces)
            raw_df.columns = raw_df.columns.str.strip()

            # Replace Inf values which CICIDS CSVs commonly contain
            raw_df.replace([np.inf, -np.inf], np.nan, inplace=True)

            st.success(f"✅ Loaded **{uploaded.name}** — `{len(raw_df):,}` rows × `{len(raw_df.columns)}` columns")

            # Diagnostics expander
            with st.expander("🔍 Column Diagnostics (click to inspect)", expanded=False):
                matched = [f for f in feature_names if f in raw_df.columns]
                missing = [f for f in feature_names if f not in raw_df.columns]
                st.markdown(f"**Model expects:** `{len(feature_names)}` features")
                st.markdown(f"**Matched in CSV:** `{len(matched)}`  ✅")
                if missing:
                    st.warning(f"**Not found in CSV ({len(missing)}):** {missing[:10]}{'…' if len(missing)>10 else ''}")
                    st.markdown("These will default to **0** which may bias predictions toward Normal Traffic.")
                else:
                    st.success("All feature columns found — predictions should be accurate.")

            # Build feature matrix
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

            with st.spinner(f"Classifying {len(features_df):,} rows…"):
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

            # ---- KPIs ----
            st.markdown("---")
            total  = len(results)
            n_atk  = int((results['Prediction'] != 'Normal Traffic').sum())
            n_norm = total - n_atk
            atk_pct = n_atk / total * 100 if total else 0

            k1, k2, k3, k4 = st.columns(4)
            with k1: metric_card("Total Rows",    f"{total:,}",   "info")
            with k2: metric_card("Attacks Found", f"{n_atk:,}",   "danger")
            with k3: metric_card("Normal",        f"{n_norm:,}",  "safe")
            with k4: metric_card("Attack Rate",   f"{atk_pct:.1f}%", "warn")

            # ---- Distribution chart ----
            st.markdown("<br>", unsafe_allow_html=True)
            ac = results['Prediction'].value_counts().reset_index()
            ac.columns = ['Type', 'Count']
            colors = ['#68D391' if t == 'Normal Traffic' else '#FC8181' for t in ac['Type']]
            fig = go.Figure(go.Bar(
                x=ac['Type'], y=ac['Count'],
                marker_color=colors, text=ac['Count'], textposition='outside'
            ))
            fig.update_layout(
                title="Attack Type Distribution",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FAFAFA'), xaxis_title="", yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)

            # ---- Full table ----
            st.markdown("#### 📋 Row-by-Row Predictions")
            show_cols = ['Status', 'Prediction', 'Confidence %']
            for opt in ['src_ip', 'dst_ip', 'Label', 'Attack Type']:
                if opt in results.columns:
                    show_cols.insert(0, opt)

            def hl_csv(row):
                return ['background-color:rgba(252,129,129,0.12)']*len(row) \
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

            # ---- Per-row XAI ----
            st.markdown("---")
            st.markdown("#### 🧠 Row-Level XAI — Explain Any Prediction")
            atk_rows = results[results['Prediction'] != 'Normal Traffic']

            if atk_rows.empty:
                st.success("🎉 No attacks found in this CSV!")
            else:
                row_opts = atk_rows.apply(
                    lambda r: f"Row {r.name} | {r['Prediction']} ({r['Confidence %']:.1f}%)",
                    axis=1
                ).tolist()
                sel_row = st.selectbox("Pick a row to explain:", row_opts, key="csv_xai")
                if sel_row:
                    ridx  = int(sel_row.split(" | ")[0].replace("Row ", ""))
                    plabel = results.loc[ridx, 'Prediction']
                    pconf  = results.loc[ridx, 'Confidence %']
                    st.markdown(
                        f"**Row {ridx}** &nbsp;|&nbsp; <span style='color:#FC8181'>**{plabel}**</span>"
                        f" &nbsp;|&nbsp; Confidence: `{pconf:.1f}%`",
                        unsafe_allow_html=True
                    )
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
