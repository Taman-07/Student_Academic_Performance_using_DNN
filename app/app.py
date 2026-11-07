"""
Student Performance Predictor - Streamlit app
Run from the project root:   streamlit run app/app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

# ======================================================================
# Styling (translucent surfaces so it works in light and dark themes)
# ======================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp {font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;}
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1280px;}
footer, #MainMenu, .stAppDeployButton {visibility: hidden; display: none;}
header[data-testid="stHeader"] {background: transparent;}

/* ---------- hero ---------- */
.hero {position: relative; overflow: hidden; padding: 1.8rem 2rem; border-radius: 20px; color: #fff;
       background: linear-gradient(120deg, #0f766e 0%, #155e75 50%, #3730a3 100%);
       box-shadow: 0 10px 30px rgba(15,118,110,.25); margin-bottom: 1.4rem;}
.hero:after {content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px;
             border-radius: 50%; background: rgba(255,255,255,.08);}
.hero h1 {margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.02em; color: #fff;}
.hero p {margin: .5rem 0 1rem 0; max-width: 720px; opacity: .92; font-size: 1.02rem; line-height: 1.5; color: #fff;}
.chips {display: flex; flex-wrap: wrap; gap: .5rem;}
.chip {padding: .3rem .75rem; border-radius: 999px; font-size: .82rem; font-weight: 600;
       background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28); color: #fff;}

/* ---------- cards ---------- */
.card {padding: 1.25rem 1.35rem; border-radius: 18px; border: 1px solid rgba(128,128,128,.22);
       background: rgba(128,128,128,.06); box-shadow: 0 2px 10px rgba(0,0,0,.04);}
.section-title {font-size: 1.05rem; font-weight: 700; margin: 0 0 .2rem 0;}
.muted {opacity: .68; font-size: .85rem;}
.eyebrow {text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; font-weight: 700; opacity: .6;}

/* ---------- result ---------- */
.result-top {display: flex; align-items: center; gap: 1.2rem;}
.ring-wrap {position: relative; width: 132px; height: 132px; flex: none;}
.ring {position: absolute; inset: 0; border-radius: 50%;
       background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(128,128,128,.22) 0);
       -webkit-mask: radial-gradient(farthest-side, #0000 calc(100% - 13px), #000 calc(100% - 12px));
       mask: radial-gradient(farthest-side, #0000 calc(100% - 13px), #000 calc(100% - 12px));}
.ring-text {position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;}
.ring-val {font-size: 1.9rem; font-weight: 800; line-height: 1;}
.ring-sub {font-size: .7rem; opacity: .65; margin-top: 2px; text-transform: uppercase; letter-spacing: .06em;}
.band {font-size: 2.3rem; font-weight: 800; line-height: 1.05; letter-spacing: -.02em;}
.badge {display: inline-block; margin-top: .45rem; padding: .22rem .7rem; border-radius: 999px;
        font-size: .78rem; font-weight: 700; color: #fff;}

.bar-row {display: flex; align-items: center; gap: .7rem; margin: .5rem 0;}
.bar-label {width: 76px; font-weight: 600; font-size: .88rem;}
.bar-track {flex: 1; height: 12px; border-radius: 8px; background: rgba(128,128,128,.2); overflow: hidden;}
.bar-fill {height: 100%; border-radius: 8px;}
.bar-pct {width: 46px; text-align: right; font-size: .88rem; font-weight: 600; font-variant-numeric: tabular-nums;}

.tip {padding: .9rem 1.05rem; border-radius: 14px; margin-top: .9rem; border-left: 5px solid;
      background: rgba(128,128,128,.08); font-size: .92rem; line-height: 1.5;}
.obs {display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .6rem;}
.obs span {padding: .3rem .7rem; border-radius: 10px; font-size: .8rem; font-weight: 600; border: 1px solid;}
.obs .good {background: rgba(16,185,129,.12); border-color: rgba(16,185,129,.45);}
.obs .warn {background: rgba(245,158,11,.12); border-color: rgba(245,158,11,.5);}
.obs .info {background: rgba(99,102,241,.12); border-color: rgba(99,102,241,.45);}

/* ---------- sidebar & tabs ---------- */
section[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
.side-brand {font-size: 1.15rem; font-weight: 800; letter-spacing: -.01em;}
.kpi {display: flex; justify-content: space-between; align-items: baseline; padding: .55rem 0;
      border-bottom: 1px dashed rgba(128,128,128,.25);}
.kpi:last-child {border-bottom: 0;}
.kpi b {font-size: 1.15rem;}
.stTabs [data-baseweb="tab-list"] {gap: .35rem;}
.stTabs [data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: .55rem 1rem; font-weight: 600;}
.stButton > button {border-radius: 10px; font-weight: 600;}

/* keep the result panel visible while scrolling the inputs */
div[data-testid="stColumn"]:has(.sticky-marker), div[data-testid="column"]:has(.sticky-marker) {
    position: sticky; top: 3.5rem; align-self: flex-start;}

.footer {text-align: center; margin-top: 2rem; padding-top: 1rem; font-size: .8rem; opacity: .6;
         border-top: 1px solid rgba(128,128,128,.2);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(s: str):
    """Collapse indentation/newlines so Markdown never treats the HTML as a code block."""
    st.markdown(" ".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def html_str(s: str) -> str:
    return " ".join(line.strip() for line in s.splitlines())


BAND_STYLE = {
    "Low":     {"color": "#ef4444", "emoji": "🔴", "range": "Final grade 0 - 9"},
    "Average": {"color": "#f59e0b", "emoji": "🟡", "range": "Final grade 10 - 13"},
    "High":    {"color": "#10b981", "emoji": "🟢", "range": "Final grade 14 - 20"},
}
BAND_ADVICE = {
    "Low": "Likely to finish in the lower band. Early academic support such as tutoring or regular check-ins "
           "would help most. Past failures, absences and study time are the first levers to look at.",
    "Average": "Likely to finish in the middle band. Small gains in study habits and attendance "
               "could move this student into the High band.",
    "High": "Likely to finish in the top band. Keeping attendance and study habits steady is the main thing to protect.",
}

# ======================================================================
# Load artifacts saved by the notebook
# ======================================================================
HERE = Path(__file__).resolve().parent
MODEL_DIR_CANDIDATES = [HERE.parent / "models", HERE.parent / "notebooks" / "models", Path.cwd() / "models"]


def find_model_dir():
    for d in MODEL_DIR_CANDIDATES:
        if (d / "model_metadata.json").exists():
            return d
    return None


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model_dir = find_model_dir()
    if model_dir is None:
        return None
    with open(model_dir / "model_metadata.json") as f:
        meta = json.load(f)
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    model_file = model_dir / Path(meta["model_path"]).name
    if meta["model_kind"] == "sklearn":
        model = joblib.load(model_file)
    else:
        import tensorflow as tf  # only needed if the winner was the DNN
        model = tf.keras.models.load_model(model_file)
    return meta, preprocessor, model


def predict(meta, preprocessor, model, student: dict):
    row = pd.DataFrame([student]).reindex(columns=meta["input_columns"])
    x = preprocessor.transform(row)
    x = np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)
    probs = model.predict_proba(x)[0] if meta["model_kind"] == "sklearn" else model.predict(x, verbose=0)[0]
    idx_to_label = {v: k for k, v in meta["label_mapping"].items()}
    return {idx_to_label[i]: float(p) for i, p in enumerate(probs)}


def global_importance(meta, preprocessor, model):
    """Feature importance per ORIGINAL input column (Random Forest only)."""
    if meta["model_kind"] != "sklearn" or not hasattr(model, "feature_importances_"):
        return None
    names = [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]
    cats = meta["categorical_features"]

    def original(n):
        if n in meta["numerical_features"]:
            return n
        m = [c for c in cats if n.startswith(c + "_")]
        return max(m, key=len) if m else n

    return pd.Series(model.feature_importances_, index=names).groupby(original).sum().sort_values(ascending=False)


def observations(s: dict):
    """Simple rule-based notes about the inputs (not model explanations)."""
    out = []
    delta = s["G2"] - s["G1"]
    if delta >= 2:
        out.append(("good", f"Grades improving (G1 {s['G1']} → G2 {s['G2']})"))
    elif delta <= -2:
        out.append(("warn", f"Grades declining (G1 {s['G1']} → G2 {s['G2']})"))
    if s["failures"] > 0:
        out.append(("warn", f"{s['failures']} past failure{'s' if s['failures'] > 1 else ''}"))
    else:
        out.append(("good", "No past failures"))
    if s["absences"] >= 10:
        out.append(("warn", f"High absences ({s['absences']})"))
    elif s["absences"] <= 2:
        out.append(("good", "Excellent attendance"))
    if s["studytime"] <= 1:
        out.append(("warn", "Low weekly study time"))
    elif s["studytime"] >= 3:
        out.append(("good", "Strong study routine"))
    if s["higher"] == "no":
        out.append(("warn", "Not aiming for higher education"))
    if s["G2"] < 10:
        out.append(("info", "Current grade is below 10"))
    return out[:6]


artifacts = load_artifacts()

# ======================================================================
# Hero
# ======================================================================
if artifacts is None:
    html("""<div class="hero"><h1>🎓 Student Performance Predictor</h1>
            <p>Predict a student's final-grade band from academic, family and lifestyle data.</p></div>""")
    st.error(
        "Could not find `model_metadata.json`. Run the notebook to the end so it saves the `models/` folder, "
        "then place that folder next to `app/` (or leave it in `notebooks/models`)."
    )
    st.stop()

meta, preprocessor, model = artifacts
test_acc = meta["test_metrics"].get("Accuracy")

chips = [f"🧠 {meta['final_model']}", f"📊 {len(meta['input_columns'])} input features", "📚 UCI Student Performance data"]
if test_acc is not None:
    chips.insert(1, f"🎯 {test_acc:.1f}% test accuracy")
html(f"""
<div class="hero">
  <h1>🎓 Student Performance Predictor</h1>
  <p>Enter a student's details and the model predicts whether their <b>final grade (G3)</b> will land in the
  <b>Low</b>, <b>Average</b> or <b>High</b> performance band. The result updates instantly as you change any input.</p>
  <div class="chips">{''.join(f'<span class="chip">{c}</span>' for c in chips)}</div>
</div>
""")

# ======================================================================
# State and presets
# ======================================================================
DEFAULTS = dict(
    G1=11, G2=11, failures=0, absences=4, studytime=2, schoolsup="no", paid="no", higher="yes",
    school="GP", sex="F", age=17, address="U", reason="course", traveltime=1, activities="no", internet="yes",
    famsize="GT3", Pstatus="T", guardian="mother", famsup="yes", Medu=3, Fedu=3, Mjob="other", Fjob="other",
    famrel=4, freetime=3, goout=3, health=4, romantic="no", Dalc=1, Walc=1, nursery="yes",
)
PRESETS = {
    "Typical": {},
    "Struggling": dict(G1=6, G2=5, failures=2, absences=14, studytime=1, higher="no", goout=5, Walc=3),
    "Strong": dict(G1=16, G2=17, failures=0, absences=1, studytime=4, higher="yes", goout=2, Walc=1),
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset(name):
    for k, v in {**DEFAULTS, **PRESETS[name]}.items():
        st.session_state[k] = v


# ======================================================================
# Sidebar
# ======================================================================
with st.sidebar:
    html('<div class="side-brand">🎓 Student Predictor</div><div class="muted">Academic performance analytics</div>')
    st.write("")
    st.markdown("**Quick examples**")
    for name in PRESETS:
        st.button(f"{name} student", key=f"preset_{name}", on_click=apply_preset, args=(name,), use_container_width=True)
    st.caption("Load an example, then adjust any value.")
    st.divider()
    st.markdown("**Model performance**")
    rows = ""
    if test_acc is not None:
        rows += f'<div class="kpi"><span class="muted">Test accuracy</span><b>{test_acc:.1f}%</b></div>'
    rows += f'<div class="kpi"><span class="muted">Cross-validated</span><b>{meta["cv_accuracy_mean"]:.1f}%</b></div>'
    rows += f'<div class="kpi"><span class="muted">Classes</span><b>{len(meta["label_mapping"])}</b></div>'
    html(rows)
    st.caption("Trained on the UCI Student Performance dataset (Mathematics, 395 students).")

# ======================================================================
# Main layout: inputs (left) | live result (right)
# ======================================================================
YN = ["yes", "no"]
LEVEL = [1, 2, 3, 4, 5]
edu_labels = {0: "None", 1: "Primary", 2: "5th-9th grade", 3: "Secondary", 4: "Higher"}
jobs = ["teacher", "health", "services", "at_home", "other"]

left, right = st.columns([1.55, 1], gap="large")

with left:
    html('<div class="section-title">Student details</div><div class="muted">Fill in each tab. Defaults are a typical student.</div>')
    st.write("")
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Academic", "🧑 Profile", "👪 Family", "🎯 Lifestyle"])

    with tab1:
        c1, c2 = st.columns(2)
        c1.slider("G1 - first period grade (0-20)", 0, 20, key="G1")
        c2.slider("G2 - second period grade (0-20)", 0, 20, key="G2")
        c1, c2 = st.columns(2)
        c1.selectbox("Past class failures", [0, 1, 2, 3], key="failures")
        c2.number_input("Absences", 0, 100, key="absences")
        c1, c2 = st.columns(2)
        c1.selectbox("Weekly study time", [1, 2, 3, 4], key="studytime",
                     format_func=lambda x: {1: "< 2 hours", 2: "2-5 hours", 3: "5-10 hours", 4: "> 10 hours"}[x])
        c2.selectbox("Wants higher education", YN, key="higher")
        c1, c2 = st.columns(2)
        c1.selectbox("Extra school support", YN, key="schoolsup")
        c2.selectbox("Paid extra classes", YN, key="paid")

    with tab2:
        c1, c2 = st.columns(2)
        c1.selectbox("School", ["GP", "MS"], key="school")
        c2.selectbox("Sex", ["F", "M"], key="sex")
        c1, c2 = st.columns(2)
        c1.slider("Age", 15, 22, key="age")
        c2.selectbox("Home address", ["U", "R"], key="address", format_func=lambda x: "Urban" if x == "U" else "Rural")
        c1, c2 = st.columns(2)
        c1.selectbox("Reason for choosing school", ["course", "home", "reputation", "other"], key="reason")
        c2.selectbox("Travel time to school", [1, 2, 3, 4], key="traveltime",
                     format_func=lambda x: {1: "< 15 min", 2: "15-30 min", 3: "30-60 min", 4: "> 1 hour"}[x])
        c1, c2 = st.columns(2)
        c1.selectbox("Extra-curricular activities", YN, key="activities")
        c2.selectbox("Internet at home", YN, key="internet")

    with tab3:
        c1, c2 = st.columns(2)
        c1.selectbox("Family size", ["GT3", "LE3"], key="famsize",
                     format_func=lambda x: "More than 3" if x == "GT3" else "3 or fewer")
        c2.selectbox("Parents live", ["T", "A"], key="Pstatus", format_func=lambda x: "Together" if x == "T" else "Apart")
        c1, c2 = st.columns(2)
        c1.selectbox("Guardian", ["mother", "father", "other"], key="guardian")
        c2.selectbox("Family educational support", YN, key="famsup")
        c1, c2 = st.columns(2)
        c1.selectbox("Mother's education", list(edu_labels), key="Medu", format_func=edu_labels.get)
        c2.selectbox("Father's education", list(edu_labels), key="Fedu", format_func=edu_labels.get)
        c1, c2 = st.columns(2)
        c1.selectbox("Mother's job", jobs, key="Mjob")
        c2.selectbox("Father's job", jobs, key="Fjob")

    with tab4:
        st.caption("Scale: 1 = very low, 5 = very high")
        c1, c2 = st.columns(2)
        c1.select_slider("Family relationship quality", LEVEL, key="famrel")
        c2.select_slider("Free time after school", LEVEL, key="freetime")
        c1, c2 = st.columns(2)
        c1.select_slider("Going out with friends", LEVEL, key="goout")
        c2.select_slider("Current health", LEVEL, key="health")
        c1, c2 = st.columns(2)
        c1.select_slider("Workday alcohol use", LEVEL, key="Dalc")
        c2.select_slider("Weekend alcohol use", LEVEL, key="Walc")
        c1, c2 = st.columns(2)
        c1.selectbox("In a relationship", YN, key="romantic")
        c2.selectbox("Attended nursery school", YN, key="nursery")

student = {k: st.session_state[k] for k in DEFAULTS}
missing = [c for c in meta["input_columns"] if c not in student]
if missing:
    st.error(f"The saved model expects columns this form does not provide: {missing}")
    st.stop()

probs = predict(meta, preprocessor, model, student)
label = max(probs, key=probs.get)
style = BAND_STYLE[label]
conf = probs[label] * 100

with right:
    html('<span class="sticky-marker"></span>')
    bars = ""
    for name in ["Low", "Average", "High"]:
        p = probs.get(name, 0) * 100
        bars += (
            f'<div class="bar-row"><div class="bar-label">{name}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{p:.1f}%;background:{BAND_STYLE[name]["color"]}"></div></div>'
            f'<div class="bar-pct">{p:.0f}%</div></div>'
        )
    obs = "".join(f'<span class="{kind}">{text}</span>' for kind, text in observations(student))
    html(f"""
    <div class="card" style="border-color:{style['color']}66">
      <div class="eyebrow">Predicted performance</div>
      <div class="result-top" style="margin-top:.7rem">
        <div class="ring-wrap">
          <div class="ring" style="--p:{conf:.0f};--c:{style['color']}"></div>
          <div class="ring-text"><div class="ring-val">{conf:.0f}%</div><div class="ring-sub">confidence</div></div>
        </div>
        <div>
          <div class="band" style="color:{style['color']}">{label}</div>
          <span class="badge" style="background:{style['color']}">{style['range']}</span>
        </div>
      </div>
      <div style="margin-top:1.2rem" class="eyebrow">Probability by band</div>
      {bars}
      <div class="tip" style="border-color:{style['color']}"><b>What this means.</b> {BAND_ADVICE[label]}</div>
      <div style="margin-top:1rem" class="eyebrow">Profile highlights</div>
      <div class="obs">{obs}</div>
    </div>
    """)
    st.caption("A statistical estimate from one school's historical data. Use it to prioritise support, not as a final decision.")

# ======================================================================
# Explanations
# ======================================================================
st.write("")
t1, t2 = st.tabs(["📈 What drives the predictions?", "⚙️ How the model works"])

with t1:
    imp = global_importance(meta, preprocessor, model)
    if imp is None:
        st.info("Feature importance is shown when the final model is the Random Forest.")
    else:
        top = imp.head(8)
        mx = float(top.max())
        rows = "".join(
            f'<div class="bar-row"><div class="bar-label" style="width:110px">{name}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{v / mx * 100:.1f}%;background:#0f766e"></div></div>'
            f'<div class="bar-pct">{v * 100:.1f}%</div></div>'
            for name, v in top.items()
        )
        html(f'<div class="card"><div class="section-title">Top 8 features across all students</div>{rows}</div>')
        st.caption("Share of the model's decision attributed to each input. Earlier-period grades (G1, G2) usually "
                   "dominate because the final grade closely follows them.")

with t2:
    c1, c2, c3 = st.columns(3)
    cards = [
        ("1 · Inputs", f"{len(meta['input_columns'])} demographic, academic, family and lifestyle attributes per student."),
        ("2 · Preprocessing", "Missing values filled, numbers scaled, categories one-hot encoded with a saved pipeline."),
        ("3 · Prediction", f"{meta['final_model']} outputs a probability for Low, Average and High; the highest wins."),
    ]
    for col, (title, body) in zip((c1, c2, c3), cards):
        with col:
            html(f'<div class="card"><div class="section-title">{title}</div><div class="muted" style="font-size:.9rem">{body}</div></div>')
    st.write("")
    st.markdown(
        "The model was compared against a Deep Neural Network (baseline and tuned) and a Random Forest "
        "(baseline and tuned), and the best performer on the held-out test set was selected."
    )

html('<div class="footer">Built with Streamlit · Data: UCI Student Performance (Mathematics) · Predictions are estimates, not decisions</div>')
