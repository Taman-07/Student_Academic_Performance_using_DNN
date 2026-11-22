import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

ART = Path(__file__).parent / "artifacts"

# ------------------------------------------------------------------
# Custom styling (only affects the look, not the logic)
# ------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}

/* intro banner above the form */
.intro-box {
    background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    color: white; padding: 22px 28px; border-radius: 18px; margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(79, 70, 229, 0.30);
}
.intro-box h2 {margin: 0 0 4px 0; color: white; font-size: 1.6rem;}
.intro-box p {margin: 0; opacity: 0.92; font-size: 0.98rem;}

/* section cards */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.section-title) {
    border-radius: 18px;
    border: 1px solid rgba(124, 58, 237, 0.25);
    background: rgba(124, 58, 237, 0.04);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
    padding: 6px 10px 2px 10px;
}
.section-title {font-size: 1.25rem; font-weight: 700; margin-bottom: 0;}
.section-sub {font-size: 0.88rem; opacity: 0.65; margin-bottom: 14px;}

/* nicer inputs */
div[data-baseweb="select"] > div {border-radius: 10px;}
div[data-testid="stSlider"] label p, div[data-testid="stSelectbox"] label p {font-weight: 600;}

/* predict button */
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #4f46e5, #db2777);
    color: white; border: none; border-radius: 14px;
    padding: 0.8rem 1rem; font-size: 1.15rem; font-weight: 700;
    box-shadow: 0 6px 18px rgba(219, 39, 119, 0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-2px); box-shadow: 0 10px 24px rgba(219, 39, 119, 0.45); color: white;
}
</style>
""", unsafe_allow_html=True)

# Which feature goes into which section of the form
SECTIONS = [
    ("📚 Academic Information", "Grades, study habits and school details",
     ["G1", "G2", "absences", "failures", "studytime", "higher", "paid",
      "schoolsup", "school", "reason", "traveltime"]),
    ("👨‍👩‍👧 Family & Background", "Home environment and parents",
     ["age", "sex", "address", "famsize", "Pstatus", "Medu", "Fedu",
      "Mjob", "Fjob", "guardian", "famsup", "famrel"]),
    ("🌟 Lifestyle & Behaviour", "Social life, health and daily habits",
     ["goout", "freetime", "Walc", "Dalc", "health", "romantic",
      "internet", "activities", "nursery"]),
]

# ------------------------------------------------------------------
# Friendly names for the dataset columns
# ------------------------------------------------------------------
LABELS = {
    "school": "School", "sex": "Sex", "age": "Age",
    "address": "Home address type", "famsize": "Family size",
    "Pstatus": "Parents' cohabitation status",
    "Medu": "Mother's education (0-4)", "Fedu": "Father's education (0-4)",
    "Mjob": "Mother's job", "Fjob": "Father's job",
    "reason": "Reason for choosing this school", "guardian": "Guardian",
    "traveltime": "Home-to-school travel time (1-4)",
    "studytime": "Weekly study time (1-4)",
    "failures": "Number of past class failures",
    "schoolsup": "Extra educational support from school",
    "famsup": "Family educational support", "paid": "Extra paid classes",
    "activities": "Extra-curricular activities", "nursery": "Attended nursery school",
    "higher": "Wants to take higher education", "internet": "Internet access at home",
    "romantic": "In a romantic relationship",
    "famrel": "Quality of family relationships (1-5)",
    "freetime": "Free time after school (1-5)",
    "goout": "Going out with friends (1-5)",
    "Dalc": "Workday alcohol consumption (1-5)",
    "Walc": "Weekend alcohol consumption (1-5)",
    "health": "Current health status (1-5)",
    "absences": "Number of school absences",
    "G1": "First period grade (0-20)", "G2": "Second period grade (0-20)",
}

HELP = {
    "Medu": "0 = none, 1 = primary (4th grade), 2 = 5th-9th grade, 3 = secondary, 4 = higher education",
    "Fedu": "0 = none, 1 = primary (4th grade), 2 = 5th-9th grade, 3 = secondary, 4 = higher education",
    "traveltime": "1 = under 15 min, 2 = 15-30 min, 3 = 30 min-1 hour, 4 = over 1 hour",
    "studytime": "1 = under 2 hours, 2 = 2-5 hours, 3 = 5-10 hours, 4 = over 10 hours",
    "famrel": "1 = very bad ... 5 = excellent",
    "freetime": "1 = very low ... 5 = very high",
    "goout": "1 = very low ... 5 = very high",
    "Dalc": "1 = very low ... 5 = very high",
    "Walc": "1 = very low ... 5 = very high",
    "health": "1 = very bad ... 5 = very good",
}

VALUE_LABELS = {
    "school": {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"},
    "sex": {"F": "Female", "M": "Male"},
    "address": {"U": "Urban", "R": "Rural"},
    "famsize": {"LE3": "3 or fewer members", "GT3": "More than 3 members"},
    "Pstatus": {"T": "Living together", "A": "Apart"},
}

# Result text for each grade category
RESULT_INFO = {
    "Low": ("error", "Needs Improvement",
            "Predicted final grade is below 10. Focus on attendance, regular study time and extra support."),
    "Medium": ("warning", "Average - Employable with upskilling",
               "Predicted final grade is between 10 and 14. A little more consistency can move this student up."),
    "High": ("success", "Excellent - Highly employable",
             "Predicted final grade is 15 or above. Keep up the strong performance."),
}


def pretty(feature):
    return LABELS.get(feature, feature)


def pretty_value(feature, value):
    mapping = VALUE_LABELS.get(feature, {})
    if value in mapping:
        return mapping[value]
    if value in ("yes", "no"):
        return value.capitalize()
    return str(value).replace("_", " ").capitalize()


# ------------------------------------------------------------------
# Load the trained files (cached so it only happens once)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(ART / "dnn_model.keras", compile=False)
    preprocessor = joblib.load(ART / "preprocessor.joblib")
    with open(ART / "metadata.json") as f:
        meta = json.load(f)
    return model, preprocessor, meta


try:
    model, preprocessor, meta = load_artifacts()
except Exception as e:
    st.error("Could not load the model files. Make sure the `artifacts` folder next to `app.py` "
             "contains dnn_model.keras, preprocessor.joblib and metadata.json.")
    st.exception(e)
    st.stop()

features = meta["selected_features"]
feature_info = meta["features"]
class_names = meta["class_names"]

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("🎓 Student Academic Performance Predictor")
st.caption("Deep Neural Network model trained on the UCI Student Performance dataset. "
           "Enter the student details to predict the final grade category.")

tab_predict, tab_model = st.tabs(["🔮 Predict", "📊 Model Information"])

# ------------------------------------------------------------------
# Tab 1: prediction form
# ------------------------------------------------------------------
with tab_predict:
    st.markdown(
        '<div class="intro-box"><h2>🧑‍🎓 Enter Student Details</h2>'
        '<p>Fill in the three sections below, then click <b>Predict Performance</b> '
        'to see the predicted grade category.</p></div>',
        unsafe_allow_html=True)

    def render_input(feat):
        info = feature_info[feat]
        if info["type"] == "categorical":
            return st.selectbox(
                pretty(feat), info["options"],
                format_func=lambda v, f=feat: pretty_value(f, v),
                help=HELP.get(feat), key=feat)
        lo, hi = int(info["min"]), int(info["max"])
        default = min(max(int(round(info["default"])), lo), hi)
        return st.slider(pretty(feat), lo, hi, default, help=HELP.get(feat), key=feat)

    # any selected feature not listed in SECTIONS goes to an "Other" card
    placed = {f for _, _, fl in SECTIONS for f in fl}
    sections = list(SECTIONS)
    leftover = [f for f in features if f not in placed]
    if leftover:
        sections.append(("🔧 Other Details", "Additional information", leftover))

    values = {}
    with st.form("student_form", border=False):
        for title, subtitle, feat_list in sections:
            feat_list = [f for f in feat_list if f in features]
            if not feat_list:
                continue
            with st.container(border=True):
                st.markdown(
                    f'<div class="section-title">{title}</div>'
                    f'<div class="section-sub">{subtitle}</div>',
                    unsafe_allow_html=True)
                cols = st.columns(3)
                for i, feat in enumerate(feat_list):
                    with cols[i % 3]:
                        values[feat] = render_input(feat)

        submitted = st.form_submit_button("🔮  Predict Performance", type="primary", width="stretch")

    if submitted:
        # one row, columns in the same order used during training
        row = pd.DataFrame([values])[features]
        x = preprocessor.transform(row).astype("float32")
        proba = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(proba))
        pred_name = class_names[pred_idx]

        level, title, message = RESULT_INFO.get(pred_name, ("info", pred_name, ""))

        st.divider()
        st.subheader("Prediction result")
        c1, c2 = st.columns([1, 1])

        with c1:
            getattr(st, level)(f"### {pred_name}: {title}")
            st.write(message)
            st.metric("Model confidence", f"{proba[pred_idx] * 100:.1f}%")

        with c2:
            prob_df = pd.DataFrame({"Category": class_names, "Probability": proba}).set_index("Category")
            st.bar_chart(prob_df, y="Probability")

        with st.expander("Show the input used for this prediction"):
            show = pd.DataFrame({
                "Feature": [pretty(f) for f in features],
                "Value": [pretty_value(f, values[f]) for f in features],
            })
            st.dataframe(show, hide_index=True, width="stretch")

# ------------------------------------------------------------------
# Tab 2: model info
# ------------------------------------------------------------------
with tab_model:
    st.subheader("Grade categories")
    st.table(pd.DataFrame({
        "Category": ["Low", "Medium", "High"],
        "Final grade (G3)": ["0 - 9", "10 - 14", "15 - 20"],
        "Meaning": ["Needs improvement", "Average", "Excellent"],
    }))

    st.subheader("Model comparison (test data)")
    if "results" in meta:
        st.dataframe(pd.DataFrame(meta["results"]).round(3), width="stretch")

    st.subheader("Features used by the model")
    st.write(", ".join(pretty(f) for f in features))

    st.subheader("Model architecture")
    lines = []
    model.summary(print_fn=lambda s: lines.append(s))
    st.code("\n".join(lines))