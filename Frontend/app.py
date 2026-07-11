# ============================================================================
#  PUNE SMART TOWING INTELLIGENCE SYSTEM
#  A Streamlit application for towing-risk prediction and yard assignment
#  across Pune Traffic Divisions.
#
#  NOTE ON ML LOGIC: The feature engineering, label encoding, model
#  hyperparameters and prediction logic in this file are carried over
#  unchanged from the original research notebook (real_world_data.ipynb,
#  the "final production model" cell). Only the surrounding application
#  (UI, caching, charts, navigation) is new.
# ============================================================================

import os
import re
import difflib
import joblib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pune Smart Towing Intelligence System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR,"Backend","Pune_With_Real_Features_v5.csv")

# ----------------------------------------------------------------------------
# CUSTOM CSS  —  "Night Patrol" theme: deep navy command-console base with
# an amber/hazard-stripe accent borrowed from tow-truck warning markings.
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">

<style>
:root{
    --bg:        #0A0E1A;
    --surface:   #121A2B;
    --surface2:  #1A2438;
    --border:    rgba(255,255,255,0.08);
    --text:      #E8ECF4;
    --muted:     #8B96AC;
    --amber:     #FF9F43;
    --amber-dim: #C97A2E;
    --red:       #FF5470;
    --teal:      #2DD4BF;
    --blue:      #5B8DEF;
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4, .hero-title, .section-title { font-family: 'Space Grotesk', sans-serif; }
.mono { font-family: 'JetBrains Mono', monospace; }

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(255,159,67,0.06), transparent 40%),
        radial-gradient(circle at 85% 10%, rgba(45,212,191,0.05), transparent 35%),
        var(--bg);
    color: var(--text);
}

/* Hide default streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] { background: transparent; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0D1320;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
}
.sidebar-brand {
    padding: 0.4rem 0 1.1rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.1rem;
}
.sidebar-brand h2 {
    font-size: 1.15rem;
    margin: 0;
    color: var(--text);
    line-height: 1.3;
}
.sidebar-brand span {
    color: var(--amber);
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* FIX: sidebar text invisible on dark background */
section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stRadio label p,
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: var(--text) !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
    background: var(--surface2) !important;
    border-radius: 8px;
}
section[data-testid="stSidebar"] .mono,
section[data-testid="stSidebar"] .kpi-value,
section[data-testid="stSidebar"] .kpi-label {
    color: var(--text) !important;
}

/* FIX: text inputs, selectboxes, dropdowns — white bg but text was too light */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background-color: #1A2438 !important;
    color: #E8ECF4 !important;
    border: 1px solid rgba(255,159,67,0.35) !important;
    font-weight: 500;
}
.stTextInput input::placeholder {
    color: #8B96AC !important;
    opacity: 1 !important;
}
/* Dropdown menu options list */
div[data-baseweb="popover"] li,
div[data-baseweb="menu"] li {
    color: #E8ECF4 !important;
}
div[data-baseweb="popover"] ul,
div[data-baseweb="menu"] ul {
    background-color: #1A2438 !important;
}
/* Selected value text + label above the field */
.stSelectbox label, .stTextInput label, .stNumberInput label, .stMultiSelect label {
    color: var(--text) !important;
}

/* Hazard stripe divider — signature element */
.hazard-strip {
    height: 6px;
    width: 100%;
    border-radius: 4px;
    margin: 0.3rem 0 1.6rem 0;
    background: repeating-linear-gradient(
        135deg,
        var(--amber) 0px, var(--amber) 14px,
        #1A1A1A 14px, #1A1A1A 28px
    );
    opacity: 0.85;
}

/* Hero */
.hero {
    padding: 2.4rem 2.2rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #161F35 0%, #0F1626 100%);
    border: 1px solid var(--border);
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero::after{
    content:"";
    position:absolute; right:-60px; top:-60px;
    width:220px; height:220px; border-radius:50%;
    background: radial-gradient(circle, rgba(255,159,67,0.18), transparent 70%);
}
.hero-eyebrow {
    color: var(--amber);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 2.3rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 0.6rem 0;
    line-height: 1.15;
}
.hero-sub {
    color: var(--muted);
    font-size: 1.02rem;
    max-width: 640px;
    line-height: 1.55;
}

/* KPI cards */
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.3rem;
    height: 100%;
}
.kpi-label {
    color: var(--muted);
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.45rem;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}
.kpi-delta { font-size: 0.78rem; margin-top: 0.45rem; }
.kpi-amber { color: var(--amber); }
.kpi-teal  { color: var(--teal); }
.kpi-red   { color: var(--red); }
.kpi-blue  { color: var(--blue); }

/* Section title */
.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text);
    margin: 1.6rem 0 0.2rem 0;
}
.section-sub { color: var(--muted); font-size: 0.88rem; margin-bottom: 0.9rem;}

/* Generic content card */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.3rem 1.4rem;
    margin-bottom: 1rem;
}

/* Result banner */
.result-card {
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    border: 1px solid var(--border);
    margin-top: 0.6rem;
}
.result-high { background: linear-gradient(135deg, rgba(255,84,112,0.16), rgba(255,84,112,0.04)); border-color: rgba(255,84,112,0.4);}
.result-low  { background: linear-gradient(135deg, rgba(45,212,191,0.16), rgba(45,212,191,0.04)); border-color: rgba(45,212,191,0.4);}

.badge {
    display:inline-block; padding: 0.25rem 0.7rem; border-radius: 999px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.05em; text-transform: uppercase;
}
.badge-critical { background: rgba(255,84,112,0.18); color: #FF8AA0; border: 1px solid rgba(255,84,112,0.4);}
.badge-high     { background: rgba(255,159,67,0.18); color: var(--amber); border: 1px solid rgba(255,159,67,0.4);}
.badge-moderate { background: rgba(255,214,10,0.16); color: #FFD60A; border: 1px solid rgba(255,214,10,0.35);}
.badge-low      { background: rgba(45,212,191,0.16); color: var(--teal); border: 1px solid rgba(45,212,191,0.4);}

/* Yard card */
.yard-card {
    background: linear-gradient(135deg, rgba(91,141,239,0.14), rgba(91,141,239,0.02));
    border: 1px solid rgba(91,141,239,0.35);
    border-radius: 16px;
    padding: 1.7rem;
    text-align: center;
}
.yard-card .yard-icon { font-size: 2.2rem; }
.yard-card .yard-name { font-family:'Space Grotesk',sans-serif; font-size: 1.4rem; font-weight: 700; color: #fff; margin: 0.4rem 0;}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, var(--amber), var(--amber-dim));
    color: #1A1206;
    border: none;
    font-weight: 700;
    border-radius: 10px;
    padding: 0.55rem 1.4rem;
    font-family: 'Space Grotesk', sans-serif;
}
.stButton>button:hover { filter: brightness(1.08); }

/* Dataframe / metric tweaks */
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #FFFFFF !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; }

hr { border-color: var(--border); }
/* Streamlit 1.32 Selectbox Fix */

div[data-baseweb="select"] > div {
    background-color: #1A2438 !important;
    color: #E8ECF4 !important;
}

div[data-baseweb="select"] input {
    color: #E8ECF4 !important;
    -webkit-text-fill-color: #E8ECF4 !important;
}

div[data-baseweb="select"] span {
    color: #E8ECF4 !important;
}

div[data-baseweb="select"] svg {
    fill: #E8ECF4 !important;
}
div[data-baseweb="select"] > div > div {
    background-color: #1A2438 !important;
}
div[data-baseweb="base-input"] {
    background-color: #1A2438 !important;
}
[data-baseweb="select"] [data-testid="stMarkdownContainer"] {
    color: #E8ECF4 !important;
}
input[role="combobox"] {
    background-color: #1A2438 !important;
    color: #E8ECF4 !important;
}

input[role="combobox"]::placeholder {
    color: #8B96AC !important;
    opacity: 1 !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    return df


# ----------------------------------------------------------------------------
# ORIGINAL ML LOGIC (verbatim from notebook cell 18 — "final production model")
# Feature engineering, label encoding, and RandomForest hyperparameters are
# unchanged. Only wrapped in cached functions for the Streamlit app.
# ----------------------------------------------------------------------------
def word_in_text(text, words):
    for w in words:
        if re.search(r"\b" + re.escape(w) + r"\b", str(text).lower()):
            return 1
    return 0


def extract_features(text):
    t = str(text).lower()
    return {
        "near_junction": word_in_text(t, ["junction", "chowk", "intersection", "crossing"]),
        "near_bus_stop": word_in_text(t, ["bus stop", "bus stand", "depot", "pmpml"]),
        "near_highway": word_in_text(t, ["highway", "nh-", "bypass", "expressway"]),
        "footpath_violation": word_in_text(t, ["footpath", "pavement", "sidewalk"]),
        "both_sides_banned": word_in_text(t, ["both sides", "either side"]),
        "immediate_tow_mention": word_in_text(t, ["immediately towed", "tow away", "will be towed"]),
        "near_landmark": word_in_text(t, ["temple", "college", "hospital", "market", "school", "station"]),
        "narrow_road": word_in_text(t, ["narrow", "tight lane"]),
        "no_parking_explicit": word_in_text(t, ["no parking"]),
        "meters_distance_mention": word_in_text(t, ["meters", "metre"]),
        "market_area": word_in_text(t, ["market", "bazaar"]),
        "zero_tolerance_kw": word_in_text(
            t, ["zero tolerance", "strictly prohibited", "strictly banned", "absolute ban"]
        ),
        "description_length": len(str(text)),
    }


def find_division_match(text, division_classes, cutoff=0.6, allow_fuzzy=True):
    """
    Resolves typed text to a Division Name, tolerant of minor typos and the
    dataset's inconsistent spacing (e.g. 'Khadki  Traffic Division' with a
    double space). Tries, in order:
      1. exact case-insensitive match
      2. substring containment (either direction), min 4 chars
      3. fuzzy closest-match (difflib) above the cutoff, only if allow_fuzzy
    Returns the matched Division Name string, or None.
    """
    text_clean = re.sub(r"\s+", " ", str(text).strip().lower())
    if not text_clean:
        return None

    norm_map = {re.sub(r"\s+", " ", d.strip().lower()): d for d in division_classes}

    if text_clean in norm_map:
        return norm_map[text_clean]

    bare_text = re.sub(r"\s*traffic\s*division\s*$", "", text_clean).strip()
    bare_norm_map = {
        re.sub(r"\s*traffic\s*division\s*$", "", norm): orig
        for norm, orig in norm_map.items()
    }

    if len(bare_text) >= 4:
        for bare_norm, orig in bare_norm_map.items():
            if bare_text in bare_norm or bare_norm in bare_text:
                return orig

    if allow_fuzzy:
        close = difflib.get_close_matches(bare_text, list(bare_norm_map.keys()), n=1, cutoff=cutoff)
        if close:
            return bare_norm_map[close[0]]

    return None


def resolve_location(input_text, division_classes, road_data):
    """
    Shared 4-tier resolution used by both the Risk Predictor and the Yard
    Finder, so typing an area name (even with a typo) never gets silently
    overridden by whatever the Area/Division dropdown happens to show:
      1. Strict Division/Area match (exact or substring, no typos) — wins
         outright, since an unrelated road's description may mention the
         area only in passing.
      2. Road Name match.
      3. Fuzzy (typo-tolerant) Division/Area match.
      4. No match at all -> caller should prompt the user to check spelling.
    Returns a dict: {status, division_name, yard_name, road_name, description_text}
    status is one of "division", "road", "fuzzy_division", "no_match".

    road_name is the road text to DISPLAY (None when the match came from an
    area/division search rather than a specific road).
    description_text is the text to feed into extract_features() — it is
    ALWAYS a real descriptive sentence from the matched division/road, never
    just the short typed input, so the model sees the same kind of text it
    was trained on.
    """
    text_clean = str(input_text).strip()
    if not text_clean:
        return {"status": "no_match", "division_name": None, "yard_name": None, "road_name": None, "description_text": None}

    # Strict division/area matching is only meaningful for short,
    # area-name-shaped queries (e.g. "Khadki", "Bharati Vidyapeeth"). A long
    # pasted road description may mention an unrelated area in passing
    # (e.g. "...border with Chaturshringi..."), so skip straight to Road
    # Name matching for longer inputs to avoid a false area-name hit.
    SHORT_QUERY_LIMIT = 60
    strict_division = (
        find_division_match(text_clean, division_classes, allow_fuzzy=False)
        if len(text_clean) <= SHORT_QUERY_LIMIT else None
    )
    if strict_division:
        sub = road_data[road_data["Division Name"] == strict_division]
        if not sub.empty:
            row = sub.iloc[0]
            return {
                "status": "division",
                "division_name": row["Division Name"],
                "yard_name": row["Yard Name"],
                "road_name": None,
                "description_text": row["Road Name"],
            }

    road_match = road_data[road_data["Road Name"].str.contains(re.escape(text_clean), case=False, na=False)]
    if not road_match.empty:
        row = road_match.iloc[0]
        return {
            "status": "road",
            "division_name": row["Division Name"],
            "yard_name": row["Yard Name"],
            "road_name": row["Road Name"],
            "description_text": row["Road Name"],
        }

    fuzzy_division = find_division_match(text_clean, division_classes, allow_fuzzy=True)
    if fuzzy_division:
        sub = road_data[road_data["Division Name"] == fuzzy_division]
        if not sub.empty:
            row = sub.iloc[0]
            return {
                "status": "fuzzy_division",
                "division_name": row["Division Name"],
                "yard_name": row["Yard Name"],
                "road_name": None,
                "description_text": row["Road Name"],
            }

    return {"status": "no_match", "division_name": None, "yard_name": None, "road_name": None, "description_text": None}


FEATURES = [
    "near_junction", "near_bus_stop", "near_highway",
    "footpath_violation", "both_sides_banned",
    "immediate_tow_mention", "near_landmark",
    "narrow_road", "no_parking_explicit",
    "meters_distance_mention", "market_area",
    "zero_tolerance_kw", "description_length",
    "div_enc", "date_enc",
]


@st.cache_resource
def train_model(df: pd.DataFrame):
    data = df.copy()

    data["zero_tolerance_kw"] = data["Road Name"].apply(
        lambda x: word_in_text(
            x, ["zero tolerance", "strictly prohibited", "strictly banned", "absolute ban"]
        )
    )

    le_div = LabelEncoder()
    le_date = LabelEncoder()
    le_div.fit(data["Division Name"])
    le_date.fit(data["Date Type"])
    data["div_enc"] = le_div.transform(data["Division Name"])
    data["date_enc"] = le_date.transform(data["Date Type"])

    X = data[FEATURES]
    y = data["zero_tolerance"]

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    model.fit(X, y)

    return model, le_div, le_date


def predict_risk(model, le_div, le_date, data, road_name_input, division_input, date_type_input):
    """Replicates the original notebook's prediction flow (cell 18), with
    area-name resolution (resolve_location) layered on top so a typed
    division/area name is never silently overridden by the dropdown."""
    resolved = resolve_location(road_name_input, le_div.classes_, data)

    if resolved["status"] in ("division", "fuzzy_division", "road"):
        division_name = resolved["division_name"]
        yard_name = resolved["yard_name"]
        input_text = resolved["description_text"]
        match_found = True
    else:
        division_name = division_input
        input_text = road_name_input
        yard_name = division_name.replace("Traffic Division", "Traffic Yard")
        match_found = False

    date_type = date_type_input if date_type_input in le_date.classes_ else "Both"

    feats = extract_features(input_text)
    feats["div_enc"] = le_div.transform([division_name])[0]
    feats["date_enc"] = le_date.transform([date_type])[0]

    input_df = pd.DataFrame([feats])[FEATURES]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    confidence = probability.max() * 100
    risk_score = probability[1] * 100  # probability of class 1 (zero_tolerance / high risk)

    return {
        "match_found": match_found,
        "no_match_at_all": resolved["status"] == "no_match",
        "division_name": division_name,
        "yard_name": yard_name,
        "prediction": int(prediction),
        "confidence": confidence,
        "risk_score": risk_score,
    }


def risk_category(risk_score: float):
    if risk_score >= 75:
        return "Critical Risk", "badge-critical", "#FF5470"
    elif risk_score >= 50:
        return "High Risk", "badge-high", "#FF9F43"
    elif risk_score >= 25:
        return "Moderate Risk", "badge-moderate", "#FFD60A"
    else:
        return "Low Risk", "badge-low", "#2DD4BF"


# ----------------------------------------------------------------------------
# LOAD DATA + MODEL ONCE
# ----------------------------------------------------------------------------
df = load_data()
model, le_div, le_date = train_model(df)

model = joblib.load(os.path.join(BASE_DIR, "Backend", "towing_model.pkl"))
le_div = joblib.load(os.path.join(BASE_DIR, "Backend", "label_encoder_div.pkl"))
le_date = joblib.load(os.path.join(BASE_DIR, "Backend", "label_encoder_date.pkl"))

DIVISIONS = sorted(df["Division Name"].unique().tolist())
DATE_TYPES = ["Both", "Odd", "Even"]
VEHICLE_TYPES = ["Car / Sedan", "Two-Wheeler", "Auto-Rickshaw", "SUV / MUV", "Bus", "Truck / Heavy Vehicle", "Other"]

# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <span>PUNE TRAFFIC POLICE · AI MODULE</span>
            <h2>🚦 Smart Towing<br/>Intelligence</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "🚓 Risk Predictor",
            "🏢 Yard Finder",
            "📊 Analytics",
            "ℹ️ About Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="mono" style="font-size:0.74rem; color:var(--muted); line-height:1.8;">
        RECORDS&nbsp;LOADED &nbsp; <span style="color:var(--text)">{len(df)}</span><br/>
        DIVISIONS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:var(--text)">{df['Division Name'].nunique()}</span><br/>
        MODEL &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:var(--text)">RandomForest</span><br/>
        STATUS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:var(--teal)">● ACTIVE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
if page == "🏠 Dashboard":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">PUNE SMART TOWING INTELLIGENCE SYSTEM</div>
            <div class="hero-title">Predict towing risk before<br/>you park.</div>
            <div class="hero-sub">
                A machine-learning command center for Pune's traffic enforcement data —
                covering 50 traffic divisions, real towing-zone descriptions, and a
                RandomForest risk model trained on zero-tolerance parking signals.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hazard-strip"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">KPI Statistics</div>', unsafe_allow_html=True)
    total = len(df)
    towed_pct = df["Towed"].mean() * 100
    zt_pct = df["zero_tolerance"].mean() * 100
    n_divisions = df["Division Name"].nunique()
    n_yards = df["Yard Name"].nunique()
    avg_desc_len = df["description_length"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">Total Records</div>
            <div class="kpi-value">{total}</div>
            <div class="kpi-delta kpi-blue">Towing zone entries</div></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">Historically Towed</div>
            <div class="kpi-value">{towed_pct:.1f}%</div>
            <div class="kpi-delta kpi-red">{int(df['Towed'].sum())} of {total} zones</div></div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">Zero-Tolerance Zones</div>
            <div class="kpi-value">{zt_pct:.1f}%</div>
            <div class="kpi-delta kpi-amber">High enforcement priority</div></div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">Divisions / Yards</div>
            <div class="kpi-value">{n_divisions} / {n_yards}</div>
            <div class="kpi-delta kpi-teal">Across Pune &amp; PCMC</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Quick Insights</div>', unsafe_allow_html=True)
    colL, colR = st.columns([1.3, 1])

    with colL:
        top_div = (
            df.groupby("Division Name")["zero_tolerance"].mean().sort_values(ascending=False).head(8) * 100
        )
        fig = px.bar(
            top_div[::-1],
            orientation="h",
            labels={"value": "Zero-Tolerance Rate (%)", "Division Name": ""},
            color=top_div[::-1].values,
            color_continuous_scale=["#2DD4BF", "#FF9F43", "#FF5470"],
        )
        fig.update_layout(
            title="Top 8 High-Risk Divisions",
            template="plotly_dark",
            paper_bgcolor="#121A2B",
            plot_bgcolor="#121A2B",
            font={"color": "#E8ECF4"},
            showlegend=False,
            coloraxis_showscale=False,
            height=360,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with colR:
        towed_counts = df["Towed"].map({1: "Towed", 0: "Not Towed"}).value_counts()
        fig2 = px.pie(
            values=towed_counts.values,
            names=towed_counts.index,
            hole=0.55,
            color=towed_counts.index,
            color_discrete_map={"Towed": "#FF5470", "Not Towed": "#2DD4BF"},
        )
        fig2.update_layout(
            title="Towing Outcome Split",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=360,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        f"""
        <div class="panel">
        <b>Reading the board:</b> on average, towing-zone descriptions run
        <span class="mono">{avg_desc_len:.0f}</span> characters, and divisions with the highest
        zero-tolerance rates tend to mention junctions, market areas, and explicit
        "no parking" language most often. Use the <b>Risk Predictor</b> to score a specific
        road, or the <b>Yard Finder</b> to look up where a towed vehicle would be taken.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# PAGE: RISK PREDICTOR
# ============================================================================
elif page == "🚓 Risk Predictor":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">MODULE 02 · RISK ENGINE</div>
            <div class="hero-title">Vehicle Towing Risk Predictor</div>
            <div class="hero-sub">RandomForest model trained on 15 engineered features from real
            towing-zone descriptions — junction proximity, market areas, explicit no-parking
            language, division and date-type patterns.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hazard-strip"></div>', unsafe_allow_html=True)

    road_options = ["✏️ Type a custom road / location"] + sorted(df["Road Name"].unique().tolist())

    with st.container():
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            road_choice = st.selectbox(
                "Road Name",
                road_options,
                format_func=lambda x: x if len(x) <= 70 else x[:67] + "...",
            )
            if road_choice == "✏️ Type a custom road / location":
                road_name = st.text_input("Enter road / location name", placeholder="e.g. FC Road near Goodluck Chowk")
            else:
                road_name = road_choice

            vehicle_type = st.selectbox("Vehicle Type", VEHICLE_TYPES)

        with col2:
            division_input = st.selectbox("Area / Division", DIVISIONS)
            date_type_input = st.selectbox("Date Type (odd/even parking rule)", DATE_TYPES, index=0)

        st.markdown("</div>", unsafe_allow_html=True)

        predict_clicked = st.button("⚡ Predict Towing Risk", use_container_width=False)

    if predict_clicked:
        if not road_name or not road_name.strip():
            st.warning("Please enter or select a road name to continue.")
        else:
            result = predict_risk(model, le_div, le_date, df, road_name, division_input, date_type_input)

            if result["no_match_at_all"]:
                st.warning(
                    "⚠️ We couldn't match **"
                    + road_name
                    + "** to any road or area in our database. "
                    "Please check the spelling and try again, or pick a value from the dropdown list "
                    "so it matches our records exactly."
                )

            cat_label, cat_class, cat_color = risk_category(result["risk_score"])
            card_class = "result-high" if result["prediction"] == 1 else "result-low"
            verdict = "🔴 High Risk — Likely Immediate Tow" if result["prediction"] == 1 else "🟢 Low Risk — Relatively Safe"

            st.markdown(f'<div class="result-card {card_class}">', unsafe_allow_html=True)
            rc1, rc2 = st.columns([1.3, 1])

            with rc1:
                st.markdown(f"#### {verdict}")
                st.write("")
                st.markdown(
                    f"""
                    **Division:** {result['division_name']}<br>
                    **Match found:** {"Yes ✅" if result['match_found'] else "No — used manual division ⚠️"}<br>
                    **Vehicle Type:** {vehicle_type} <span style="color:var(--muted); font-size:0.8rem;">(contextual — not a model input)</span><br>
                    **Date Type:** {date_type_input}<br>
                    """,
                    unsafe_allow_html=True,
                )
                m1, m2 = st.columns(2)
                m1.metric("Risk Score", f"{result['risk_score']:.1f}%")
                m2.metric("Confidence Score", f"{result['confidence']:.1f}%")
                st.link_button(
    "🗺️ Find Yard on Maps",
    f"https://www.google.com/maps/search/{result['yard_name']}+Pune"
)

            with rc2:
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=result["risk_score"],
                        number={"suffix": "%", "font": {"color": "#fff", "family": "JetBrains Mono"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#8B96AC"},
                            "bar": {"color": cat_color},
                            "bgcolor": "rgba(0,0,0,0)",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0, 25], "color": "rgba(45,212,191,0.25)"},
                                {"range": [25, 50], "color": "rgba(255,214,10,0.22)"},
                                {"range": [50, 75], "color": "rgba(255,159,67,0.25)"},
                                {"range": [75, 100], "color": "rgba(255,84,112,0.25)"},
                            ],
                        },
                    )
                )
                gauge.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=240,
                    margin=dict(l=20, r=20, t=20, b=10),
                )
                st.plotly_chart(gauge, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# PAGE: YARD FINDER
# ============================================================================
elif page == "🏢 Yard Finder":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">MODULE 03 · LOGISTICS</div>
            <div class="hero-title">Vehicle Yard Finder</div>
            <div class="hero-sub">Look up which towing yard a vehicle would be taken to,
            based on road match or division assignment.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hazard-strip"></div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    yf1, yf2 = st.columns(2)
    with yf1:
        road_options_yf = ["✏️ Type a custom road / location"] + sorted(df["Road Name"].unique().tolist())
        road_choice_yf = st.selectbox(
            "Road Name",
            road_options_yf,
            format_func=lambda x: x if len(x) <= 70 else x[:67] + "...",
            key="yf_road_choice",
        )
        if road_choice_yf == "✏️ Type a custom road / location":
            road_name_yf = st.text_input(
                "Enter road / location name",
                placeholder="e.g. FC Road near Goodluck Chowk",
                key="yf_road_input",
            )
        else:
            road_name_yf = road_choice_yf
    with yf2:
        division_input_yf = st.selectbox(
            "Area / Division (used only if no match is found above)",
            DIVISIONS,
            key="yf_division",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    find_clicked = st.button("🔍 Find Assigned Yard")

    if find_clicked:
        if not road_name_yf or not road_name_yf.strip():
            st.warning("Please enter or select a road name to continue.")
        else:
            resolved = resolve_location(road_name_yf, le_div.classes_, df)

            if resolved["status"] == "no_match":
                st.warning(
                    "⚠️ We couldn't match **"
                    + road_name_yf
                    + "** to any road or area in our database. "
                    "Please check the spelling and enter it correctly so it matches our records — "
                    "or pick a value from the dropdown list above instead of typing it manually."
                )
                division_name = division_input_yf
                yard_name = division_name.replace("Traffic Division", "Traffic Yard")
                found = False
            else:
                division_name = resolved["division_name"]
                yard_name = resolved["yard_name"]
                found = True

            yc1, yc2 = st.columns([1, 1.2])
            with yc1:
                st.markdown(
                    f"""
                    <div class="yard-card">
                        <div class="yard-icon">🏢</div>
                        <div class="kpi-label" style="margin-top:0.6rem;">ASSIGNED TOWING YARD</div>
                        <div class="yard-name">{yard_name}</div>
                        <div class="mono" style="color:var(--muted); font-size:0.82rem;">{division_name}</div>
                        <div style="margin-top:0.8rem;">
                            <span class="badge {'badge-low' if found else 'badge-moderate'}">
                            {'Matched' if found else 'Not found — check spelling'}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with yc2:
                yard_records = df[df["Yard Name"] == yard_name]
                if not yard_records.empty:
                    st.markdown("**Yard activity snapshot**")
                    st.metric("Records linked to this yard", len(yard_records))
                    st.metric("Towed rate at this yard", f"{yard_records['Towed'].mean()*100:.1f}%")
                    st.link_button(
    "🗺️ Navigate to Yard",
    f"https://www.google.com/maps/search/{yard_name}+Pune+Traffic+Yard"
)
                else:
                    st.info("No historical records found for this yard yet.")

    with st.expander("📋 View full Division → Yard directory"):
        directory = df[["Division Name", "Yard Name"]].drop_duplicates().sort_values("Division Name")
        st.dataframe(directory, use_container_width=True, hide_index=True)

    with st.expander("🔎 Browse known road / location names"):
        st.dataframe(
            df[["Road Name", "Division Name"]].sort_values("Road Name"),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================================
# PAGE: ANALYTICS
# ============================================================================
elif page == "📊 Analytics":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">MODULE 04 · CITYWIDE ANALYTICS</div>
            <div class="hero-title">Analytics Dashboard</div>
            <div class="hero-sub">Explore towing patterns, yard load, and division-level risk
            across all 297 recorded towing zones.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hazard-strip"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔥 High-Risk Areas", "🚛 Towing Distribution", "🏢 Yard Distribution", "📍 Area-wise Insights"]
    )

    with tab1:
        risk_by_div = (
            df.groupby("Division Name")["zero_tolerance"].mean().sort_values(ascending=False).head(15) * 100
        )
        fig = px.bar(
            risk_by_div[::-1],
            orientation="h",
            labels={"value": "Zero-Tolerance Rate (%)", "Division Name": ""},
            color=risk_by_div[::-1].values,
            color_continuous_scale=["#2DD4BF", "#FF9F43", "#FF5470"],
        )
        fig.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False, coloraxis_showscale=False, height=520, margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Top 15 divisions ranked by share of zero-tolerance towing zones.")

    with tab2:
        cdist1, cdist2 = st.columns(2)
        with cdist1:
            towed_counts = df["Towed"].map({1: "Towed", 0: "Not Towed"}).value_counts()
            fig2 = px.pie(
                values=towed_counts.values, names=towed_counts.index, hole=0.5,
                color=towed_counts.index,
                color_discrete_map={"Towed": "#FF5470", "Not Towed": "#2DD4BF"},
            )
            fig2.update_layout(
                title="Towed vs Not Towed", template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=380,
            )
            st.plotly_chart(fig2, use_container_width=True)
        with cdist2:
            by_date = df.groupby("Date Type")["Towed"].mean().sort_values(ascending=False) * 100
            fig3 = px.bar(
                by_date, labels={"value": "Towed Rate (%)", "Date Type": ""},
                color=by_date.index, color_discrete_sequence=["#FF9F43", "#5B8DEF", "#2DD4BF"],
            )
            fig3.update_layout(
                title="Towed Rate by Date Type", template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=380, showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        yard_counts = df["Yard Name"].value_counts().head(15)
        fig4 = px.bar(
            yard_counts[::-1], orientation="h",
            labels={"value": "Records Assigned", "Yard Name": ""},
            color=yard_counts[::-1].values, color_continuous_scale=["#1A2438", "#5B8DEF"],
        )
        fig4.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False, coloraxis_showscale=False, height=520, margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("Top 15 towing yards by number of records assigned to them.")

    with tab4:
        area_summary = (
            df.groupby("Division Name")
            .agg(
                Records=("Road Name", "count"),
                Towed_Rate=("Towed", "mean"),
                Zero_Tolerance_Rate=("zero_tolerance", "mean"),
                Avg_Description_Length=("description_length", "mean"),
            )
            .reset_index()
        )
        area_summary["Towed_Rate"] = (area_summary["Towed_Rate"] * 100).round(1)
        area_summary["Zero_Tolerance_Rate"] = (area_summary["Zero_Tolerance_Rate"] * 100).round(1)
        area_summary["Avg_Description_Length"] = area_summary["Avg_Description_Length"].round(0)
        area_summary = area_summary.sort_values("Towed_Rate", ascending=False)

        st.dataframe(
            area_summary.rename(
                columns={
                    "Division Name": "Division",
                    "Towed_Rate": "Towed Rate (%)",
                    "Zero_Tolerance_Rate": "Zero-Tolerance Rate (%)",
                    "Avg_Description_Length": "Avg. Description Length",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        fig5 = px.scatter(
            area_summary, x="Towed_Rate", y="Zero_Tolerance_Rate", size="Records",
            hover_name="Division Name", color="Towed_Rate",
            color_continuous_scale=["#2DD4BF", "#FF9F43", "#FF5470"],
            labels={"Towed_Rate": "Towed Rate (%)", "Zero_Tolerance_Rate": "Zero-Tolerance Rate (%)"},
        )
        fig5.update_layout(
            title="Division Risk Map — Towed Rate vs Zero-Tolerance Rate",
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=460, coloraxis_showscale=False,
        )
        st.plotly_chart(fig5, use_container_width=True)

# ============================================================================
# PAGE: ABOUT
# ============================================================================
elif page == "ℹ️ About Project":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">MODULE 05 · PROJECT INFO</div>
            <div class="hero-title">About This Project</div>
            <div class="hero-sub">What the system does, how it was built, and what data
            powers it.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hazard-strip"></div>', unsafe_allow_html=True)

    a1, a2 = st.columns([1.3, 1])
    with a1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Project Description")
        st.write(
            """
            The **Pune Smart Towing Intelligence System** helps drivers and traffic
            administrators understand towing risk before it happens. It is built on a
            real dataset of Pune traffic-division towing zones, capturing structural
            signals from each zone's description — proximity to junctions and bus
            stops, footpath or "both-sides-banned" violations, explicit no-parking
            language, market-area context, and more.

            A RandomForest classifier, trained on these engineered features, estimates
            the probability that a given road or location falls into a **zero-tolerance**
            enforcement zone — i.e., one where vehicles are towed immediately and
            without warning. The same dataset also drives a yard-assignment lookup, so
            a vehicle owner can immediately find out where a towed vehicle would be
            taken.
            """
        )
        st.markdown("#### Technologies Used")
        t1, t2, t3 = st.columns(3)
        t1.markdown("**Frontend**\n- Streamlit\n- Custom CSS\n- Plotly")
        t2.markdown("**ML / Data**\n- scikit-learn\n- RandomForestClassifier\n- pandas / numpy")
        t3.markdown("**Techniques**\n- Label Encoding\n- Keyword feature extraction\n- Class-balanced training")
        st.markdown("</div>", unsafe_allow_html=True)

    with a2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Dataset Summary")
        st.markdown(
            f"""
            <div class="mono" style="font-size:0.85rem; line-height:2;">
            FILE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Pune_With_Real_Features_v5.csv<br/>
            RECORDS &nbsp;&nbsp;&nbsp;&nbsp; {len(df)}<br/>
            COLUMNS &nbsp;&nbsp;&nbsp;&nbsp; {df.shape[1]}<br/>
            DIVISIONS &nbsp;&nbsp;&nbsp; {df['Division Name'].nunique()}<br/>
            YARDS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {df['Yard Name'].nunique()}<br/>
            TOWED RATE &nbsp; {df['Towed'].mean()*100:.1f}%<br/>
            ZERO-TOL RATE {df['zero_tolerance'].mean()*100:.1f}%
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("##### Feature columns")
        st.code(", ".join(FEATURES[:-2]) + ", div_enc, date_enc", language="text")
        st.markdown("##### Model")
        st.write(
            "`RandomForestClassifier(n_estimators=300, class_weight='balanced', "
            "random_state=42)` — validated accuracy ≈ 73% during research notebook "
            "experimentation across Logistic Regression, KNN, Decision Tree, SVM, "
            "AdaBoost, Gradient Boosting, and XGBoost baselines."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="panel" style="border-color: rgba(255,159,67,0.3);">
        <b>Note on Vehicle Type:</b> the underlying dataset does not contain a vehicle-type
        field, so this input is collected for contextual display only and does not
        influence the model's prediction — keeping the original ML logic, features,
        and outputs unchanged.
        </div>
        """,
        unsafe_allow_html=True,
    )
