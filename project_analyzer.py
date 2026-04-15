import io
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Project Duration Analyzer",
    page_icon="📊",
    layout="wide",
)

# ----------------------------
# CLEAN PROFESSIONAL STYLE
# ----------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #ffffff;
}

/* Layout spacing */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
}

/* Headings */
h1, h2, h3 {
    color: #1f3b5c;
    font-weight: 600;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f5f7fa;
    border-right: 1px solid #e6e9ef;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e6e9ef;
    border-radius: 10px;
    padding: 12px;
}

/* Tables */
div[data-testid="stDataFrame"] {
    background-color: #ffffff;
    border-radius: 8px;
    border: 1px solid #e6e9ef;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background-color: #f5f7fa;
    border-radius: 6px 6px 0 0;
    color: #4a5568;
    padding: 8px 14px;
}

.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    border-bottom: 2px solid #2c6ca3;
    color: #1f3b5c;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Defaults
# ----------------------------
UNITS = ["days", "weeks", "hours", "months", "minutes"]

UNIT_TO_MINUTES = {
    "minutes": 1,
    "hours": 60,
    "days": 60 * 24,
    "weeks": 60 * 24 * 7,
    "months": 60 * 24 * 30,
}

DEFAULT_DATA = pd.DataFrame([
    {"Activity": "Design", "Label": "A", "Predecessors": "-", "Minimum": 16, "Average": 21, "Maximum": 26, "Unit": "days"},
    {"Activity": "Build prototype", "Label": "B", "Predecessors": "A", "Minimum": 3, "Average": 6, "Maximum": 9, "Unit": "days"},
    {"Activity": "Evaluate equipment", "Label": "C", "Predecessors": "A", "Minimum": 5, "Average": 7, "Maximum": 9, "Unit": "days"},
    {"Activity": "Test prototype", "Label": "D", "Predecessors": "B", "Minimum": 2, "Average": 3, "Maximum": 4, "Unit": "days"},
    {"Activity": "Write equipment report", "Label": "E", "Predecessors": "C,D", "Minimum": 4, "Average": 6, "Maximum": 8, "Unit": "days"},
    {"Activity": "Write methods report", "Label": "F", "Predecessors": "C,D", "Minimum": 6, "Average": 8, "Maximum": 10, "Unit": "days"},
    {"Activity": "Write final report", "Label": "G", "Predecessors": "E,F", "Minimum": 1, "Average": 2, "Maximum": 3, "Unit": "days"},
])

# ----------------------------
# Helpers
# ----------------------------
def parse_predecessors(text):
    if text in ["", "-", None]:
        return []
    return [x.strip() for x in text.split(",")]

def convert_to_minutes(df):
    df["Factor"] = df["Unit"].map(UNIT_TO_MINUTES)
    df["Min_std"] = df["Minimum"] * df["Factor"]
    df["Avg_std"] = df["Average"] * df["Factor"]
    df["Max_std"] = df["Maximum"] * df["Factor"]
    return df

def pert(a, m, b):
    return (a + 4*m + b) / 6

def simulate(df, n=10000):
    results = []
    for _ in range(n):
        durations = {}
        for _, row in df.iterrows():
            durations[row["Label"]] = np.random.triangular(row["Min_std"], row["Avg_std"], row["Max_std"])

        ef = {}
        for _, row in df.iterrows():
            preds = parse_predecessors(row["Predecessors"])
            start = max([ef[p] for p in preds], default=0)
            ef[row["Label"]] = start + durations[row["Label"]]

        results.append(max(ef.values()))
    return np.array(results)

def convert_back(values, unit):
    return values / UNIT_TO_MINUTES[unit]

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Settings")
    display_unit = st.selectbox("Display results in", UNITS, index=0)
    n_sims = st.slider("Simulations", 1000, 20000, 10000)

# ----------------------------
# UI
# ----------------------------
st.title("Project Duration Analyzer")

df = st.data_editor(
    DEFAULT_DATA,
    num_rows="dynamic",
    column_config={
        "Unit": st.column_config.SelectboxColumn("Unit", options=UNITS)
    }
)

df = convert_to_minutes(df)

if st.button("Run Analysis"):

    sim = simulate(df, n_sims)
    sim_display = convert_back(sim, display_unit)

    mean = sim_display.mean()
    p95 = np.percentile(sim_display, 95)

    c1, c2 = st.columns(2)
    c1.metric("Mean Duration", f"{mean:.2f} {display_unit}")
    c2.metric("P95 Duration", f"{p95:.2f} {display_unit}")

    # Histogram
    fig, ax = plt.subplots()
    ax.hist(sim_display, bins=30, color="#4c87b9", alpha=0.8)
    ax.axvline(mean, color="black", linestyle="--")
    ax.axvline(p95, color="red", linestyle=":")
    ax.set_title("Project Duration Distribution")
    ax.set_xlabel(display_unit)
    st.pyplot(fig)
