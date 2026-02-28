import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import matplotlib.pyplot as plt

from src.preprocessing import load_and_prepare_sequences
from src.model import build_model
from src.intervention import generate_intervention, recommend_action

st.set_page_config(page_title="Behavioural Early Warning System", layout="wide")

st.title("🎓 Student Burnout & Dropout Risk Dashboard")

# =====================================
# LOAD DATA
# =====================================

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = load_and_prepare_sequences(
    "data/synthetic_data.csv"
)

# =====================================
# CACHE MODEL
# =====================================

@st.cache_resource
def train_model():
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(
        X_train,
        {
            "burnout_output": y_class_train,
            "dropout_output": y_reg_train
        },
        epochs=5,
        batch_size=32,
        verbose=0
    )
    return model

model = train_model()

# =====================================
# PREDICT ALL STUDENTS
# =====================================

class_preds, reg_preds = model.predict(X_test, verbose=0)

risk_scores = reg_preds.flatten() * 100
burnout_probs = class_preds.flatten()

students_df = pd.DataFrame({
    "Student_ID": range(len(risk_scores)),
    "Burnout_Probability": burnout_probs,
    "Dropout_Risk_Score": risk_scores
})

# =====================================
# KPI SUMMARY CARDS
# =====================================

high_risk = students_df[students_df["Dropout_Risk_Score"] > 70].shape[0]
moderate_risk = students_df[
    (students_df["Dropout_Risk_Score"] > 40) &
    (students_df["Dropout_Risk_Score"] <= 70)
].shape[0]
low_risk = students_df[students_df["Dropout_Risk_Score"] <= 40].shape[0]

kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric("🔴 High Risk Students", high_risk)
kpi2.metric("🟡 Moderate Risk Students", moderate_risk)
kpi3.metric("🟢 Low Risk Students", low_risk)

st.markdown("---")

# =====================================
# SIDEBAR FILTERS
# =====================================

st.sidebar.header("🔎 Filters")

risk_threshold = st.sidebar.slider(
    "Minimum Dropout Risk Score",
    0, 100, 20
)

burnout_threshold = st.sidebar.slider(
    "Minimum Burnout Probability",
    0.0, 1.0, 0.2
)

search_student = st.sidebar.text_input("Search Student ID")

filtered_df = students_df[
    (students_df["Dropout_Risk_Score"] >= risk_threshold) &
    (students_df["Burnout_Probability"] >= burnout_threshold)
]

if search_student:
    filtered_df = filtered_df[
        filtered_df["Student_ID"].astype(str).str.contains(search_student)
    ]

# =====================================
# STUDENT SELECTION
# =====================================

student_index = st.selectbox(
    "Select Student",
    filtered_df["Student_ID"].tolist()
)

selected_student = X_test[student_index]

# Predict selected student
class_pred, reg_pred = model.predict(selected_student.reshape(1, 12, 5), verbose=0)

burnout_prob = class_pred[0][0]
risk_score = reg_pred[0][0] * 100

# =====================================
# METRICS DISPLAY
# =====================================

col1, col2 = st.columns(2)

with col1:
    st.metric("Burnout Probability", f"{burnout_prob:.2f}")

with col2:
    if risk_score > 70:
        st.error(f"Dropout Risk Score: {risk_score:.2f} (High Risk)")
    elif risk_score > 40:
        st.warning(f"Dropout Risk Score: {risk_score:.2f} (Moderate Risk)")
    else:
        st.success(f"Dropout Risk Score: {risk_score:.2f} (Low Risk)")

# =====================================
# BEHAVIOURAL EXPLANATION
# =====================================

triggers = generate_intervention(selected_student)

st.subheader("Behavioural Risk Indicators")

for t in triggers:
    st.write("•", t)

st.subheader("Recommended Intervention")
st.success(recommend_action(triggers))

# =====================================
# INTERACTIVE FEATURE VISUALIZATION
# =====================================

st.subheader("Behavioural Trends Over 12 Weeks")

feature_names = [
    "LMS Logins",
    "Assignment Delay",
    "Attendance Rate",
    "Sentiment Score",
    "Activity Irregularity"
]

selected_feature = st.selectbox(
    "Select Feature to Visualize",
    feature_names
)

feature_index = feature_names.index(selected_feature)

fig = px.line(
    y=selected_student[:, feature_index],
    title=f"{selected_feature} Trend (12 Weeks)"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================
# RISK DISTRIBUTION
# =====================================

st.subheader("📈 Dropout Risk Distribution")

hist_fig = px.histogram(
    students_df,
    x="Dropout_Risk_Score",
    nbins=20,
    title="Distribution of Dropout Risk Scores"
)

st.plotly_chart(hist_fig, use_container_width=True)

# =====================================
# TOP 10 HIGH RISK LEADERBOARD
# =====================================

st.subheader("🚨 Top 10 High Risk Students")

top_10 = students_df.sort_values(
    by="Dropout_Risk_Score",
    ascending=False
).head(10)

st.dataframe(top_10, use_container_width=True)

# =====================================
# FILTERED STUDENTS TABLE
# =====================================

st.subheader("📊 Filtered Students Overview")

st.dataframe(
    filtered_df.sort_values(
        by="Dropout_Risk_Score",
        ascending=False
    ),
    use_container_width=True
)

# =====================================
# ADVANCED EXPLAINABILITY SECTION
# =====================================

st.subheader("🧠 Advanced Model Explainability")

def compute_feature_impact(student_data):
    baseline_pred = model.predict(
        student_data.reshape(1, 12, 5),
        verbose=0
    )[1][0][0]

    impacts = []

    for i in range(student_data.shape[1]):
        modified = student_data.copy()

        # Apply 5% perturbation
        modified[:, i] *= 1.05

        new_pred = model.predict(
            modified.reshape(1, 12, 5),
            verbose=0
        )[1][0][0]

        impact = abs(new_pred - baseline_pred)
        impacts.append(impact)

    return impacts


feature_names = [
    "LMS Logins",
    "Assignment Delay",
    "Attendance Rate",
    "Sentiment Score",
    "Activity Irregularity"
]

impacts = compute_feature_impact(selected_student)

# Normalize for better visualization
impacts = np.array(impacts)
impacts_normalized = impacts / (impacts.max() + 1e-6)

radar_df = pd.DataFrame({
    "Feature": feature_names,
    "Impact": impacts_normalized
})

radar_fig = px.line_polar(
    radar_df,
    r="Impact",
    theta="Feature",
    line_close=True,
    title="Feature Influence Profile (Normalized)"
)

radar_fig.update_traces(fill="toself")

st.plotly_chart(radar_fig, use_container_width=True)

# =====================================
# TIME-STEP IMPORTANCE HEATMAP
# =====================================

st.subheader("🔥 Weekly Risk Sensitivity Heatmap")

baseline_pred = model.predict(
    selected_student.reshape(1, 12, 5),
    verbose=0
)[1][0][0]

heatmap_data = np.zeros((12, 5))

for week in range(12):
    for feature in range(5):
        modified = selected_student.copy()
        modified[week, feature] *= 1.05

        new_pred = model.predict(
            modified.reshape(1, 12, 5),
            verbose=0
        )[1][0][0]

        heatmap_data[week, feature] = abs(new_pred - baseline_pred)

heatmap_df = pd.DataFrame(
    heatmap_data,
    columns=feature_names
)

heatmap_fig = px.imshow(
    heatmap_df,
    labels=dict(x="Feature", y="Week", color="Impact"),
    title="Sensitivity of Dropout Risk Across Weeks"
)

st.plotly_chart(heatmap_fig, use_container_width=True)