# 🎓 Behavioural Early Warning System  
### Student Burnout & Dropout Risk Prediction using Multi-Task LSTM

AI-powered behavioural intelligence system for early identification of at-risk students.

---

## 🚀 Project Overview

This project implements a **Multi-Output LSTM-based Early Warning System** that predicts:

- 🔹 Burnout Probability (Binary Classification)
- 🔹 Dropout Risk Score (Regression scaled 0–100)

The system analyzes **12-week behavioural sequences** including:

- LMS Logins
- Assignment Delays
- Attendance Rate
- Sentiment Score
- Activity Irregularity

It integrates:
- Deep Learning
- Temporal Modeling
- Risk Segmentation
- Explainability (Perturbation-based Sensitivity)
- Interactive Streamlit Dashboard

---

## 🧠 Model Architecture

Input:  
`12 × 5 behavioural sequence per student`

Architecture:
- LSTM (128 units)
- Shared Dense Layer (64 units)
- Dual Output Heads:
  - Burnout → Sigmoid Activation
  - Dropout Risk → Linear Activation

Multi-task learning improves shared representation learning.

---

## 📊 Results

- Accuracy: 100% (Synthetic Dataset)
- F1 Score: 1.00
- Dropout MAE: ~0.025
- Risk Segmentation:
  - 72% Low Risk
  - 28% High Risk
  - 0% Moderate Risk

Performance reflects controlled synthetic behavioural data generation.

---

## 🔍 Explainability

Instead of static SHAP, this project implements:

- Perturbation-Based Sensitivity Analysis
- Radar Feature Influence Profile
- Weekly Temporal Heatmap

This preserves sequence order and aligns better with LSTM memory states.

---

## 📈 Dashboard Features

- KPI Summary Cards
- Top 10 High-Risk Leaderboard
- Risk Distribution Visualization
- Feature Trend Selector
- Behavioural Trigger Detection
- Automated Intervention Recommendations

Built using **Streamlit + Plotly**

---

## 🏗 Project Structure


behavioural-early-warning-system/
│
├── app/
│ └── streamlit_app.py
│
├── src/
│ ├── model.py
│ ├── preprocessing.py
│ └── intervention.py
│
├── data/
│ └── synthetic_data.csv
│
├── notebooks/
│
├── requirements.txt
└── README.md


---

## ▶️ How to Run

### 1️⃣ Clone Repository

git clone https://github.com/YOUR\_USERNAME/behavioural-early-warning-system.git

cd behavioural-early-warning-system


### 2️⃣ Install Dependencies

pip install -r requirements.txt


### 3️⃣ Run Dashboard

streamlit run app/streamlit_app.py


---

## 🌍 Future Scope

- Real institutional data integration
- Federated Learning for multi-campus privacy
- Real-time streaming risk prediction
- REST API deployment
- Cloud-native scaling

---

## 👨‍💻 Author

**Anieruth S**  
Integrated M.Tech CSE (Business Analytics)  
VIT Chennai  

---

## 📌 License

For academic and research demonstration purposes.
