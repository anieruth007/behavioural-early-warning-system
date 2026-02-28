import numpy as np
import pandas as pd

np.random.seed(42)

NUM_STUDENTS = 5000
WEEKS = 12

data = []

for student_id in range(NUM_STUDENTS):
    base_engagement = np.random.normal(50, 10)
    burnout_trend = np.random.choice([0, 1], p=[0.7, 0.3])
    
    for week in range(WEEKS):
        lms_logins = base_engagement - week * np.random.uniform(0, 2) if burnout_trend else base_engagement + np.random.uniform(-5, 5)
        assignment_delay = np.random.normal(2 + week*0.3 if burnout_trend else 1, 1)
        attendance = np.random.normal(85 - week*2 if burnout_trend else 90, 5)
        sentiment = np.random.normal(-0.3 if burnout_trend else 0.2, 0.2)
        irregularity = np.random.uniform(0.5, 1.5) * week if burnout_trend else np.random.uniform(0.1, 0.5)

        data.append([
            student_id,
            week,
            max(lms_logins, 0),
            max(assignment_delay, 0),
            max(min(attendance,100),0),
            max(min(sentiment,1),-1),
            irregularity,
            burnout_trend
        ])

columns = [
    "student_id",
    "week",
    "lms_logins",
    "assignment_delay",
    "attendance_rate",
    "sentiment_score",
    "activity_irregularity",
    "burnout_flag"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("synthetic_data.csv", index=False)

print("Synthetic dataset generated successfully.")