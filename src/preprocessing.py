import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_and_prepare_sequences(filepath):

    df = pd.read_csv(filepath)

    feature_cols = [
        "lms_logins",
        "assignment_delay",
        "attendance_rate",
        "sentiment_score",
        "activity_irregularity"
    ]

    # Sort properly
    df = df.sort_values(["student_id", "week"])

    students = df["student_id"].unique()

    X = []
    y_class = []
    y_reg = []

    for student in students:
        student_data = df[df["student_id"] == student]

        sequence = student_data[feature_cols].values

        # Burnout label (classification)
        burnout_label = student_data["burnout_flag"].iloc[0]

        # Dropout probability (regression target simulated)
        dropout_prob = 0.8 if burnout_label == 1 else 0.2

        X.append(sequence)
        y_class.append(burnout_label)
        y_reg.append(dropout_prob)

    X = np.array(X)
    y_class = np.array(y_class)
    y_reg = np.array(y_reg)

    # Scale features
    scaler = MinMaxScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(X.shape)

    # Train test split (student-wise)
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test