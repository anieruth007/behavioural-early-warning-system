from src.preprocessing import load_and_prepare_sequences
from src.model import build_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from src.intervention import generate_intervention, recommend_action

# Load data
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = load_and_prepare_sequences(
    "data/synthetic_data.csv"
)

# Build model
model = build_model((X_train.shape[1], X_train.shape[2]))

# Train quickly (few epochs)
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

# Predict
class_pred, reg_pred = model.predict(X_test)

# Intervention logic
from src.intervention import generate_intervention, recommend_action

print("\nSample Behavioural Explanation:")

sample_seq = X_test[0]
triggers = generate_intervention(sample_seq)

for t in triggers:
    print("-", t)

print("\nRecommended Action:")
print(recommend_action(triggers))

# Convert classification to binary
class_pred_binary = (class_pred > 0.5).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_class_test, class_pred_binary))

print("\nClassification Report:")
print(classification_report(y_class_test, class_pred_binary))

# Convert regression to risk score
risk_score = reg_pred.flatten() * 100

print("\nSample Risk Scores (0-100):")
print(risk_score[:10])