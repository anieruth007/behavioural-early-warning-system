from src.preprocessing import load_and_prepare_sequences
from src.model import build_model
import numpy as np

# Load data
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = load_and_prepare_sequences(
    "data/synthetic_data.csv"
)

# Build model
model = build_model((X_train.shape[1], X_train.shape[2]))

model.summary()

# Train
history = model.fit(
    X_train,
    {
        "burnout_output": y_class_train,
        "dropout_output": y_reg_train
    },
    validation_split=0.2,
    epochs=10,
    batch_size=32
)

# Evaluate
results = model.evaluate(
    X_test,
    {
        "burnout_output": y_class_test,
        "dropout_output": y_reg_test
    }
)

print("Test Results:", results)