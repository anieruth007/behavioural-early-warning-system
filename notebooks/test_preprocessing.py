from src.preprocessing import load_and_prepare_sequences

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = load_and_prepare_sequences(
    "data/synthetic_data.csv"
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Classification target shape:", y_class_train.shape)
print("Regression target shape:", y_reg_train.shape)
