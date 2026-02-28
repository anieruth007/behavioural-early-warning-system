import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


def build_model(input_shape):

    inputs = Input(shape=input_shape)

    x = LSTM(64, return_sequences=False)(inputs)
    x = Dropout(0.3)(x)

    # Classification head
    class_output = Dense(32, activation="relu")(x)
    class_output = Dense(1, activation="sigmoid", name="burnout_output")(class_output)

    # Regression head
    reg_output = Dense(32, activation="relu")(x)
    reg_output = Dense(1, activation="linear", name="dropout_output")(reg_output)

    model = Model(inputs=inputs, outputs=[class_output, reg_output])

    model.compile(
        optimizer="adam",
        loss={
            "burnout_output": "binary_crossentropy",
            "dropout_output": "mse"
        },
        metrics={
            "burnout_output": ["accuracy"],
            "dropout_output": ["mae"]
        }
    )

    return model