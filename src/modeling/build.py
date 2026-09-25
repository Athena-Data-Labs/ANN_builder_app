import os
import numpy as np
import tensorflow as tf
import pickle

# Run the first TensorFlow operation before scikit-learn loads to avoid a
# training deadlock on macOS.
tf.constant(0)

def build_ann(
    X_train: np.ndarray,
    y_train: np.ndarray,
    layers_units: list[int],
    output_units: int,
    hidden_activation: str,
    output_activation: str,
    loss: str,
    batch_size: int,
    epochs: int,
    validation_split: float = 0.2,
    model_path: str = "models/ann_model.pkl",
    history_path: str = "reports/loss_history.pkl",
) -> tf.keras.models.Sequential:
    """
    Builds, trains, and saves an ANN with a variable number of units per layer.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        layers_units (list[int]): Units per hidden layer.
        output_units (int): Output layer units.
        hidden_activation (str): Activation for hidden layers.
        output_activation (str): Activation for output layer.
        loss (str): Loss function.
        batch_size (int): Batch size.
        epochs (int): Number of epochs.
        validation_split (float): Fraction of training data for validation.
        model_path (str): Path to save the model, or None to keep it in memory.
        history_path (str): Path to save the loss history, or None to skip saving.

    Returns:
        tf.keras.models.Sequential: Trained ANN model.
    """
    # Ensure output directories exist
    if model_path is not None:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    if history_path is not None:
        os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)

    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)))

    for units in layers_units:
        ann.add(tf.keras.layers.Dense(units=units, activation=hidden_activation))

    ann.add(tf.keras.layers.Dense(units=output_units, activation=output_activation))
    ann.compile(optimizer="adam", loss=loss)

    history = ann.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=validation_split,
    )

    # Save model
    if model_path is not None:
        with open(model_path, "wb") as file:
            pickle.dump(ann, file)

    # Save loss history
    if history_path is not None:
        with open(history_path, "wb") as f:
            pickle.dump(
                {
                    "train_loss": history.history["loss"],
                    "val_loss": history.history.get("val_loss", []),
                },
                f,
            )

    return ann

# df = pd.read_csv("data/processed/Churn.csv")
# df.head()

# num_cols = df.drop("Exited", axis=1).select_dtypes(include=["float", "int"]).columns
# cat_cols = df.drop("Exited", axis=1).select_dtypes("object").columns


# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(), num_cols),
#         ("cat", OneHotEncoder(), cat_cols),
#     ]
# )


# X = df.drop("Exited", axis=1)
# y = df["Exited"]


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.30, random_state=42
# )

# X_train = preprocessor.fit_transform(X_train)
# X_test = preprocessor.transform(X_test)


# X_train = np.array(X_train)
# X_test = np.array(X_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)


# ann = tf.keras.models.Sequential()
# ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
# ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
# ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
# ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# ann.compile(
#     optimizer="adam",
#     loss="binary_crossentropy",
#     metrics=[
#         tf.keras.metrics.BinaryAccuracy(name="accuracy"),
#         tf.keras.metrics.Precision(name="precision"),
#         tf.keras.metrics.Recall(name="recall"),
#         tf.keras.metrics.AUC(name="auc"),
#     ],
# )


# ann.fit(X_train, y_train, batch_size=32, epochs=160)
