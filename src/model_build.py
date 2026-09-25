# Load TensorFlow before the other numeric libraries to avoid a macOS deadlock.
from modeling.build import build_ann
import streamlit as st
import time
import io
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder, BinaryEncoder
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Normalizer,
)

from visualization.visualize import (
    plot_neural_network,
    cm_map,
    metrics_bar_chart,
    plot_error_metrics,
    plot_predicted_vs_actual,
    plot_cumulative_gain,
    plot_loss_curve,
)


def build():
    st.markdown(
        "Upload your dataset to begin the preprocessing and model-building process."
    )
    upload_tab, preprocess_tab, build_train_tab, predict_tab = st.tabs(
        [
            "Upload Training Data",
            "Preprocessing",
            "Build & Train Model",
            "Predict & Download Results",
        ]
    )

    with upload_tab:
        st.header("📂 Upload Your Dataset")

        uploaded_file = st.file_uploader(
            "📥 Upload data for model training (CSV only):", type="csv"
        )

        cleaned_df = st.session_state.get("processed_df")
        if isinstance(cleaned_df, pd.DataFrame) and not cleaned_df.empty:
            if st.button("Use cleaned dataset"):
                st.session_state["uploaded_df"] = cleaned_df.copy()
                st.success("✅ Cleaned dataset selected for model training.")

        if uploaded_file is not None:
            try:
                file_contents = uploaded_file.getvalue()
                if file_contents != st.session_state.get("training_upload"):
                    with st.spinner("Reading the uploaded file..."):
                        uploaded_df = pd.read_csv(uploaded_file)
                    if uploaded_df.empty:
                        st.warning("The CSV is empty. Upload a file with data rows.")
                        return
                    st.session_state["uploaded_df"] = uploaded_df
                    st.session_state["training_upload"] = file_contents

                uploaded_df = st.session_state["uploaded_df"]
                st.success("✅ File uploaded successfully!")

                st.markdown("### 📋 Data Preview")
                st.dataframe(uploaded_df.head(5), hide_index=True)

                num_feat = uploaded_df.select_dtypes(include=["float", "int"]).columns
                cat_feat = uploaded_df.select_dtypes(
                    include=["category", "object"]
                ).columns

                col1, col2 = st.columns([1, 1])
                with col1:
                    with st.expander("🔢 Numerical Features", expanded=False):
                        if len(num_feat) > 0:
                            st.write(f"**{len(num_feat)} Numerical Features Found:**")
                            st.table(pd.DataFrame({"**Numerical Features**": num_feat}))
                        else:
                            st.info("No numerical features found in the dataset.")

                with col2:
                    with st.expander("🔤 Categorical Features", expanded=False):
                        if len(cat_feat) > 0:
                            st.write(f"**{len(cat_feat)} Categorical Features Found:**")
                            st.table(
                                pd.DataFrame({"**Categorical Features**": cat_feat})
                            )
                        else:
                            st.info("No categorical features found in the dataset.")

            except Exception as e:
                st.error(f"⚠️ An error occurred while reading the file: {e}")
                return
        elif "uploaded_df" in st.session_state:
            st.info("Using the dataset already loaded in this session.")
            st.dataframe(st.session_state["uploaded_df"].head(5), hide_index=True)
        else:
            st.info("📥 Please upload a CSV file to proceed.")

    with preprocess_tab:
        st.header("⚙️ Configure Preprocessing")

        if "uploaded_df" not in st.session_state:
            st.warning(
                "⚠️ Please upload a dataset in the 'Upload Training Data' tab first."
            )
            return

        uploaded_df = st.session_state["uploaded_df"]

        if not uploaded_df.empty:
            with st.expander("🎯 Target Feature Selection", expanded=True):
                if st.session_state.get("model_target") not in uploaded_df.columns:
                    st.session_state.pop("model_target", None)
                target = st.selectbox(
                    "Choose Target Feature",
                    uploaded_df.columns,
                    key="model_target",
                    help="Select the column that represents the target variable.",
                )

                st.session_state["target"] = target

            with st.expander("🔀 Train-Test Split & Random State", expanded=True):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader("Train-Test Split")
                    test_size = st.slider(
                        "Select Test Size",
                        min_value=0.1,
                        max_value=0.9,
                        step=0.1,
                        value=0.3,
                        key="model_test_size",
                        help="Proportion of the dataset to include in the test split.",
                    )
                with col2:
                    st.subheader("Random State")
                    random_state = st.selectbox(
                        "Select Random State",
                        options=[None, 0, 1, 42, 100],
                        key="model_random_state",
                        help="Random seed for reproducibility.",
                    )

            with st.expander("📏 Scaling & 🔤 Encoding Options", expanded=False):
                col3, col4 = st.columns([1, 1])
                with col3:
                    st.subheader("Scaling Options")
                    scaler_option = st.selectbox(
                        "Select Scaler for Numerical Features",
                        key="model_scaler",
                        options=[
                            "StandardScaler",
                            "MinMaxScaler",
                            "MaxAbsScaler",
                            "RobustScaler",
                            "Normalizer",
                        ],
                        help="Choose a scaler to normalize numerical features.",
                    )
                with col4:
                    st.subheader("Encoding Options")
                    encoder_option = st.selectbox(
                        "Select Encoder for Categorical Features",
                        key="model_encoder",
                        options=[
                            "OneHotEncoder",
                            "OrdinalEncoder",
                            "TargetEncoder",
                            "BinaryEncoder",
                        ],
                        help="Choose an encoder to transform categorical features.",
                    )

            st.session_state["preprocessing"] = {
                "target": target,
                "test_size": test_size,
                "random_state": random_state,
                "scaler_option": scaler_option,
                "encoder_option": encoder_option,
            }

            st.success("✅ Preprocessing configuration saved successfully!")
            st.json(st.session_state["preprocessing"])

        else:
            st.warning("⚠️ Please upload a dataset in the 'Upload Data' tab first.")

    with build_train_tab:
        st.header("🛠️ Build and Train the Model")

        if "uploaded_df" not in st.session_state:
            st.warning(
                '⚠️ Please upload a dataset in the "Upload Training Data" tab first.'
            )
            return

        uploaded_df = st.session_state["uploaded_df"]

        if "preprocessing" in st.session_state:
            options = st.session_state["preprocessing"]
            target = options["target"]
            test_size = options["test_size"]
            random_state = options["random_state"]
            scaler_option = options["scaler_option"]
            encoder_option = options["encoder_option"]

            prepared_df = st.session_state.get("prepared_df")
            if (
                prepared_df is None
                or not uploaded_df.equals(prepared_df)
                or options != st.session_state.get("prepared_options")
            ):
                for key in (
                    "X_train", "X_test", "y_train", "y_test", "preprocessor",
                    "X_columns", "prepared_df", "prepared_options", "ann_model",
                    "loss_history", "y_pred", "prediction_results",
                ):
                    st.session_state.pop(key, None)

            if "X_train" not in st.session_state:
                if uploaded_df.shape[1] < 2:
                    st.warning("Choose a dataset with a target column and at least one input column.")
                    return
                if uploaded_df.isna().any().any():
                    st.warning("The dataset has missing values. Fill or remove them in Data Cleaning & Preprocessing, then use the cleaned dataset.")
                    return
                numeric_df = uploaded_df.select_dtypes(include=["number"])
                if not np.isfinite(numeric_df.to_numpy()).all():
                    st.warning("The dataset contains infinite values. Replace them before preprocessing.")
                    return
                if not pd.api.types.is_numeric_dtype(uploaded_df[target]):
                    st.warning("Choose a numeric target. Binary classification needs labels 0 and 1; regression needs numeric values.")
                    return
                try:
                    df_features = uploaded_df.drop(columns=[target])
                    num_feat = df_features.select_dtypes(include=["float", "int"]).columns
                    cat_feat = df_features.select_dtypes(
                        include=["category", "object", "bool"]
                    ).columns

                    scalers = {
                        "StandardScaler": StandardScaler(),
                        "MinMaxScaler": MinMaxScaler(),
                        "MaxAbsScaler": MaxAbsScaler(),
                        "RobustScaler": RobustScaler(),
                        "Normalizer": Normalizer(),
                    }
                    selected_scaler = scalers[scaler_option]

                    encoders = {
                        "OneHotEncoder": OneHotEncoder(
                            handle_unknown="ignore", sparse_output=False
                        ),
                        "OrdinalEncoder": OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                        "TargetEncoder": TargetEncoder(),
                        "BinaryEncoder": BinaryEncoder(),
                    }
                    selected_encoder = encoders[encoder_option]

                    preprocessor = ColumnTransformer(
                        transformers=[
                            ("num", selected_scaler, num_feat),
                            ("cat", selected_encoder, cat_feat),
                        ]
                    )

                    X = uploaded_df.drop(columns=[target])
                    y = uploaded_df[target]
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )

                    X_train = X_train.reset_index(drop=True)
                    X_test = X_test.reset_index(drop=True)
                    y_train = y_train.reset_index(drop=True)
                    y_test = y_test.reset_index(drop=True)

                    X_train = preprocessor.fit_transform(X_train, y_train)
                    X_test = preprocessor.transform(X_test)

                    X_train = np.array(X_train)
                    X_test = np.array(X_test)
                    y_train = np.array(y_train)
                    y_test = np.array(y_test)

                    if X_train.shape[1] == 0:
                        st.warning("No usable input features remain. Choose another target or encoding option.")
                        return
                    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
                        st.warning("Preprocessing produced missing or infinite values. Check the data and encoding options.")
                        return

                    st.session_state["X_train"] = X_train
                    st.session_state["y_train"] = y_train
                    st.session_state["X_test"] = X_test
                    st.session_state["y_test"] = y_test

                    st.session_state["preprocessor"] = preprocessor
                    st.session_state["X_columns"] = df_features.columns.tolist()
                    st.session_state["prepared_df"] = uploaded_df.copy()
                    st.session_state["prepared_options"] = options.copy()
                except (ValueError, TypeError) as e:
                    st.warning(f"Could not prepare the data. Check the target, split size and encoding options: {e}")
                    return

            config_tab, viz_train, test_tab = st.tabs(
                [
                    "⛭ Configure Model",
                    "⚒︎ Train & Visualize Model",
                    "⨖ Testing & Model Performance",
                ]
            )

            with config_tab:
                st.header("🧠 Configure Neural Network")

                with st.expander("🔢 Hidden Layer Configuration", expanded=True):
                    config_col1, config_col2 = st.columns([1, 1])
                    with config_col1:
                        st.write(
                            "Define the number of hidden layers and neurons per layer."
                        )
                        num_layers = st.number_input(
                            "Number of Hidden Layers",
                            key="model_num_layers",
                            min_value=1,
                            max_value=10,
                            value=3,
                            step=1,
                        )

                        layers_units = []
                        for i in range(num_layers):
                            neurons = st.number_input(
                                f"Neurons in Layer {i + 1}",
                                min_value=1,
                                max_value=512,
                                value=32,
                                step=1,
                                key=f"layer_{i}",
                            )
                            layers_units.append(neurons)
                    with config_col2:
                        st.write("Selected Hidden Layers Configuration:", layers_units)

                with st.expander("🔗 Output Layer Configuration", expanded=False):
                    st.write("Define the output layer parameters.")
                    output_units = st.number_input(
                        "Number of Output Neurons",
                        key="model_output_units",
                        min_value=1,
                        value=1,
                        step=1,
                    )
                    output_activation = st.selectbox(
                        "Activation Function for Output Layer",
                        key="model_output_activation",
                        options=["sigmoid", "softmax", "linear"],
                    )

                with st.expander("⚙️ Training Parameters", expanded=False):
                    st.write("Set the training parameters for the model.")
                    batch_size = st.number_input(
                        "Batch Size",
                        key="model_batch_size",
                        min_value=1,
                        value=32,
                        step=1,
                        help="Number of samples per gradient update.",
                    )
                    epochs = st.number_input(
                        "Epochs",
                        key="model_epochs",
                        min_value=1,
                        value=10,
                        step=1,
                        help="Number of epochs to train the model.",
                    )

                with st.expander("⚙️ Loss Function Configuration", expanded=False):
                    loss = st.selectbox(
                        "Choose the Loss Function for Model Training",
                        key="model_loss",
                        options=[
                            "binary_crossentropy",
                            "mean_squared_error",
                            "mean_absolute_error",
                            "categorical_crossentropy",
                        ],
                        help=(
                            "Select the appropriate loss function based on your task:\n"
                            "- `binary_crossentropy`: For binary classification tasks.\n"
                            "- `categorical_crossentropy`: For multi-class classification with one-hot encoded labels.\n"
                            "- `mean_squared_error`: For regression tasks.\n"
                            "- `mean_absolute_error`: For regression tasks with less sensitivity to outliers."
                        ),
                    )

                with st.expander(
                    "🔧 Activation Function for Hidden Layers", expanded=False
                ):
                    hidden_activation = st.selectbox(
                        "Activation Function for Hidden Layers",
                        key="model_hidden_activation",
                        options=["relu", "tanh", "sigmoid"],
                        help="Choose the activation function for the hidden layers.",
                    )

                with st.expander("📊 Prediction Threshold", expanded=False):
                    task_type = st.radio(
                        "Select Task Type",
                        key="model_task_type",
                        options=["Binary Classification", "Regression"],
                        help="Choose the type of task. For regression, no threshold is required.",
                    )

                    st.session_state["task_type"] = task_type

                    if task_type == "Binary Classification":
                        pred_threshold = st.number_input(
                            "Select Prediction Threshold",
                            key="model_pred_threshold",
                            min_value=0.1,
                            max_value=0.9,
                            value=0.5,
                            step=0.1,
                            help="Threshold for binary classification predictions.",
                        )
                    else:
                        pred_threshold = None

                st.session_state["model_config"] = {
                    "layers_units": layers_units,
                    "output_units": output_units,
                    "hidden_activation": hidden_activation,
                    "output_activation": output_activation,
                    "loss": loss,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "pred_threshold": pred_threshold,
                }
                st.success("✅ Model configuration saved successfully!")
                st.json(st.session_state["model_config"])

            training_config = st.session_state["model_config"].copy()
            training_config.pop("pred_threshold")
            training_config["task_type"] = task_type
            if training_config != st.session_state.get("training_config"):
                for key in ("ann_model", "loss_history", "y_pred", "prediction_results"):
                    st.session_state.pop(key, None)
                st.session_state["training_config"] = training_config

            training_error = None
            if output_units != 1:
                training_error = "Binary classification and regression require one output neuron for the selected target."
            elif task_type == "Binary Classification":
                if set(uploaded_df[target].unique()) != {0, 1}:
                    training_error = "Binary classification requires a target containing both 0 and 1. Choose another target or use regression."
                elif len(np.unique(st.session_state["y_train"])) != 2:
                    training_error = "The training split contains only one class. Adjust the test size or random state so both classes are present."
                elif output_activation != "sigmoid" or loss != "binary_crossentropy":
                    training_error = "For binary classification, select sigmoid output activation and binary_crossentropy loss."
            elif output_activation == "softmax" or loss not in ("mean_squared_error", "mean_absolute_error"):
                training_error = "For regression, use mean_squared_error or mean_absolute_error loss. A single softmax output is always 1; choose linear or sigmoid instead."
            elif output_activation == "sigmoid":
                st.info("Sigmoid limits regression predictions to 0–1. Use linear output activation for targets outside that range.")
            if len(st.session_state["X_train"]) < 2:
                training_error = "At least two training rows are needed for the validation split. Reduce the test size or add more data."

            if training_error:
                st.warning(training_error)

            with viz_train:
                st.header("📊 Visualization & Training")

                with st.expander("📈 Neural Network Visualization", expanded=True):
                    st.write(
                        "Visualize the structure of the configured neural network."
                    )
                    if st.button("Visualize Model", icon="📊"):
                        with st.spinner("Generating graph... Please wait."):
                            if "model_config" in st.session_state:
                                fig = plot_neural_network(
                                    df=uploaded_df,
                                    layers_units=st.session_state["model_config"][
                                        "layers_units"
                                    ],
                                    output_units=st.session_state["model_config"][
                                        "output_units"
                                    ],
                                    input_units=st.session_state["X_train"].shape[1],
                                )
                                st.pyplot(fig)
                                st.success("✅ Neural network visualization completed!")
                            else:
                                st.warning(
                                    "⚠️ Please configure the model in the 'Model Configuration' tab first."
                                )

                with st.expander("🚀 Model Training", expanded=True):
                    st.write(
                        "Train the configured neural network on the uploaded dataset."
                    )
                    if st.button("Build/Train Model", icon="🎬", disabled=training_error is not None):
                        if "model_config" in st.session_state:
                            for key in ("ann_model", "loss_history", "y_pred", "prediction_results"):
                                st.session_state.pop(key, None)
                            try:
                                with st.spinner("Training the model... Please wait."):
                                    X_train = st.session_state.get("X_train", None)
                                    y_train = st.session_state.get("y_train", None)

                                    if X_train is None or y_train is None:
                                        st.warning(
                                            "⚠️ Please preprocess and split the data in the 'Build & Train Model' tab first."
                                        )
                                        return

                                    ann_model = build_ann(
                                        X_train=X_train,
                                        y_train=y_train,
                                        layers_units=st.session_state["model_config"][
                                            "layers_units"
                                        ],
                                        output_units=st.session_state["model_config"][
                                            "output_units"
                                        ],
                                        hidden_activation=st.session_state[
                                            "model_config"
                                        ]["hidden_activation"],
                                        output_activation=st.session_state[
                                            "model_config"
                                        ]["output_activation"],
                                        loss=st.session_state["model_config"]["loss"],
                                        batch_size=st.session_state["model_config"][
                                            "batch_size"
                                        ],
                                        epochs=st.session_state["model_config"][
                                            "epochs"
                                        ],
                                        model_path=None,
                                        history_path=None,
                                    )
                                    st.session_state["ann_model"] = ann_model
                                    st.session_state["loss_history"] = ann_model.history.history
                                st.success("🎉 Model training completed successfully!")
                            except Exception as e:
                                st.error(
                                    f"⚠️ An error occurred during model training: {e}"
                                )

            with test_tab:
                st.header("📈 Model Performance")
                task_type = st.session_state.get("task_type", None)
                X_train = st.session_state.get("X_train", None)
                y_train = st.session_state.get("y_train", None)
                X_test = st.session_state.get("X_test", None)
                y_test = st.session_state.get("y_test", None)

                if task_type is None:
                    st.warning(
                        "⚠️ Please configure the model in the 'Build & Train Model' tab first."
                    )
                    return

                if X_test is None or y_test is None:
                    st.warning(
                        "⚠️ Please preprocess and split the data in the 'Build & Train Model' tab first."
                    )
                    return

                try:
                    with st.spinner("Model is making predictions... Please wait."):
                        if st.button("Run Model", icon="🏃‍♀️", disabled="ann_model" not in st.session_state):
                            st.markdown("---")

                            # Lazy import for TensorFlow-related modules
                            from modeling.predict import predict

                            start_time = time.time()
                            y_pred = predict(
                                X_input=X_test,
                                task_type=task_type,
                                pred_threshold=st.session_state["model_config"][
                                    "pred_threshold"
                                ],
                                ann_model=st.session_state["ann_model"],
                            )

                            # Ensure y_pred is 1D
                            if (
                                isinstance(y_pred, np.ndarray)
                                and y_pred.ndim > 1
                                and y_pred.shape[1] == 1
                            ):
                                y_pred = y_pred.ravel()

                            # st.write(f"DEBUG: y_test shape: {y_test.shape}")
                            # st.write(f"DEBUG: y_pred shape: {y_pred.shape}")

                            st.session_state["y_pred"] = y_pred

                            execution_time = time.time() - start_time

                            if task_type == "Binary Classification":
                                y_test = y_test.astype(int)
                                class_report = classification_report(
                                    y_test, y_pred, labels=[0, 1], output_dict=True, zero_division=0
                                )
                                class_labels = [0, 1]
                                acc_score = accuracy_score(y_pred=y_pred, y_true=y_test)
                                cm = confusion_matrix(y_pred=y_pred, y_true=y_test, labels=class_labels)
                                precision = class_report["weighted avg"]["precision"]
                                recall = class_report["weighted avg"]["recall"]
                                f1_score = class_report["weighted avg"]["f1-score"]

                            elif task_type == "Regression":
                                mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
                                mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
                                r2 = r2_score(y_true=y_test, y_pred=y_pred) if len(y_test) > 1 else np.nan

                            test_col1, test_col2 = st.columns([1, 2.5])

                            with test_col1:
                                st.subheader("📋 Key Metrics")
                                st.metric(
                                    "Execution Time", f"{execution_time:.2f} seconds"
                                )

                                if task_type == "Binary Classification":
                                    metrics_df = pd.DataFrame(
                                        {
                                            "Metric": [
                                                "Accuracy",
                                                "Precision",
                                                "Recall",
                                                "F1 Score",
                                            ],
                                            "Value": [
                                                f"{acc_score:.2%}",
                                                f"{precision:.2%}",
                                                f"{recall:.2%}",
                                                f"{f1_score:.2%}",
                                            ],
                                        }
                                    )
                                    st.table(metrics_df)

                                elif task_type == "Regression":
                                    metrics_df = pd.DataFrame(
                                        {
                                            "Metric": [
                                                "Mean Squared Error",
                                                "Mean Absolute Error",
                                                "R² Score",
                                            ],
                                            "Value": [
                                                f"{mse:.2f}",
                                                f"{mae:.2f}",
                                                f"{r2:.2f}",
                                            ],
                                        }
                                    )
                                    st.table(metrics_df)

                            with test_col2:
                                st.subheader("📊 Model Evaluation Metrics")

                                loss_history = st.session_state.get("loss_history", {})
                                train_loss = loss_history.get("loss", [])
                                val_loss = loss_history.get("val_loss", [])

                                with st.expander("Loss Curve", expanded=True):
                                    st.markdown("""
                                                **`Loss Curve Interpretation:`**
                                                - The loss curve shows how the model's error changes during training.
                                                - The blue line is the training loss; the orange line (if present) is the validation loss.
                                                - Ideally, both lines should decrease and stabilize. If the validation loss rises while training loss drops, the model may be overfitting.
                                                """)
                                    st.markdown("---")
                                    st.plotly_chart(
                                        plot_loss_curve(train_loss, val_loss)
                                    )

                                if task_type == "Binary Classification":
                                    with st.expander("Confusion Matrix", expanded=True):
                                        st.markdown("""
                                        **`Confusion Matrix Interpretation:`**
                                        - Shows the counts of true positives, true negatives, false positives, and false negatives.
                                        - Diagonal values indicate correct predictions; off-diagonal values indicate misclassifications.
                                        """)
                                        st.markdown("---")
                                        st.plotly_chart(
                                            cm_map(
                                                data_cm=cm, class_labels=class_labels
                                            ),
                                            use_container_width=True,
                                        )

                                    with st.expander(
                                        "Classification Metrics Bar Chart",
                                        expanded=True,
                                    ):
                                        st.markdown("""
                                        **`Classification Metrics Bar Chart Interpretation:`**
                                        - Visualizes key metrics such as precision, recall, and F1-score for each class.
                                        - Higher values indicate better model performance for that metric.
                                        """)
                                        st.markdown("---")
                                        st.plotly_chart(
                                            metrics_bar_chart(
                                                class_report=class_report
                                            ),
                                            use_container_width=True,
                                        )

                                elif task_type == "Regression":
                                    with st.expander(
                                        "Error Metrics Bar Chart", expanded=True
                                    ):
                                        st.markdown("""
                                        **`Error Metrics Bar Chart Interpretation:`**
                                        - Displays regression error metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² Score.
                                        - Lower error values and higher R² indicate better model performance.
                                        """)
                                        st.markdown("---")
                                        st.plotly_chart(
                                            plot_error_metrics(mse, mae, r2),
                                            use_container_width=True,
                                        )

                                    with st.expander(
                                        "Predicted vs. Actual Values", expanded=True
                                    ):
                                        st.markdown("""
                                        **`Predicted vs. Actual Values Interpretation:`**
                                        - Compares the model's predictions to the true values.
                                        - Points close to the diagonal line indicate accurate predictions.
                                        """)
                                        st.markdown("---")
                                        st.plotly_chart(
                                            plot_predicted_vs_actual(
                                                y_test=y_test, y_pred=y_pred
                                            ),
                                            use_container_width=True,
                                        )

                                    with st.expander(
                                        "Cumulative Gain Chart", expanded=True
                                    ):
                                        st.markdown("""
                                        **`Cumulative Gain Chart Interpretation:`**
                                        - Shows the cumulative share of actual values, starting with the highest predicted values.
                                        - A curve above the baseline means larger actual values tend to receive higher predictions.
                                        """)
                                        st.markdown("---")
                                        if np.any(y_test < 0) or np.sum(y_test) <= 0:
                                            st.info("Cumulative gain needs non-negative actual values with a positive total.")
                                        else:
                                            st.plotly_chart(
                                                plot_cumulative_gain(
                                                    y_test=y_test, y_pred=y_pred
                                                ),
                                                use_container_width=True,
                                            )
                        else:
                            st.info(
                                'Click "Build/Train Model" first, then "Run Model" to get model performance.'
                                if "ann_model" not in st.session_state else
                                'Click the "Run Model" button to get model performance.'
                            )

                except Exception as e:
                    st.error(f"⚠️ An error occurred while preparing the model: {e}")

    with predict_tab:
        st.header("📥 Predict & Download Results")

        if "ann_model" not in st.session_state:
            st.warning(
                "⚠️ Please train the model in the 'Build & Train Model' tab first."
            )
            return

        target = st.session_state.get("target", None)
        X_columns = st.session_state.get("X_columns", None)
        preprocessor = st.session_state.get("preprocessor", None)

        # st.write("Debug: X_columns in session state:", X_columns)
        # st.write("Debug: Preprocessor in session state:", preprocessor)

        if X_columns is None or preprocessor is None:
            st.warning(
                "⚠️ Please configure and train the model in the **'Build & Train Model'** tab first."
            )
            return

        with st.expander("📂 Upload Dataset for Predictions", expanded=True):
            new_data_file = st.file_uploader(
                "Upload new data for predictions:", type="csv"
            )

            if new_data_file is not None:
                try:
                    prediction_input = (
                        new_data_file.getvalue(),
                        st.session_state["model_config"]["pred_threshold"],
                    )
                    if prediction_input != st.session_state.get("prediction_input"):
                        st.session_state.pop("prediction_results", None)
                        st.session_state["prediction_input"] = prediction_input
                    new_data_df = pd.read_csv(new_data_file)
                    if new_data_df.empty:
                        st.warning("The prediction CSV is empty. Upload a file with data rows.")
                        return

                    missing_columns = [
                        col
                        for col in X_columns
                        if col != target and col not in new_data_df.columns
                    ]
                    if missing_columns:
                        st.warning(
                            f"⛔️ The uploaded file is missing required training columns: **{missing_columns}**. Add them before generating predictions."
                        )
                        return
                    if new_data_df[X_columns].isna().any().any():
                        st.warning("The prediction inputs have missing values. Fill or remove them before generating predictions.")
                        return

                    st.write("📋 Uploaded Data Preview:")
                    st.dataframe(new_data_df.head(3), hide_index=True)
                    st.success("✅ File uploaded successfully!")
                except Exception as e:
                    st.error(f"⚠️ An error occurred while reading the file: {e}")
                    return

        with st.expander("🔮 Generate Predictions", expanded=True):
            if new_data_file is None:
                st.session_state.pop("prediction_results", None)
                st.warning(
                    "⚠️ Please upload data in the **'Upload Dataset for Predictions'** section first."
                )
            else:
                st.info(
                    f"The model is prepared to predict the **{st.session_state.get('target', None)}**"
                )

                if st.button("Generate Predictions"):
                    st.session_state.pop("prediction_results", None)
                    try:
                        # Lazy import for TensorFlow-related modules
                        from modeling.predict import predict

                        new_processed_df = st.session_state.get(
                            "preprocessor", None
                        ).transform(new_data_df[X_columns])
                        if not np.isfinite(new_processed_df).all():
                            st.warning("The prediction inputs contain missing or infinite values after preprocessing. Check the uploaded data.")
                            return
                        predictions = predict(
                            X_input=new_processed_df,
                            task_type=st.session_state["task_type"],
                            pred_threshold=st.session_state["model_config"][
                                "pred_threshold"
                            ],
                            ann_model=st.session_state["ann_model"],
                        )
                        new_data_df[
                            f"Predicted {st.session_state.get('target', None)}"
                        ] = predictions
                        st.session_state["prediction_results"] = new_data_df
                    except Exception as e:
                        st.error(
                            f"⚠️ An error occurred during prediction or download preparation: {e}"
                        )

                if "prediction_results" in st.session_state:
                    new_data_df = st.session_state["prediction_results"]
                    st.success("✅ Predictions generated successfully!")
                    st.write(
                        f"📊 Predictions for the **{st.session_state.get('target', None)}** variable:"
                    )
                    st.dataframe(new_data_df, hide_index=True)

                    csv_buffer = io.StringIO()
                    new_data_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()

                    st.download_button(
                        label="Download Predictions",
                        data=csv_data,
                        file_name=f"Predictions_{st.session_state.get('target', 'target')}.csv",
                        mime="text/csv",
                    )
