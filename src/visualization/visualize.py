import pandas as pd
import numpy as np
import plotly.express as px
from matplotlib.figure import Figure
import plotly.graph_objects as go
from sklearn.metrics import r2_score


def fig_pi_chart(df: pd.DataFrame, target: str, color: str = None) -> go.Figure:
    """
    Creates a pie chart using Plotly Express for the specified target column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target (str): The column name to use for the pie chart values.
        color (str, optional): The column name to group or color the pie chart slices.

    Returns:
        go.Figure: The generated pie chart.
    """

    fig = px.pie(
        data_frame=df,
        names=target,
        color=color if color in df.columns else None,
        title=f"Pie Chart for {target}" + (f" grouped by {color}" if color else ""),
    )

    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def fig_area_chart(df: pd.DataFrame, x: str, y: str, color: str = None) -> go.Figure:
    """
    Creates a smooth area chart using Plotly Express.

    Args:
        df (pd.DataFrame): The input DataFrame.
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.
        color (str, optional): The column name for grouping areas by color. Defaults to None.

    Returns:
        go.Figure: The generated smooth area chart.
    """

    df = df.copy()
    df[y] = pd.to_numeric(df[y], errors="coerce")
    df = df.dropna(subset=[x, y])

    df = df.sort_values(by=x, ascending=True)

    df[y] = df[y].interpolate(method="linear")

    fig = px.area(
        data_frame=df,
        x=x,
        y=y,
        color=color,
        title=f"Area Chart: {y} vs {x}" + (f" by {color}" if color else ""),
        labels={x: x, y: y},
        line_shape="spline",
    )

    fig.update_layout(
        xaxis=dict(title=x, showgrid=False, zeroline=False),
        yaxis=dict(title=y, showgrid=True, zeroline=False),
        # margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


def fig_histogram(df: pd.DataFrame, x: str, color: str, nbins: int) -> go.Figure:
    """
    Creates a histogram using Plotly Express.

    Args:
        df (pd.DataFrame): The input data frame.
        x (str): The column name for the x-axis.
        color (str): The column name for the color grouping.
        nbins (int): The number of bins for the histogram.

    Returns:
        plotly.graph_objects.Figure: The generated histogram figure.
    """

    fig = px.histogram(
        data_frame=df,
        x=x,
        color=color,
        nbins=nbins,
        title=f"Distribution of {x} by {color} Status",
        # color_discrete_sequence=px.colors.qualitative.Set2,
    )
    return fig


def fig_scatter(
    df: pd.DataFrame, x: str, y: str, color: str = None, size: str = None
) -> go.Figure:
    """
    Creates a scatter plot using Plotly Express.

    Args:
        df (pd.DataFrame): The input DataFrame.
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.
        color (str, optional): The column name for color grouping. Defaults to None.
        size (str, optional): The column name for marker size. Defaults to None.

    Returns:
        go.Figure: The generated scatter plot.
    """
    # Generate a dynamic title
    title = (
        f"Scatter Plot: {x} vs {y}"
        + (f" by {color}" if color else "")
        + (f" with Marker Size {size}" if size else "")
    )

    # Create the scatter plot
    fig = px.scatter(
        data_frame=df,
        x=x,
        y=y,
        color=color,
        size=size,
        title=title,
        # color_discrete_sequence=px.colors.qualitative.Bold,
    )

    return fig


def fig_heat_map(
    df: pd.DataFrame,
    target: str = None,
    color_scale: str = "Viridis",
    # width: int = 800,
    # height: int = 800,
) -> go.Figure:
    """
    Creates a heatmap for the correlation matrix. If a target column is specified,
    it creates a heatmap for the correlation of all features with the target column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target (str, optional): The target column name for correlation. If None, the full correlation matrix is used.
        color_scale (str): The color scale for the heatmap (default is "Viridis").
        width (int): The width of the heatmap (default is 800).
        height (int): The height of the heatmap (default is 800).

    Returns:
        go.Figure: The generated heatmap figure.
    """
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        raise ValueError(
            "The DataFrame does not contain any numeric columns to compute correlations."
        )

    if target is None:
        # Generate heatmap for the full correlation matrix
        corr_matrix = numeric_df.corr().round(2)
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale=color_scale,
            # title="Correlation Heatmap",
            labels={"color": "Correlation"},
            # width=width,
            # height=height,
        )
    else:
        # Validate the target column
        if target not in df.columns:
            raise ValueError(
                f"The target column '{target}' does not exist in the DataFrame."
            )
        if target not in numeric_df.columns:
            raise ValueError(f"The target column {target} must be numeric.")

        # Generate heatmap for correlations with the target column
        corr_matrix_target = numeric_df.corr()[[target]].round(2)
        fig = px.imshow(
            corr_matrix_target,
            text_auto=True,
            color_continuous_scale=color_scale,
            # title=f"Correlation Heatmap with '{target}'",
            labels={"color": f"Correlation with {target}"},
            # width=width,
            # height=height,
        )

    return fig


def fig_box_plt(
    df: pd.DataFrame, column: str = None, group_by: str = None
) -> go.Figure:
    """
    Creates a box plot for numerical features in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str, optional): The specific column to plot. If None, all numerical columns are plotted.
        group_by (str, optional): The column to group by for coloring the boxes. Defaults to None.

    Returns:
        plotly.graph_objects.Figure: The generated box plot.
    """
    numerical_df = df.select_dtypes(include=["number"])

    if column:
        if column not in numerical_df.columns:
            raise ValueError(
                f"The column '{column}' is not numerical or does not exist in the DataFrame."
            )
        fig = px.box(
            df,
            y=column,
            color=group_by,
            title=f"Box Plot for {column}"
            + (f" grouped by {group_by}" if group_by else ""),
            # color_discrete_sequence=px.colors.qualitative.Bold,
        )
    else:
        numerical_df = df.select_dtypes(include=["number"])
        melted_df = numerical_df.melt(var_name="Feature", value_name="Value")

        fig = px.box(
            melted_df,
            color="Feature",
            title="Box Plot for Numerical Features",
            # color_discrete_sequence=px.colors.qualitative.Bold,
        )

    return fig


def metrics_bar_chart(class_report: dict) -> go.Figure:
    """
    Generate a bar chart for class-wise metrics using Plotly.

    Parameters:
    - class_report: Classification report as a dictionary (output of classification_report with output_dict=True).

    Returns:
    - fig: Plotly figure object for the bar chart.
    """
    import pandas as pd
    import plotly.graph_objects as go

    metrics_df = pd.DataFrame(class_report).transpose()
    # Remove 'accuracy', 'macro avg', 'weighted avg' rows
    metrics_df = metrics_df.loc[
        ~metrics_df.index.isin(["accuracy", "macro avg", "weighted avg"])
    ]
    metrics_df = metrics_df[["precision", "recall", "f1-score"]]

    # Sort classes if possible
    try:
        metrics_df = metrics_df.sort_index(key=lambda x: x.astype(str))
    except Exception:
        pass

    # colors = {"precision": "#1f77b4", "recall": "#ff7f0e", "f1-score": "#2ca02c"}

    fig = go.Figure()
    for metric in ["precision", "recall", "f1-score"]:
        fig.add_trace(
            go.Bar(
                x=metrics_df.index.astype(str),
                y=metrics_df[metric],
                name=metric.capitalize(),
                # marker_color=colors[metric],
                text=[f"{v:.2f}" for v in metrics_df[metric]],
                textposition="outside",
                hovertemplate=f"<b>%{{x}}</b><br>{metric.capitalize()}: %{{y:.3f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="Class-wise Metrics",
            # font=dict(size=16, family="Arial", color="black"),
        ),
        xaxis_title="Class",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.1], showgrid=True, zeroline=False),
        barmode="group",
        legend_title="Metric",
        # margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


def cm_map(data_cm: np.array, class_labels: list) -> go.Figure:
    """
    Generate a Plotly heatmap for the confusion matrix.

    Parameters:
    - data_cm: Confusion matrix (2D array or list).
    - class_labels: List of class labels for the target variable.

    Returns:
    - fig: Plotly figure object for the confusion matrix.
    """

    data_cm = np.array(data_cm)
    totals = data_cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(
        data_cm, totals, out=np.zeros_like(data_cm, dtype=float), where=totals != 0
    ) * 100
    annotations = [
        [f"{count}<br>{pct:.1f}%" for count, pct in zip(row_counts, row_pcts)]
        for row_counts, row_pcts in zip(data_cm, cm_percent)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=data_cm,
            x=class_labels,
            y=class_labels,
            colorscale="Blues",
            showscale=True,
            text=annotations,
            texttemplate="%{text}",
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            colorbar=dict(title="Count"),
            reversescale=False,
        )
    )

    fig.update_layout(
        title=dict(
            text="Confusion Matrix",
            # font=dict(size=16, family="Arial", color="black"),
        ),
        xaxis_title="Predicted Labels",
        yaxis_title="True Labels",
        xaxis=dict(
            tickmode="array",
            tickvals=class_labels,
            ticktext=class_labels,
            side="top",
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=class_labels,
            ticktext=class_labels,
            autorange="reversed",
            tickfont=dict(size=12),
        ),
        # margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


def plot_neural_network(
    df: pd.DataFrame, layers_units: int, output_units: int, input_units: int = None
) -> go.Figure:
    """
    Plots a neural network diagram with the input layer size based on the number of features in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        layers_units (list[int]): List of integers representing the number of neurons in each hidden layer.
        output_units (int): Number of neurons in the output layer.

    Returns:
        matplotlib.figure.Figure: The generated neural network plot.
    """

    if input_units is None:
        input_units = df.shape[1] - 1

    fig = Figure(figsize=(8, 4))
    ax = fig.subplots()
    layer_positions = (
        [0] + [i + 1 for i in range(len(layers_units))] + [len(layers_units) + 1]
    )
    max_neurons = max(input_units, max(layers_units, default=0), output_units)

    for i, layer in enumerate([input_units] + layers_units + [output_units]):
        for j in range(layer):
            ax.scatter(
                layer_positions[i],
                j - layer / 2 + max_neurons / 2,
                s=160,
                color="skyblue",
                edgecolor="black",
                linewidths=0.3,
            )

    # Plot connections
    for i in range(len(layer_positions) - 1):  # Iterate over layer pairs
        # Determine the number of neurons in the previous and current layers
        prev_layer = [input_units] + layers_units + [output_units]
        curr_layer = prev_layer[i + 1]  # Current layer size
        prev_layer = prev_layer[i]  # Previous layer size

        for j in range(prev_layer):  # Iterate over neurons in the previous layer
            for k in range(curr_layer):  # Iterate over neurons in the current layer
                ax.plot(
                    [layer_positions[i], layer_positions[i + 1]],
                    [
                        j - prev_layer / 2 + max_neurons / 2,
                        k - curr_layer / 2 + max_neurons / 2,
                    ],
                    color="gray",
                    linewidth=0.2,
                )

    ax.axis("off")
    return fig


def plot_error_metrics(mse, mae, r2) -> go.Figure:
    """
    Plots a bar chart for regression error metrics using Plotly.

    Args:
        mse (float): Mean Squared Error.
        mae (float): Mean Absolute Error.
        r2 (float): R² Score.

    Returns:
        plotly.graph_objects.Figure: The generated bar chart.
    """
    metrics = {
        "Metric": ["Mean Squared Error", "Mean Absolute Error", "R² Score"],
        "Value": [mse, mae, r2],
        "Short": ["MSE", "MAE", "R²"],
        "Color": ["#1f77b4", "#ff7f0e", "#2ca02c"],
    }

    fig = px.bar(
        metrics,
        x="Metric",
        y="Value",
        text="Value",
        color="Metric",
        color_discrete_map={
            # "Mean Squared Error": "#1f96b4",
            # "Mean Absolute Error": "#ff7f0e",
            "R² Score": "#33e833",
        },
        labels={"Value": "Metric Value", "Metric": "Error Metric"},
        title="Regression Error Metrics",
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>",
        showlegend=False,
    )

    fig.update_layout(
        title=dict(
            text="Regression Error Metrics",
            # font=dict(size=16, family="Arial", color="black"),
        ),
        xaxis=dict(
            title="Metric",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Value",
            showgrid=True,
            zeroline=False,
            rangemode="tozero",
        ),
        # margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


def plot_predicted_vs_actual(y_test, y_pred) -> go.Figure:
    """
    Plots the predicted vs. actual values for regression tasks.

    Args:
        y_test (array-like): Actual target values.
        y_pred (array-like): Predicted target values.

    Returns:
        plotly.graph_objects.Figure: The generated interactive plot.
    """

    y_test = np.ravel(y_test)
    y_pred = np.ravel(y_pred)
    residuals = np.abs(y_test - y_pred)

    df = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": y_pred,
            "Residual": residuals,
            "Index": np.arange(len(y_test)),
        }
    )

    r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan
    mae = np.mean(residuals)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    fig = px.scatter(
        df,
        x="Actual",
        y="Predicted",
        color="Residual",
        color_continuous_scale="Viridis",
        hover_data=["Residual", "Index"],
        title=f"Predicted vs. Actual Values<br>R²={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}",
        labels={"Actual": "Actual Values", "Predicted": "Predicted Values"},
        opacity=0.7,
    )

    min_val = min(df["Actual"].min(), df["Predicted"].min())
    max_val = max(df["Actual"].max(), df["Predicted"].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Ideal Fit (y = x)",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                "Predicted vs. Actual Values"
                f"<br><span style='font-size:12px'>R²={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}</span>"
            ),
            # font=dict(size=16, family="Arial", color="black"),
        ),
        xaxis=dict(
            title="Actual Values",
            showgrid=False,
            zeroline=True,
        ),
        yaxis=dict(
            title="Predicted Values",
            showgrid=True,
            zeroline=True,
        ),
        legend=dict(
            title="Legend",
            orientation="v",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1,
        ),
        # margin=dict(l=40, t=40, b=40, r=40),
        coloraxis_colorbar=dict(title="|Residual|"),
    )

    return fig


def plot_cumulative_gain(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model_color: str = "blue",
    baseline_color: str = "red",
) -> go.Figure:
    """
    Plots the cumulative gain chart for regression tasks using Plotly.

    Args:
        y_test (array-like): Actual target values.
        y_pred (array-like): Predicted target values.
        model_color (str): Color for the model curve.
        baseline_color (str): Color for the baseline curve.

    Returns:
        plotly.graph_objects.Figure: The generated cumulative gain chart.

    Note:
        Cumulative gain charts are more common for classification tasks.
        For regression, this plot shows the cumulative proportion of actual values
        as you move through the sorted predictions.
    """
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    if y_test.shape != y_pred.shape:
        raise ValueError("y_test and y_pred must have the same shape.")
    if y_test.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if not np.issubdtype(y_test.dtype, np.number) or not np.issubdtype(
        y_pred.dtype, np.number
    ):
        raise ValueError("Inputs must be numeric.")

    if np.any(y_test < 0) or np.sum(y_test) <= 0:
        raise ValueError("Cumulative gain needs non-negative actual values with a positive total.")

    sorted_indices = np.argsort(y_pred)[::-1]
    y_test_sorted = y_test[sorted_indices]

    cumulative_actual = np.r_[0, np.cumsum(y_test_sorted) / np.sum(y_test_sorted)]
    cumulative_predicted = np.linspace(0, 1, len(y_test_sorted) + 1)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=cumulative_predicted,
            y=cumulative_actual,
            mode="lines",
            name="Model (Cumulative Gain)",
            line=dict(color=model_color, width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Baseline (Random)",
            line=dict(color=baseline_color, dash="dash", width=2),
        )
    )

    fig.update_layout(
        title=dict(
            text="Cumulative Gain Chart",
            # font=dict(size=16, family="Arial", color="black"),
        ),
        xaxis=dict(
            title="Cumulative Predicted Proportion",
            showgrid=False,
            zeroline=True,
            range=[0, 1],
        ),
        yaxis=dict(
            title="Cumulative Actual Proportion",
            showgrid=True,
            zeroline=True,
            range=[0, 1],
        ),
        legend=dict(
            title="Legend",
            orientation="v",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1,
        ),
        # margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


def plot_loss_curve(
    train_loss: list, val_loss: list = None, title: str = "Loss Curve"
) -> go.Figure:
    """
    Plots the training (and optionally validation) loss curve over epochs.

    Args:
        train_loss (list or np.ndarray): Training loss values per epoch.
        val_loss (list or np.ndarray, optional): Validation loss values per epoch.
        title (str): Plot title.

    Returns:
        plotly.graph_objects.Figure: The generated loss curve plot.
    """
    epochs = list(range(1, len(train_loss) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_loss,
            mode="lines+markers",
            name="Training Loss",
            line=dict(color="blue"),
        )
    )
    if val_loss is not None:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=val_loss,
                mode="lines+markers",
                name="Validation Loss",
                line=dict(color="orange"),
            )
        )
    fig.update_layout(
        title=dict(
            text=title,
            # font=dict(size=16, family="Arial", color="black"),
        ),
        xaxis_title="Epoch",
        yaxis_title="Loss",
        xaxis=dict(showgrid=False, zeroline=True),
        yaxis=dict(showgrid=True, zeroline=True),
        legend_title="Curve",
        # margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig
