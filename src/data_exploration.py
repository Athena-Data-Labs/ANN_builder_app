import pandas as pd
import streamlit as st
# import plotly.express as px
# import matplotlib.pyplot as plt

from visualization.visualize import (
    fig_histogram,
    fig_scatter,
    fig_heat_map,
    fig_box_plt,
    fig_area_chart,
    fig_pi_chart,
)


def data_exploration():
    st.markdown("Upload your dataset and explore it interactively with visualizations.")

    tab1, tab2, tab3 = st.tabs(
        ["📂 Upload Data", "📊 Data Exploration", "📈 Data Visualization"]
    )

    with tab1:
        st.header("📂 Upload Your Dataset")
        st.markdown("Upload a CSV file to begin exploring your data.")
        uploaded_file = st.file_uploader("Please upload a CSV file:", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state["df"] = df
                st.success("✅ File uploaded successfully!")
                st.write("### Data Preview")
                st.dataframe(df.head(5), use_container_width=True)
                st.write(
                    f"**Number of Rows:** {df.shape[0]} | **Number of Columns:** {df.shape[1]}"
                )
            except Exception as e:
                st.error(f"⚠️ An error occurred while reading the file: {e}")
        else:
            st.info("📥 Please upload a CSV file to proceed.")

    with tab2:
        st.header("📊 Data Exploration")
        st.markdown("Explore the structure and summary of your dataset.")

        if "df" in st.session_state:
            df = st.session_state["df"]

            st.subheader("🔍 Data Preview")
            with st.expander("📑 Data Preview"):
                st.dataframe(df.head(5))
            with st.expander("📈 Descriptive Statistics", expanded=False):
                st.dataframe(df.describe(), use_container_width=True)

            with st.expander("🚨 Missing Values"):
                missing_values = df.isnull().sum()
                if missing_values.sum() > 0:
                    st.write(missing_values[missing_values > 0])
                else:
                    st.success("No missing values found!")

            st.subheader("📊 Correlation Matrix")
            with st.expander("🔗 Correlation Matrix", expanded=False):
                st.markdown(
                    "The correlation matrix shows the relationships between numerical features."
                )
                num_df = df.select_dtypes(include=["number"])
                corr_matrix = num_df.corr()
                st.dataframe(corr_matrix, use_container_width=True)

            st.subheader("📋 Data Summary")

            with st.expander("Data Summary"):
                st.write(f"**Number of Rows:** {df.shape[0]}")
                st.write(f"**Number of Columns:** {df.shape[1]}")
                st.write("**Column Data Types:**")
                st.write(df.dtypes)
        else:
            st.warning("⚠️ Please upload a dataset in the **Upload Data** tab first.")

    with tab3:
        st.header("📈 Data Visualization")
        st.markdown("Create interactive visualizations to better understand your data.")

        if "df" not in st.session_state:
            st.warning("⚠️ Please upload a dataset in the **Upload Data** tab first.")
            return

        df = st.session_state["df"]

        with st.expander("🔧 Visualization Settings", expanded=True):
            st.markdown("Configure the settings for your visualizations below:")

            col1, col2, col3 = st.columns(3)
            with col1:
                target = st.selectbox(
                    "🎯 Select Target Variable",
                    [None] + df.columns.tolist(),
                    help="Choose the column that represents the target variable. It will be used for correlation, marker size, and box plots. Select 'None' to exclude a target variable.",
                )
            with col2:
                x = st.selectbox(
                    "📊 Select X-Axis Variable",
                    df.columns,
                    help="Choose the column to be used for the X-axis in visualizations.",
                )
            with col3:
                y = st.selectbox(
                    "📈 Select Y-Axis Variable",
                    df.columns,
                    help="Choose the column to be used for the Y-axis in visualizations.",
                )

            col4, col5 = st.columns([2, 1])
            with col4:
                color = st.selectbox(
                    "🎨 Select Color Variable",
                    [None] + df.columns.to_list(),
                    help="Choose the column to color the data points in visualizations.",
                )
            with col5:
                n_bins = st.slider(
                    "🔢 Number of Bins (for Histogram)",
                    min_value=3,
                    max_value=75,
                    value=35,
                    step=1,
                    help="Adjust the number of bins for the histogram visualization.",
                )
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "📊 Histogram",
                "📦 Box Plot",
                "📈 Area Chart",
                "🫧 Scatter Plot",
                "🔥 Heatmap",
                "🥧 Pi Chart",
            ]
        )

        with tab1:
            st.markdown(
                """
                <h2 style='color:#4F8BF9; font-weight:700;'>📊 Histogram</h2>
                <div style='color:#555; font-size:1.1em; margin-bottom:10px;'>
                    Explore the distribution of a <b>numerical variable</b> using a histogram.<br>
                    <span style='color:#4F8BF9;'>Select an <b>X variable</b></span> to generate the chart.<br>
                    Optionally, group bars by another column using the <b>color</b> option.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if x:
                st.markdown(
                    f"<h4 style='margin-top:0;'>Histogram of <span style='color:#4F8BF9;'>{x}</span></h4>",
                    unsafe_allow_html=True,
                )
                if color:
                    st.markdown(
                        f"<span style='color:#888;'>Grouped by <b>{color}</b></span>",
                        unsafe_allow_html=True,
                    )
                st.success(
                    "#### Histogram Interpretation\n"
                    "- 📈 **Distribution:** Shows the spread of values for the selected variable.\n"
                    "- 🏔️ **Shape:** Detect skewness, modality, and outliers.\n"
                    "- 📊 **Frequency:** Bar height = count in each range."
                )
                st.divider()
                st.plotly_chart(
                    fig_histogram(df=df, x=x, color=color, nbins=n_bins),
                    use_container_width=True,
                )
            else:
                st.warning(
                    "Please select an X variable (column) to generate the histogram."
                )

        with tab2:
            st.markdown(
                """
                <h2 style='color:#A259EC; font-weight:700;'>📦 Box Plot</h2>
                <div style='color:#555; font-size:1.1em; margin-bottom:10px;'>
                    Analyze the distribution of a <b>numerical variable</b> and detect outliers using a box plot.<br>
                    <span style='color:#A259EC;'>Select a <b>target variable</b></span> to generate the chart.<br>
                    Optionally, group boxes by another column using the <b>color</b> option.
                </div>
                """,
                unsafe_allow_html=True,
            )
            if target:
                st.markdown(
                    f"<h4 style='margin-top:0;'>Box Plot for <span style='color:#A259EC;'>{target}</span></h4>",
                    unsafe_allow_html=True,
                )
                if color:
                    st.markdown(
                        f"<span style='color:#888;'>Grouped by <b>{color}</b></span>",
                        unsafe_allow_html=True,
                    )
                st.success(
                    "#### Box Plot Interpretation\n"
                    "- 📦 **Distribution:** Visualizes the spread, median, and outliers.\n"
                    "- 📏 **IQR:** Box shows the interquartile range (IQR); line inside is the median.\n"
                    "- ⚠️ **Outliers:** Points outside whiskers are potential outliers."
                )
                st.divider()
                try:
                    st.plotly_chart(
                        fig_box_plt(df=df, column=target, group_by=color),
                        use_container_width=True,
                    )
                except ValueError as e:
                    st.warning(str(e))
            else:
                st.warning(
                    "Please select a target variable (numerical column) to generate the box plot."
                )

        with tab3:
            st.markdown(
                """
                <h2 style='color:#F24E1E; font-weight:700;'>📈 Area Chart</h2>
                <div style='color:#555; font-size:1.1em; margin-bottom:10px;'>
                    Visualize trends over time or across categories using an area chart.<br>
                    <span style='color:#F24E1E;'>Select <b>X</b> and <b>Y</b> variables</span> to generate the chart.<br>
                    Optionally, group areas by another column using the <b>color</b> option.
                </div>
                """,
                unsafe_allow_html=True,
            )
            if x and y:
                st.markdown(
                    f"<h4 style='margin-top:0;'>Area Chart: <span style='color:#F24E1E;'>{y}</span> vs <span style='color:#F24E1E;'>{x}</span></h4>",
                    unsafe_allow_html=True,
                )
                if color:
                    st.markdown(
                        f"<span style='color:#888;'>Grouped by <b>{color}</b></span>",
                        unsafe_allow_html=True,
                    )
                st.success(
                    "#### Area Chart Interpretation\n"
                    "- 📈 **Trend:** Shows how a numerical variable (Y) changes over another (X).\n"
                    "- 🗂️ **Cumulative:** Useful for cumulative totals or trends.\n"
                    "- 🎨 **Grouping:** Color reveals differences between categories."
                )
                st.divider()
                st.plotly_chart(
                    fig_area_chart(df=df, x=x, y=y, color=color),
                    use_container_width=True,
                )
            else:
                st.warning(
                    "Please select both X and Y variables (columns) to generate the area chart."
                )

        with tab4:
            st.markdown(
                """
                <h2 style='color:#2ECC71; font-weight:700;'>🫧 Scatter Plot</h2>
                <div style='color:#555; font-size:1.1em; margin-bottom:10px;'>
                    Visualize relationships between two <b>numerical variables</b> using a scatter plot.<br>
                    <span style='color:#2ECC71;'>Select <b>X</b> and <b>Y</b> variables</span> to generate the chart.<br>
                    Optionally, group points by <b>color</b> and adjust marker size by <b>target</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )
            if x and y:
                st.markdown(
                    f"<h4 style='margin-top:0;'>Scatter Plot: <span style='color:#2ECC71;'>{x}</span> vs <span style='color:#2ECC71;'>{y}</span></h4>",
                    unsafe_allow_html=True,
                )
                if color:
                    st.markdown(
                        f"<span style='color:#888;'>Grouped by <b>{color}</b></span>",
                        unsafe_allow_html=True,
                    )
                if target:
                    st.markdown(
                        f"<span style='color:#888;'>Marker size by <b>{target}</b></span>",
                        unsafe_allow_html=True,
                    )
                st.success(
                    "#### Scatter Plot Interpretation\n"
                    "- 🔗 **Relationship:** Shows the relationship between two numerical variables.\n"
                    "- 🧩 **Patterns:** Patterns may indicate correlation or clusters.\n"
                    "- 🎨 **Grouping:** Color/size reveals additional structure."
                )
                st.divider()
                st.plotly_chart(
                    fig_scatter(df=df, x=x, y=y, color=color, size=target),
                    use_container_width=True,
                )
            else:
                st.warning(
                    "Please select both X and Y variables (columns) to generate the scatter plot."
                )

        with tab5:
            st.markdown(
                """
                <h2 style='color:#FF6F61; font-weight:700;'>🔥 Heatmap</h2>
                <div style='color:#555; font-size:1.1em; margin-bottom:10px;'>
                    Analyze correlations between <b>numerical variables</b> using a heatmap.<br>
                    <span style='color:#FF6F61;'>Select a <b>target</b> variable</span> for targeted correlations, or view the full matrix.
                </div>
                """,
                unsafe_allow_html=True,
            )
            if target:
                st.markdown(
                    f"<h4 style='margin-top:0;'>Correlation Heatmap with <span style='color:#FF6F61;'>{target}</span></h4>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<h4 style='margin-top:0;'>Full Correlation Matrix Heatmap</h4>",
                    unsafe_allow_html=True,
                )
            st.success(
                "#### Heatmap Interpretation\n"
                "- 🟪 **Correlation:** Visualizes correlation between numerical variables.\n"
                "- 🌗 **Strength:** Darker/lighter = stronger positive/negative relationships.\n"
                "- 🕵️ **Insight:** Useful for identifying multicollinearity or feature relationships."
            )
            st.divider()
            try:
                st.plotly_chart(
                    fig_heat_map(df=df, target=target), use_container_width=True
                )
            except ValueError as e:
                st.warning(str(e))

        with tab6:
            st.markdown(
                """
                <h2 style='color:#F7B731; font-weight:700;'>🥧 Pie Chart</h2>
                <div style='color:#555; font-size:1.1em; margin-bottom:10px;'>
                    Visualize the distribution of values for a selected column as a pie chart.<br>
                    <span style='color:#F7B731;'>Select a <b>target</b> variable</span> to generate the chart.<br>
                    Optionally, group slices by another column using the <b>color</b> option.
                </div>
                """,
                unsafe_allow_html=True,
            )
            if target:
                st.markdown(
                    f"<h4 style='margin-top:0;'>Pie Chart for <span style='color:#F7B731;'>{target}</span></h4>",
                    unsafe_allow_html=True,
                )
                if color:
                    st.markdown(
                        f"<span style='color:#888;'>Grouped by <b>{color}</b></span>",
                        unsafe_allow_html=True,
                    )
                st.success(
                    "#### Pie Chart Interpretation\n"
                    "- 🥧 **Proportion:** Shows the proportion of each category within a variable.\n"
                    "- 📊 **Distribution:** Useful for visualizing categorical distributions.\n"
                    "- 🎨 **Grouping:** Color shows sub-category breakdowns."
                )
                st.divider()
                st.plotly_chart(
                    fig_pi_chart(df=df, target=target, color=color),
                    use_container_width=True,
                )
            else:
                st.warning(
                    "Please select a target variable (column) to generate the pie chart."
                )
                st.plotly_chart(
                    fig_pi_chart(df=df, target=None), use_container_width=True
                )
