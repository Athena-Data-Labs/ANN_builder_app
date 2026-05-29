import pandas as pd
import streamlit as st


def _get_active_df() -> pd.DataFrame | None:
    for key in ("processed_df", "raw_df", "df"):
        value = st.session_state.get(key)
        if isinstance(value, pd.DataFrame) and not value.empty:
            return value.copy()
    return None


def _store_dataframes(df: pd.DataFrame) -> None:
    st.session_state["processed_df"] = df.copy()
    st.session_state["df"] = df.copy()


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()

    for column in cleaned_df.columns:
        series = cleaned_df[column]
        if pd.api.types.is_numeric_dtype(series):
            median_value = series.median()
            if pd.notna(median_value):
                cleaned_df[column] = series.fillna(median_value)
        else:
            mode = series.mode(dropna=True)
            replacement_value = mode.iloc[0] if not mode.empty else "Unknown"
            cleaned_df[column] = series.fillna(replacement_value)

    return cleaned_df


def clean_preprocess():
    st.markdown(
        "Upload, inspect, and clean your dataset before building the ANN model."
    )

    load_tab, review_tab, clean_tab, export_tab = st.tabs(
        ["Load Data", "Review Data", "Clean Data", "Export Data"]
    )

    with load_tab:
        st.markdown(
            """
            <h2 style='color:#4F8BF9; font-weight:700;'>📥 Load Data</h2>
            <div style='color:#555; font-size:1.05em; margin-bottom:10px;'>
                Upload a CSV file to begin preprocessing. The file will be cached in the session
                so you can review, edit, and export it from the tabs below.
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.session_state["raw_df"] = raw_df.copy()
                _store_dataframes(raw_df)
                st.success("✅ File uploaded successfully!")
                st.dataframe(raw_df.head(10), use_container_width=True)
            except Exception as error:
                st.error(f"❌ Could not read the file: {error}")
        else:
            st.info("Upload a CSV file to start cleaning your data.")

    current_df = _get_active_df()
    if current_df is not None and st.session_state.get("raw_df") is None:
        st.session_state["raw_df"] = current_df.copy()

    with review_tab:
        st.markdown(
            """
            <h2 style='color:#4F8BF9; font-weight:700;'>📊 Review Data</h2>
            <div style='color:#555; font-size:1.05em; margin-bottom:10px;'>
                Inspect the shape, types, missing values, and duplicate rows before applying cleanup.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if current_df is None:
            st.warning("Upload a CSV file in the Load Data tab first.")
        else:
            metric_columns = st.columns(4)
            metric_columns[0].metric("Rows", f"{current_df.shape[0]:,}")
            metric_columns[1].metric("Columns", f"{current_df.shape[1]:,}")
            metric_columns[2].metric(
                "Missing Cells", f"{int(current_df.isna().sum().sum()):,}"
            )
            metric_columns[3].metric(
                "Duplicates", f"{int(current_df.duplicated().sum()):,}"
            )

            st.subheader("Preview")
            st.dataframe(current_df.head(10), use_container_width=True)

            summary_left, summary_right = st.columns(2)
            with summary_left:
                st.subheader("Column Types")
                st.dataframe(current_df.dtypes.rename("dtype"), use_container_width=True)
            with summary_right:
                st.subheader("Missing Values")
                missing_values = current_df.isna().sum()
                missing_values = missing_values[missing_values > 0]
                if missing_values.empty:
                    st.success("No missing values found.")
                else:
                    st.dataframe(missing_values.rename("missing_count"), use_container_width=True)

    with clean_tab:
        st.markdown(
            """
            <h2 style='color:#4F8BF9; font-weight:700;'>🧹 Clean Data</h2>
            <div style='color:#555; font-size:1.05em; margin-bottom:10px;'>
                Make manual edits in the table below, then apply automated cleanup options to save a clean copy.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if current_df is None:
            st.warning("Upload a CSV file in the Load Data tab first.")
        else:
            preview_df = st.data_editor(
                current_df,
                use_container_width=True,
                num_rows="dynamic",
                hide_index=True,
                key="preprocess_editor",
            )

            with st.form("cleanup_form"):
                columns_to_drop = st.multiselect(
                    "Columns to remove",
                    options=list(preview_df.columns),
                )
                missing_strategy = st.radio(
                    "Missing values strategy",
                    options=[
                        "Leave as-is",
                        "Fill numeric with median and categorical with mode",
                        "Drop rows with missing values",
                    ],
                    index=1,
                )
                drop_missing_subset = st.multiselect(
                    "Drop rows only when these columns are missing",
                    options=list(preview_df.columns),
                    default=[],
                    help="Only used when dropping rows with missing values.",
                )
                drop_duplicates = st.checkbox("Remove duplicate rows", value=True)
                apply_cleanup = st.form_submit_button("Apply Cleanup")

            if apply_cleanup:
                cleaned_df = preview_df.copy()

                if columns_to_drop:
                    cleaned_df = cleaned_df.drop(columns=columns_to_drop, errors="ignore")

                if missing_strategy == "Fill numeric with median and categorical with mode":
                    cleaned_df = _fill_missing_values(cleaned_df)
                elif missing_strategy == "Drop rows with missing values":
                    if drop_missing_subset:
                        cleaned_df = cleaned_df.dropna(subset=drop_missing_subset)
                    else:
                        cleaned_df = cleaned_df.dropna()

                if drop_duplicates:
                    cleaned_df = cleaned_df.drop_duplicates()

                _store_dataframes(cleaned_df)
                st.success("✅ Cleanup applied and saved for the rest of the app.")
                st.dataframe(cleaned_df.head(10), use_container_width=True)

    with export_tab:
        st.markdown(
            """
            <h2 style='color:#4F8BF9; font-weight:700;'>📤 Export Data</h2>
            <div style='color:#555; font-size:1.05em; margin-bottom:10px;'>
                Download the latest cleaned dataset or reset back to the original upload.
            </div>
            """,
            unsafe_allow_html=True,
        )

        export_df = _get_active_df()
        if export_df is None:
            st.warning("Upload a CSV file in the Load Data tab first.")
        else:
            export_col, reset_col = st.columns(2)
            with export_col:
                csv_data = export_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download cleaned CSV",
                    data=csv_data,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with reset_col:
                if st.button("Reset to original upload", use_container_width=True):
                    original_df = st.session_state.get("raw_df")
                    if isinstance(original_df, pd.DataFrame):
                        _store_dataframes(original_df)
                        st.success("Dataset reset to the original upload.")
                        st.rerun()


