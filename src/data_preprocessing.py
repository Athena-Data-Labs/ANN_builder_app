import streamlit as st
import pandas as pd


def clean_preprocess():
    st.markdown("Explore your data and prepare it for model building")
    st.info(
        "🚧 This page is under development. Soon, you'll be able to clean and preprocess your data here, including handling missing values, removing duplicates, and preparing your dataset for modeling."
    )
    st.info(
        "Please continue to Data Exploration and Building the ANN. The rest of the app works as intended"
    )

    st.info(
        "create st.col for column in count(df.col)  create columns based o nthe number of columns and add the colum data st.dataframe(df[column]) to each col"
    )

    load_tab, process_tab = st.tabs(["Load Data", "Process Data"])

    with load_tab:
        st.markdown(
            """
            <h2 style='color:#4F8BF9; font-weight:700;'>📥 Load Data</h2>
            <div style='color:#555; font-size:1.1em; margin-bottom:10px;'>
                Upload your CSV file for processing. The data will be shown below if loaded successfully.
            </div>
            """,
            unsafe_allow_html=True,
        )
        raw_data = st.file_uploader("Upload Data for Processing", type=["csv"])
        raw_df = None
        if raw_data is not None:
            try:
                raw_df = pd.read_csv(raw_data)
                st.dataframe(raw_df)
                st.success("✅ File uploaded successfully!")
                st.session_state["raw_df"] = raw_df
            except Exception as e:
                st.error(f"❌ Could not read the file: {e}")
        else:
            st.info("Please upload a CSV file to get started.")

    with process_tab:
        st.markdown(
            """
            <h2 style='color:#4F8BF9; font-weight:700;'>🛠️ Data Formatting</h2>
            <div style='color:#555; font-size:1.1em; margin-bottom:10px;'>
                Use the header row to make basic formatting edits to your data.<br>
                <b>Tip:</b> Format numbers as currency (e.g., <span style='color:#4F8BF9;'>$</span>), dates, or categories as needed.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        raw_df = st.session_state.get("raw_df")
        if raw_df is not None:
            edited_df = st.data_editor(raw_df)
        else:
            st.warning("Please upload data in the 'Load Data' tab first.")


