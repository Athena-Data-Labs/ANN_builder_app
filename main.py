import streamlit as st
from data_exploration import data_exploration
from model_build import build
from about import about_page
from data_preprocessing import clean_preprocess

st.set_page_config(initial_sidebar_state='expanded')


def main():
    pages = {
        "🤖 About this App": about_page,
        '📝 Data Cleaning & Preprocessing': clean_preprocess,
        "📊 Data Exploration & Visualization": data_exploration,
        "🏗️ Build Artificial-Neural-Network": build,
    }

    st.sidebar.title("🧭 Navigation")
    selected_page = st.sidebar.selectbox("Select a Page", list(pages.keys()))

    st.sidebar.markdown('---')
    st.sidebar.markdown('**Version:** 0.9.0')
    st.sidebar.markdown('**Author:** Vahidin (Dean) Jupic')

    st.title(selected_page)
    pages[selected_page]()


if __name__ == "__main__":
    main()
