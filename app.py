"""QNN Studio: a no-code workspace to learn and benchmark quantum neural networks.

    streamlit run app.py
"""
import streamlit as st

from qnn_platform import ui

st.set_page_config(page_title="QNN Studio", page_icon="⚛️", layout="wide", initial_sidebar_state="auto")
ui.inject_css()

navigation = st.navigation(
    [
        st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
        st.Page("views/research.py", title="Research Workbench", icon=":material/science:"),
        st.Page("views/learning.py", title="Learning Lab", icon=":material/school:"),
    ],
    position="top",
)
navigation.run()
