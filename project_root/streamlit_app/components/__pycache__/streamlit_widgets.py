# components/streamlit_widgets.py

import streamlit as st

def file_uploader(label, type):
    """Custom file uploader widget."""
    return st.file_uploader(label, type=type)

def button(label, key=None):
    """Custom button widget."""
    return st.button(label, key=key)