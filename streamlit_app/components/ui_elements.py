# components/ui_elements.py

import streamlit as st

def header(title):
    """Display a header with the specified title."""
    st.header(title)

def subheader(title):
    """Display a subheader with the specified title."""
    st.subheader(title)
    
def sidebar_button(label, key=None):
    """Add a button to the sidebar."""
    return st.sidebar.button(label, key=key)