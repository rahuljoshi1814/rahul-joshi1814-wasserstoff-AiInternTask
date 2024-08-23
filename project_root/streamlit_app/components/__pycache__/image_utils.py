from PIL import Image
import streamlit as st

def display_image(image_path):
    """Display an image in the Streamlit app."""
    image = Image.open(image_path)
    st.image(image, caption='Uploaded Image', use_column_width=True)

def save_uploaded_file(uploaded_file, save_path):
    """Save the uploaded file to a specified path."""
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getvalue())
