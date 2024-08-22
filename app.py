import os
import streamlit as st
from PIL import Image
import pandas as pd
import logging
import json
from models.segmentation_model import extract_and_save_objects
from models.identification_model import process_image_metadata, identify_objects
from models.text_extraction_model import extract_text
from models.summarization_model import generate_summaries, save_summaries

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define directories and file paths
input_images_dir = 'data/input_images'
segmented_objects_dir = 'data/segmented_objects'
metadata_file = 'data/metadata.csv'
descriptions_file = 'data/identification_description.csv'
descriptions_json_file = 'data/identification_description.json'
text_extraction_results_file_csv = 'data/text_extraction_results.csv'
text_extraction_results_file_json = 'data/text_extraction_results.json'
summary_results_file = 'data/summaries.csv'
summary_results_json_file = 'data/summaries.json'

def save_metadata(metadata_df):
    """Save metadata DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        metadata_df.to_csv(metadata_file, index=False)
        logging.info(f"Metadata successfully saved to {metadata_file}")
    except Exception as e:
        logging.error(f"Error saving metadata to CSV: {e}")

def ensure_directories_exist():
    """Ensure necessary directories exist."""
    directories = [input_images_dir, segmented_objects_dir]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Created directory: {directory}")

def display_image(image_path):
    """Display the uploaded image in the Streamlit app."""
    try:
        image = Image.open(image_path)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    except Exception as e:
        st.error(f"Error displaying image: {e}")
        logging.error(f"Error displaying image: {e}")

def handle_image_upload(image_file):
    """Save the uploaded image to the input images directory."""
    try:
        ensure_directories_exist()
        image_path = os.path.join(input_images_dir, image_file.name)
        with open(image_path, 'wb') as f:
            f.write(image_file.getvalue())
        st.success(f"Image successfully uploaded to {image_path}")
        logging.info(f"Image uploaded to {image_path}")
        return image_path
    except Exception as e:
        st.error(f"Error saving image: {e}")
        logging.error(f"Error saving image: {e}")
        return None

def run_segmentation(image_path):
    """Run segmentation on the uploaded image."""
    try:
        master_id = os.path.splitext(os.path.basename(image_path))[0]
        extract_and_save_objects(image_path, master_id)
        st.success(f"Segmentation complete. Metadata saved to {metadata_file}.")
        logging.info(f"Segmentation complete for {image_path}")
    except Exception as e:
        st.error(f"Error during segmentation: {e}")
        logging.error(f"Error during segmentation: {e}")

def run_identification():
    """Run object identification."""
    if not os.path.exists(metadata_file):
        st.error(f"Metadata file {metadata_file} does not exist.")
        logging.error(f"Metadata file {metadata_file} does not exist.")
        return

    try:
        metadata_df = pd.read_csv(metadata_file)
        all_descriptions = []

        for _, row in metadata_df.iterrows():
            file_path = row['file_path']
            if os.path.exists(file_path):
                description = identify_objects(file_path)
                all_descriptions.append({
                    'master_id': row['master_id'],
                    'object_id': row['object_id'],
                    'file_path': file_path,
                    'description': description
                })
            else:
                logging.error(f"File {file_path} does not exist.")
                st.error(f"File {file_path} does not exist.")

        descriptions_df = pd.DataFrame(all_descriptions)
        descriptions_df.to_csv(descriptions_file, index=False)
        with open(descriptions_json_file, 'w') as json_file:
            json.dump(all_descriptions, json_file, indent=4)

        st.success(f"Object identification complete. Descriptions saved to {descriptions_file} and {descriptions_json_file}.")
        logging.info(f"Object identification complete. Descriptions saved.")
    except Exception as e:
        st.error(f"Error during identification: {e}")
        logging.error(f"Error during identification: {e}")

def run_text_extraction():
    """Run text extraction on the uploaded images."""
    try:
        extract_text(input_images_dir)
        st.success(f"Text extraction complete. Results saved to {text_extraction_results_file_csv} and {text_extraction_results_file_json}.")
        logging.info("Text extraction complete.")
    except Exception as e:
        st.error(f"Error during text extraction: {e}")
        logging.error(f"Error during text extraction: {e}")

def run_summarization():
    """Run summarization based on identification and text extraction results."""
    try:
        identification_df, text_extraction_df = load_csv_files()
        if identification_df.empty or text_extraction_df.empty:
            st.error("One or both required CSV files are empty or could not be loaded.")
            return

        summary_df = generate_summaries(identification_df, text_extraction_df)
        save_summaries(summary_df)
        st.success(f"Summarization complete. Results saved to {summary_results_file} and {summary_results_json_file}.")
        logging.info("Summarization complete.")
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        logging.error(f"Error during summarization: {e}")

def load_csv_files():
    """Load CSV files for summarization."""
    try:
        identification_df = pd.read_csv(descriptions_file, encoding='utf-8')
        text_extraction_df = pd.read_csv(text_extraction_results_file_csv, encoding='utf-8')
        return identification_df, text_extraction_df
    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        logging.error(f"Error loading CSV files: {e}")
        return pd.DataFrame(), pd.DataFrame()

def main():
    st.title("AI Pipeline Testing")

    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_path = handle_image_upload(uploaded_file)
        if image_path:
            display_image(image_path)
            
            if st.sidebar.button("Run Segmentation"):
                run_segmentation(image_path)
            
            if st.sidebar.button("Run Identification"):
                run_identification()
            
            if st.sidebar.button("Run Text Extraction"):
                run_text_extraction()
            
            if st.sidebar.button("Run Summarization"):
                run_summarization()

if __name__ == "__main__":
    main()
