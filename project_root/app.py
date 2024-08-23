import os
import streamlit as st
import json
from models.segmentation_model import extract_and_save_objects, process_all_images
from models.identification_model import identify_and_describe_object
from models.text_extraction_model import extract_text
from models.summarization_model import load_csv_files, preprocess_dataframes, generate_summaries, save_summaries
from utils.data_mapping import load_and_prepare_data, merge_data, create_data_mapping, save_data_mapping
from utils.visualization import plot_image_with_annotations, generate_summary_table

# Define directories and file paths
segmented_objects_dir = 'data/segmented_objects'
metadata_file = 'data/metadata.csv'
descriptions_file = 'data/descriptions.csv'
descriptions_json_file = 'data/descriptions.json'
text_extraction_results_file_csv = 'data/text_extraction_results.csv'
text_extraction_results_file_json = 'data/text_extraction_results.json'
input_images_dir = 'data/input_images'
summary_results_file = 'data/summaries.csv'
data_mapping_file = 'data/output/data_mapping.json'
output_dir = 'data/output/table_and_annotated/'

def ensure_directory_exists(directory):
    """Ensure that the specified directory exists; create it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def display_image(image_path):
    """Display the uploaded image in the Streamlit app."""
    st.image(image_path, caption='Uploaded Image', use_column_width=True)

def handle_image_upload(uploaded_file):
    """Save the uploaded file to a temporary location and return the file path."""
    image_path = os.path.join(input_images_dir, uploaded_file.name)
    ensure_directory_exists(input_images_dir)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return image_path

def run_segmentation(image_path):
    """Run segmentation on the uploaded image."""
    try:
        ensure_directory_exists(segmented_objects_dir)
        extract_and_save_objects(image_path, segmented_objects_dir)
        process_all_images()
        st.success("Segmentation completed successfully.")
    except Exception as e:
        st.error(f"An error occurred during segmentation: {e}")

def run_identification():
    """Run identification on segmented objects."""
    try:
        ensure_directory_exists(segmented_objects_dir)
        identify_and_describe_object(segmented_objects_dir)  # Pass only the directory
        st.success("Identification completed successfully.")
    except Exception as e:
        st.error(f"An error occurred during identification: {e}")


def run_text_extraction():
    """Run text extraction on segmented objects."""
    try:
        ensure_directory_exists(input_images_dir)
        extract_text(input_images_dir)
        st.success("Text extraction completed successfully.")
    except Exception as e:
        st.error(f"An error occurred during text extraction: {e}")
        
def run_summarization():
    """Run summarization of results."""
    try:
        if not os.path.exists(metadata_file) or not os.path.exists(text_extraction_results_file_csv):
            st.warning("Required files for summarization are missing.")
            return
        
        identification_df, text_extraction_df = load_csv_files()
        identification_df = preprocess_dataframes(identification_df, text_extraction_df)
        summary_df = generate_summaries(identification_df, text_extraction_df)
        save_summaries(summary_df)
        st.success("Summarization completed and results saved.")
    except Exception as e:
        st.error(f"An error occurred during summarization: {e}")

def run_data_mapping():
    """Run data mapping."""
    try:
        identification_df, text_extraction_df = load_and_prepare_data()
        merged_df = merge_data(identification_df, text_extraction_df)
        data_mapping = create_data_mapping(merged_df)
        save_data_mapping(data_mapping)
        st.success("Data mapping completed successfully.")
    except Exception as e:
        st.error(f"An error occurred during data mapping: {e}")

def run_visualization():
    """Run visualization of results."""
    try:
        if not os.path.exists(data_mapping_file):
            st.warning(f"Data mapping file not found: {data_mapping_file}")
            return

        with open(data_mapping_file, 'r') as json_file:
            data_mapping = json.load(json_file)['images']

        ensure_directory_exists(output_dir)

        for entry in data_mapping:
            master_id = entry['master_id']
            objects = entry['object_details']

            base_image_name = f"{master_id}.jpg"
            original_image_path = os.path.join(input_images_dir, base_image_name)
            annotated_image_path = os.path.join(output_dir, f"annotated_{base_image_name}")
            summary_table_csv_path = os.path.join(output_dir, f"summary_{master_id}.csv")
            summary_table_json_path = os.path.join(output_dir, f"summary_{master_id}.json")

            if not os.path.exists(original_image_path):
                st.warning(f"Image not found: {original_image_path}")
                continue

            plot_image_with_annotations(original_image_path, objects, annotated_image_path)
            generate_summary_table(objects, summary_table_csv_path, summary_table_json_path)

        st.success("Visualization completed and results saved.")
    except Exception as e:
        st.error(f"An error occurred during visualization: {e}")

def main():
    """Main function to run the Streamlit app."""
    st.title("AI Image Processing and text_extraction Pipeline")

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_path = handle_image_upload(uploaded_file)
        display_image(image_path)

        if st.button("Run Segmentation"):
            run_segmentation(image_path)
        
        if st.button("Run Identification"):
            run_identification()
        
        if st.button("Run Text Extraction"):
            run_text_extraction()
        
        if st.button("Run Summarization"):
            run_summarization()
        
        if st.button("Run Data Mapping"):
            run_data_mapping()
        
        if st.button("Run Visualization"):
            run_visualization()

if __name__ == "__main__":
    ensure_directory_exists(input_images_dir)
    ensure_directory_exists(output_dir)
    main()
