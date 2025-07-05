import pandas as pd
import json
import os
import csv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths
identification_results_file = 'data/descriptions.csv'
text_extraction_results_file = 'data/text_extraction_results.csv'
summary_results_file = 'data/summaries.csv'
summary_results_json_file = 'data/summaries.json'

def file_exists(file_path):
    """Check if a file exists."""
    return os.path.isfile(file_path)

def load_csv_files():
    if not file_exists(identification_results_file):
        logging.error(f"Identification results file not found: {identification_results_file}")
        raise FileNotFoundError(f"File not found: {identification_results_file}")
    
    if not file_exists(text_extraction_results_file):
        logging.error(f"Text extraction results file not found: {text_extraction_results_file}")
        raise FileNotFoundError(f"File not found: {text_extraction_results_file}")
    
    try:
        identification_df = pd.read_csv(identification_results_file, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
        text_extraction_df = pd.read_csv(text_extraction_results_file, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
        return identification_df, text_extraction_df
    except pd.errors.EmptyDataError:
        logging.error("One or more CSV files are empty.")
        raise
    except Exception as e:
        logging.error(f"Error loading CSV files: {e}")
        raise

def preprocess_dataframes(identification_df, text_extraction_df):
    # Extract filenames from file paths in identification_df
    identification_df['Image'] = identification_df['file_path'].apply(lambda x: os.path.basename(x))

    # Check if required columns are present
    required_id_columns = ['master_id', 'object_id', 'Image', 'description']
    required_text_columns = ['Image', 'BBox', 'Text', 'Confidence']

    missing_id_columns = [col for col in required_id_columns if col not in identification_df.columns]
    missing_text_columns = [col for col in required_text_columns if col not in text_extraction_df.columns]

    if missing_id_columns:
        logging.warning(f"Missing columns in identification results: {missing_id_columns}")
    if missing_text_columns:
        logging.warning(f"Missing columns in text extraction results: {missing_text_columns}")

    return identification_df

def generate_summaries(identification_df, text_extraction_df):
    # Merge results on 'Image' column
    merged_df = pd.merge(identification_df, text_extraction_df, on='Image', how='left')

    # Populate summary DataFrame
    summary_df = merged_df[['master_id', 'object_id', 'file_path', 'description', 'BBox', 'Text', 'Confidence']].copy()
    summary_df.fillna('N/A', inplace=True)  # Replace NaNs with 'N/A'

    return summary_df

def save_summaries(summary_df):
    try:
        # Save summary results to CSV
        summary_df.to_csv(summary_results_file, index=False)
        logging.info(f"Saved summary results to {summary_results_file}")

        # Save summary results to JSON
        summary_json = summary_df.to_dict(orient='records')
        with open(summary_results_json_file, 'w') as json_file:
            json.dump(summary_json, json_file, indent=4)
        logging.info(f"Saved summary results to {summary_results_json_file}")

    except Exception as e:
        logging.error(f"Error saving summaries: {e}")
        raise

if __name__ == '__main__':
    try:
        # Load CSV files
        identification_df, text_extraction_df = load_csv_files()

        # Log column names and sample rows for debugging
        logging.info(f"Identification DataFrame columns: {identification_df.columns}")
        logging.info(f"Text Extraction DataFrame columns: {text_extraction_df.columns}")
        logging.info("Sample rows from identification DataFrame:")
        logging.info(identification_df.head())
        logging.info("Sample rows from text extraction DataFrame:")
        logging.info(text_extraction_df.head())

        # Preprocess the dataframes
        identification_df = preprocess_dataframes(identification_df, text_extraction_df)

        # Generate summaries
        summary_df = generate_summaries(identification_df, text_extraction_df)

        # Save summaries
        save_summaries(summary_df)

    except Exception as e:
        logging.error(f"An error occurred during the summarization process: {e}")
