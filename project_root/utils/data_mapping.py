import pandas as pd
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

identification_results_file = 'data/descriptions.csv'
text_extraction_results_file = 'data/text_extraction_results.csv'
data_mapping_file = 'data/output/data_mapping.json'
summary_results_file = 'data/summaries.csv'
original_images_folder = 'data/input_images/'
output_dir = 'data/output/table_and_annotated/'
data_mapping_file = 'data/output/data_mapping.json'

# Check if the data_mapping_file exists before trying to open it
if not os.path.exists(data_mapping_file):
    logging.warning(f"Data mapping file not found: {data_mapping_file}. Creating a placeholder.")
    data_mapping = {"images": []}  # Placeholder data structure

    # Optionally save this placeholder structure if you want to create the file
    os.makedirs(os.path.dirname(data_mapping_file), exist_ok=True)
    with open(data_mapping_file, 'w') as json_file:
        json.dump(data_mapping, json_file, indent=4)
    logging.info(f"Placeholder data mapping file created at: {data_mapping_file}")
else:
    with open(data_mapping_file, 'r') as json_file:
        data_mapping = json.load(json_file)['images']

os.makedirs(output_dir, exist_ok=True)

# Check if the data_mapping_file exists before trying to open it
if not os.path.exists(data_mapping_file):
    raise FileNotFoundError(f"Data mapping file not found: {data_mapping_file}")

# Load data
with open(data_mapping_file, 'r') as json_file:
    data_mapping = json.load(json_file)['images']

def load_and_prepare_data():
    try:
        identification_df = pd.read_csv(identification_results_file)
        text_extraction_df = pd.read_csv(text_extraction_results_file)

        # Extract base filenames from the file paths for matching
        identification_df['Base_Image'] = identification_df['file_path'].apply(lambda x: os.path.basename(x).split('_')[0])
        text_extraction_df['Base_Image'] = text_extraction_df['Image'].apply(lambda x: x.split('_')[0])

        logging.info("Data loaded and base filenames extracted.")
        return identification_df, text_extraction_df
    except Exception as e:
        logging.error(f"Error loading CSV files: {e}")
        raise

def merge_data(identification_df, text_extraction_df):
    try:
        # Merge on base image names
        merged_df = pd.merge(identification_df, text_extraction_df, on='Base_Image', how='left')

        # Check merged DataFrame
        logging.info("Merged DataFrame with base image names:")
        logging.info(merged_df.head(10))
        
        return merged_df
    except Exception as e:
        logging.error(f"Error merging DataFrames: {e}")
        raise

def create_data_mapping(merged_df):
    try:
        data_mapping = []
        for master_id, group in merged_df.groupby('master_id'):
            object_details = []
            for _, row in group.iterrows():
                item = {
                    'object_id': row['object_id'],
                    'file_path': row['file_path'],
                    'description': row['description'],
                    'text_data': {
                        'BBox': row.get('BBox', 'N/A'),  # Use 'N/A' for missing values
                        'Text': row.get('Text', 'N/A'),
                        'Confidence': row.get('Confidence', 'N/A')
                    }
                }
                object_details.append(item)
            
            data_mapping.append({
                'master_id': master_id,
                'object_details': object_details
            })

        logging.info("Data mapping structure prepared.")
        return data_mapping
    except Exception as e:
        logging.error(f"Error creating data mapping: {e}")
        raise

def save_data_mapping(data_mapping):
    try:
        # Save data mapping to JSON
        with open(data_mapping_file, 'w') as json_file:
            json.dump({"images": data_mapping}, json_file, indent=4)
        logging.info(f"Saved data mapping to {data_mapping_file}")
    except Exception as e:
        logging.error(f"Error saving data mapping: {e}")
        raise

if __name__ == '__main__':
    identification_df, text_extraction_df = load_and_prepare_data()

    merged_df = merge_data(identification_df, text_extraction_df)

    data_mapping = create_data_mapping(merged_df)

    save_data_mapping(data_mapping)


