import os
import numpy as np
import easyocr
import pandas as pd
import json
from PIL import Image
from easyocr import Reader

# Define directories and file paths
input_images_dir = 'data/input_images'
text_extraction_results_file_csv = 'data/text_extraction_results.csv'
text_extraction_results_file_json = 'data/text_extraction_results.json'

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # You can add more languages if needed


def extract_text(input_images_dir):
    """Extract text from images in the input directory and save results to CSV and JSON files."""
    try:
        reader = Reader(['en'])  # Initialize EasyOCR reader
        text_results = []

        for image_name in os.listdir(input_images_dir):
            image_path = os.path.join(input_images_dir, image_name)
            if os.path.isfile(image_path):
                result = reader.readtext(image_path)
                text_data = {
                    'file_path': image_path,
                    'texts': [text[1] for text in result]  # Extract text from results
                }
                text_results.append(text_data)

        # Save to CSV
        text_results_df = pd.DataFrame(text_results)
        text_results_df.to_csv(text_extraction_results_file_csv, index=False)
        
        # Save to JSON
        with open(text_extraction_results_file_json, 'w') as json_file:
            json.dump(text_results, json_file, indent=4)

        print(f"Text extraction complete. Results saved to {text_extraction_results_file_csv} and {text_extraction_results_file_json}.")
    except Exception as e:
        print(f"Error during text extraction: {e}")

def process_images(directory):
    """Process all images in the specified directory and return a DataFrame with results."""
    extracted_texts = []
    
    # Ensure directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return pd.DataFrame()
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Process each image file
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                text = extract_text(file_path)
                extracted_texts.append({
                    'file_path': file_path,
                    'text': text
                })
            except Exception as e:
                print(f"Error extracting text from {file_path}: {e}")
    
    return pd.DataFrame(extracted_texts)

def save_results(results_df):
    """Save the results DataFrame to CSV and JSON files."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(text_extraction_results_file_csv), exist_ok=True)
        
        # Save to CSV
        results_df.to_csv(text_extraction_results_file_csv, index=False)
        print(f"Text extraction results successfully saved to {text_extraction_results_file_csv}")
        
        # Save to JSON
        results_df.to_json(text_extraction_results_file_json, orient='records', lines=True, indent=4)
        print(f"Text extraction results successfully saved to {text_extraction_results_file_json}")
    except Exception as e:
        print(f"Error saving text extraction results: {e}")

def run_text_extraction():
    """Run text extraction on all images in the input directory."""
    results_df = process_images(input_images_dir)
    
    if results_df.empty:
        print(f"No images processed or no text extracted.")
        return
    
    save_results(results_df)

if __name__ == '__main__':
    run_text_extraction()




