import os
import cv2
import easyocr
import pandas as pd
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories
input_images_dir = 'data/input_images'
text_extraction_results_file_csv = 'data/text_extraction_results.csv'
text_extraction_results_file_json = 'data/text_extraction_results.json'

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def extract_text(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Extract text from the image
    results = reader.readtext(image)
    return results

def process_images_and_save_results():
    # Validate the input directory
    if not os.path.exists(input_images_dir):
        logging.error(f"Input directory {input_images_dir} does not exist.")
        return
    
    # Prepare a list to store results
    results_list = []

    # Get a list of image files in the input directory
    image_files = [f for f in os.listdir(input_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        logging.warning(f"No image files found in {input_images_dir}.")
        return

    # Extract text/data from each image
    for image_file in image_files:
        image_path = os.path.join(input_images_dir, image_file)
        try:
            results = extract_text(image_path)
            
            # Process the results and store them in the list
            for (bbox, text, prob) in results:
                bbox = [[float(coord) for coord in point] for point in bbox]  # Convert bbox coordinates to float
                prob = float(prob)  # Ensure confidence is a float
                
                results_list.append({
                    'Image': image_file,
                    'BBox': bbox,
                    'Text': text,
                    'Confidence': prob
                })
            logging.info(f"Processed {image_file} successfully.")
        except Exception as e:
            logging.error(f"Error processing {image_file}: {e}")

    # Convert list to DataFrame
    results_df = pd.DataFrame(results_list)

    # Save results to CSV file
    results_df.to_csv(text_extraction_results_file_csv, index=False)
    logging.info(f"Saved text extraction results to {text_extraction_results_file_csv}")

    # Save results to JSON file
    try:
        with open(text_extraction_results_file_json, 'w') as json_file:
            json.dump(results_list, json_file, indent=4)
        logging.info(f"Saved text extraction results to {text_extraction_results_file_json}")
    except TypeError as e:
        logging.error(f"Failed to save JSON results: {e}")
        
if __name__ == '__main__':
    process_images_and_save_results()
