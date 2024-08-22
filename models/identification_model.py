import os
import json
import pandas as pd
import logging
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the CLIP model and processor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def identify_objects(image_path):
    """Identify objects in the image using CLIP."""
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Define possible object descriptions
        descriptions = ['a person', 'a car', 'a bicycle', 'a dog', 'a cat']
        text_inputs = processor(text=descriptions, return_tensors="pt", padding=True).to(device)

        # Get model predictions
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            text_features = model.get_text_features(**text_inputs)

        # Calculate similarity between image and text features
        similarity = torch.matmul(image_features, text_features.T)
        similarity_scores = similarity.squeeze().cpu().numpy()

        # Identify the most likely description
        max_index = similarity_scores.argmax()
        identified_description = descriptions[max_index]

        return identified_description
    except Exception as e:
        logging.error(f"Error identifying objects in image {image_path}: {e}")
        return None

def process_image_metadata(metadata_file, output_csv, output_json):
    """Process metadata file and identify objects in segmented images."""
    try:
        # Load metadata
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found at {metadata_file}")
        metadata_df = pd.read_csv(metadata_file)

        # Initialize lists for saving results
        results = []

        for _, row in metadata_df.iterrows():
            image_path = row['file_path']
            master_id = row['master_id']

            # Identify objects in the image
            if not os.path.exists(image_path):
                logging.warning(f"Image not found at {image_path}. Skipping.")
                continue

            description = identify_objects(image_path)
            if description is None:
                logging.warning(f"No description returned for image {image_path}.")
                continue

            # Update metadata with description
            result = {
                'file_path': image_path,
                'master_id': master_id,
                'description': description
            }
            results.append(result)

        # Save results to CSV and JSON
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_csv, index=False)
            logging.info(f"Identification results successfully saved to {output_csv}")

            with open(output_json, 'w') as f:
                json.dump(results, f, indent=4)
            logging.info(f"Identification results successfully saved to {output_json}")
        else:
            logging.warning("No results to save.")

    except FileNotFoundError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")

# Example usage
if __name__ == '__main__':
    metadata_file = 'data/metadata.csv'  # Path to metadata CSV
    output_csv = 'data/identification_description.csv'  # Output CSV file
    output_json = 'data/identification_description.json'  # Output JSON file
    process_image_metadata(metadata_file, output_csv, output_json)



