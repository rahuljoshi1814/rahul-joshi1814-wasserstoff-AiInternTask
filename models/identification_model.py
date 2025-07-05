import torch
import clip
from PIL import Image
import pandas as pd
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories
segmented_objects_dir = 'data/segmented_objects'
metadata_file = 'data/metadata.csv'
descriptions_file = 'data/descriptions.csv'
descriptions_json_file = 'data/descriptions.json'

# Define textual descriptions
descriptions = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
                'truck', 'boat', 'traffic light', 'fire hydrant', 'flower', 'stop sign', 
                'mobile phone', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
                'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'lion', 'backpack', 
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
                'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
                'knife', 'spoon', 'tiger', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
                'hair drier', 'toothbrush']

# Load CLIP model and preprocess function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
text_inputs = clip.tokenize(descriptions).to(device)  # Tokenize once and move to device

def identify_and_describe_object(image_path):
    try:
        # Load and preprocess the image
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        
        # Get the image and text features
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_inputs)
            
            # Calculate similarity between image and text features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Get the top description
        top_description = descriptions[similarity[0].argmax().item()]
        return top_description

    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None

def process_all_segmented_objects():
    try:
        metadata_df = pd.read_csv(metadata_file)
    except FileNotFoundError:
        logging.error(f"Metadata file {metadata_file} not found.")
        return
    
    all_descriptions = []

    for _, row in metadata_df.iterrows():
        object_image_path = row['file_path']
        description = identify_and_describe_object(object_image_path)
        
        if description:  # Check if description is valid
            all_descriptions.append({
                'master_id': row['master_id'],
                'object_id': row['object_id'],
                'file_path': object_image_path,
                'description': description
            })

    # Save descriptions to CSV
    descriptions_df = pd.DataFrame(all_descriptions)
    descriptions_df.to_csv(descriptions_file, index=False)
    logging.info(f"Descriptions saved to {descriptions_file}.")

    # Save descriptions to JSON
    with open(descriptions_json_file, 'w') as json_file:
        json.dump(all_descriptions, json_file, indent=4)
    logging.info(f"Descriptions saved to {descriptions_json_file}.")

if __name__ == '__main__':
    process_all_segmented_objects()
