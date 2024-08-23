import os
import cv2
import torch
import numpy as np
import pandas as pd
import sys
import logging
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import preprocess_image
from utils.postprocessing import save_image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories
input_images_dir = 'data/input_images'
segmented_objects_dir = 'data/segmented_objects'
metadata_file = 'data/metadata.csv'

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)
model.eval()

def extract_and_save_objects(image_path, master_id, score_threshold=0.5):
    try:
        # Load and preprocess the image
        image_tensor = preprocess_image(image_path, device)
    except Exception as e:
        logging.error(f"Error in preprocessing image {image_path}: {e}")
        return []

    try:
        with torch.no_grad():
            prediction = model(image_tensor)
    except Exception as e:
        logging.error(f"Error in model prediction for image {image_path}: {e}")
        return []

    masks = prediction[0].get('masks', []).cpu().numpy()
    labels = prediction[0].get('labels', []).cpu().numpy()
    scores = prediction[0].get('scores', []).cpu().numpy()
    
    original_image = cv2.imread(image_path)
    original_height, original_width, _ = original_image.shape

    os.makedirs(segmented_objects_dir, exist_ok=True)

    metadata = []
    object_id = 1

    for i in range(len(masks)):
        if scores[i] > score_threshold:
            mask = masks[i, 0]
            mask = (mask > 0.5).astype(np.uint8)
            mask_resized = cv2.resize(mask, (original_width, original_height))
            masked_image = np.zeros_like(original_image)
            masked_image[mask_resized == 1] = original_image[mask_resized == 1]

            object_file_path = os.path.join(segmented_objects_dir, f'{master_id}_{object_id}.jpg')
            save_image(masked_image, object_file_path)

            metadata.append({
                'master_id': master_id,
                'object_id': object_id,
                'file_path': object_file_path
            })
            object_id += 1

    return metadata

def process_all_images():
    all_metadata = []
    for image_file in os.listdir(input_images_dir):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_images_dir, image_file)
            master_id = os.path.splitext(image_file)[0]
            logging.info(f"Processing image {image_file} with master ID {master_id}")
            metadata = extract_and_save_objects(image_path, master_id)
            all_metadata.extend(metadata)

    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_file, index=False)
    logging.info(f"Extraction and storage complete. Metadata saved to {metadata_file}.")

if __name__ == '__main__':
    process_all_images()
