import os
import cv2
import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Load the pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def process_image(image_path):
    """Process the image for segmentation using Mask R-CNN."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}. Please check the file path.")
    
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert the image to a tensor and normalize it
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(device)
    
    # Run the model on the image
    with torch.no_grad():
        outputs = model(image_tensor)
    
    return image_rgb, outputs[0]

def extract_and_save_objects(image_path, master_id):
    """Extract and save segmented objects from the image using Mask R-CNN."""
    try:
        # Process the image using Mask R-CNN
        image_rgb, output = process_image(image_path)
        
        # Extract the masks, bounding boxes, and labels
        masks = output['masks'].cpu().numpy()
        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        
        # Ensure the directory for segmented objects exists
        os.makedirs('data/segmented_objects/', exist_ok=True)
        
        # Initialize metadata list
        metadata_list = []

        for i, mask in enumerate(masks):
            # Create a binary mask for the current object
            binary_mask = mask[0] > 0.5

            # Apply the binary mask to the original image to get the segmented object
            segmented_object = image_rgb * np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
            
            # Define the path where the segmented object will be saved
            object_id = f'{master_id}_object_{i+1}'
            segmented_image_path = f'data/segmented_objects/{object_id}.png'
            
            # Save the segmented object
            cv2.imwrite(segmented_image_path, cv2.cvtColor(segmented_object.astype(np.uint8), cv2.COLOR_RGB2BGR))
            
            # Add metadata for the current object
            metadata_list.append({
                'file_path': segmented_image_path,
                'master_id': master_id,
                'object_id': object_id,
                'bbox': boxes[i].tolist(),  # Save bounding box coordinates
                'label': int(labels[i])  # Save label (object class)
            })
        
        # Save the metadata to a CSV file
        metadata_df = pd.DataFrame(metadata_list)
        metadata_file = 'data/metadata.csv'
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        
        if os.path.exists(metadata_file):
            existing_metadata_df = pd.read_csv(metadata_file)
            metadata_df = pd.concat([existing_metadata_df, metadata_df], ignore_index=True)
        
        metadata_df.to_csv(metadata_file, index=False)
        print(f"Metadata successfully saved to {metadata_file}")

    except Exception as e:
        print(f"Error during segmentation: {e}")
        raise

# Example usage
if __name__ == '__main__':
    extract_and_save_objects('data/input_images/sample_image.png', '1')








