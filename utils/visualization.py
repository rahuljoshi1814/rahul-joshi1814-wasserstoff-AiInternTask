import pandas as pd
import matplotlib.pyplot as plt
import json
from PIL import Image

def visualize_results(image_path, mapped_data_file, output_image_path):
    """Generate final output image with annotations and save."""
    try:
        # Load the mapped data
        with open(mapped_data_file, 'r') as json_file:
            mapped_data = json.load(json_file)
        
        # Load the original image
        image = Image.open(image_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        # Annotate the image with mapped data
        for data in mapped_data:
            for obj in data['text_data']:
                # Example: Add text annotations
                plt.text(10, 10, f"{data['object_id']}: {data['description']}", fontsize=12, color='red')
        
        # Save the annotated image
        plt.axis('off')
        plt.savefig(output_image_path, bbox_inches='tight')
        plt.close()
        print(f"Visualization successfully saved to {output_image_path}")
    
    except Exception as e:
        print(f"Error during visualization: {e}")
        raise

