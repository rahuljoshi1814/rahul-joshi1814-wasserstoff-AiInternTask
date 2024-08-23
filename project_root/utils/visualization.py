import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# Define file paths
data_mapping_file = 'data/output/data_mapping.json'
summary_results_file = 'data/summaries.csv'
original_images_folder = 'data/input_images/'
output_dir = 'data/output/table_and_annotated/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load data mapping
with open(data_mapping_file, 'r') as json_file:
    data_mapping = json.load(json_file)['images']

def plot_image_with_annotations(image_path, objects, output_path):
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        return
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    for obj in objects:
        bbox_data = obj['text_data']['BBox']
        if pd.notna(bbox_data) and bbox_data not in ['N/A', 'NaN']:
            try:
                bbox = json.loads(bbox_data)
                if isinstance(bbox, list) and len(bbox) == 4:
                    x_min, y_min, x_max, y_max = bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]
                    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    annotation_text = f"{obj['description']} ({obj['text_data']['Text']})"
                    plt.text(x_min, y_min - 10, annotation_text,
                             bbox=dict(facecolor='yellow', alpha=0.5), fontsize=8, color='black')
                else:
                    print(f"Invalid BBox format for object {obj['object_id']} in {obj['file_path']}: {bbox}")
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Error parsing BBox for object {obj['object_id']} in {obj['file_path']}: {e}")

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_summary_table(objects, csv_output_path, json_output_path):
    summary_data = []
    for obj in objects:
        summary_data.append({
            'Object ID': obj['object_id'],
            'File Path': obj['file_path'],
            'Description': obj['description'],
            'BBox': obj['text_data']['BBox'],
            'Text': obj['text_data']['Text'],
            'Confidence': obj['text_data']['Confidence']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    summary_df.to_csv(csv_output_path, index=False)
    print(f"Saved summary table to CSV: {csv_output_path}")
    
    # Save as JSON
    with open(json_output_path, 'w') as json_file:
        json.dump(summary_data, json_file, indent=4)
    print(f"Saved summary table to JSON: {json_output_path}")

# Generate output for each master image
for entry in data_mapping:
    master_id = entry['master_id']
    objects = entry['object_details']
    
    base_image_name = f"{master_id}.jpg"
    original_image_path = os.path.join(original_images_folder, base_image_name)
    
    annotated_image_path = os.path.join(output_dir, f"annotated_{base_image_name}")
    summary_table_csv_path = os.path.join(output_dir, f"summary_{master_id}.csv")
    summary_table_json_path = os.path.join(output_dir, f"summary_{master_id}.json")
    
    if not os.path.exists(original_image_path):
        print(f"Image not found: {original_image_path}")
        continue
    
    plot_image_with_annotations(original_image_path, objects, annotated_image_path)
    
    generate_summary_table(objects, summary_table_csv_path, summary_table_json_path)

print(f"Annotated images and summary tables saved in {output_dir}")