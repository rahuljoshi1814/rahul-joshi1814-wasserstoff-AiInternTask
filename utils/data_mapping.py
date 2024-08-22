import pandas as pd
import json

def map_data_to_objects(descriptions_file, text_extraction_file, summary_file, output_file):
    """Map extracted data and attributes to each object and save to a file."""
    try:
        # Load the data
        descriptions_df = pd.read_csv(descriptions_file)
        text_extraction_df = pd.read_csv(text_extraction_file)
        summary_df = pd.read_csv(summary_file)
        
        # Perform the mapping
        mapped_data = []
        for _, desc_row in descriptions_df.iterrows():
            master_id = desc_row['master_id']
            object_id = desc_row['object_id']
            description = desc_row['description']
            
            text_data = text_extraction_df[text_extraction_df['master_id'] == master_id]
            summary_data = summary_df[summary_df['master_id'] == master_id]
            
            mapped_data.append({
                'master_id': master_id,
                'object_id': object_id,
                'description': description,
                'text_data': text_data.to_dict(orient='records'),
                'summary': summary_data.to_dict(orient='records')
            })
        
        # Save mapped data to JSON
        with open(output_file, 'w') as json_file:
            json.dump(mapped_data, json_file, indent=4)
        
        print(f"Data mapping successfully saved to {output_file}")
    
    except Exception as e:
        print(f"Error during data mapping: {e}")
        raise
