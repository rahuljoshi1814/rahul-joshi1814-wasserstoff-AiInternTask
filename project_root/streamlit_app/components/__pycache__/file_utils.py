# components/file_utils.py

import pandas as pd
import json

def load_csv(file_path):
    """Load a CSV file into a DataFrame."""
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")

def save_to_csv(df, file_path):
    """Save a DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)

def save_to_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
