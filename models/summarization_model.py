import pandas as pd
import json
import os

# Define file paths
descriptions_file = 'data/identification_description.csv'
text_extraction_results_file_csv = 'data/text_extraction_results.csv'
summary_results_file = 'data/summaries.csv'
summary_results_json_file = 'data/summaries.json'

def load_csv_files():
    """Load CSV files for summarization."""
    try:
        identification_df = pd.read_csv(descriptions_file, encoding='utf-8')
        text_extraction_df = pd.read_csv(text_extraction_results_file_csv, encoding='utf-8')
        
        if identification_df.empty or text_extraction_df.empty:
            raise ValueError("One or both DataFrames are empty.")
        
        return identification_df, text_extraction_df
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return pd.DataFrame(), pd.DataFrame()

def generate_summaries(identification_df, text_extraction_df):
    """Generate summaries based on identification and text extraction results."""
    try:
        # Debugging: Print DataFrame heads to understand structure
        print("Identification DataFrame head:")
        print(identification_df.head())
        print("Text Extraction DataFrame head:")
        print(text_extraction_df.head())
        
        # Check for 'file_path' column
        if 'file_path' not in identification_df.columns or 'file_path' not in text_extraction_df.columns:
            raise KeyError("'file_path' column is missing from one or both DataFrames.")
        
        # Merge DataFrames
        summary_df = pd.merge(identification_df, text_extraction_df, on='file_path', how='inner')
        
        # Debugging: Print merged DataFrame
        print("Merged DataFrame head:")
        print(summary_df.head())
        
        return summary_df
    except Exception as e:
        print(f"Error generating summaries: {e}")
        raise
def save_summaries(summary_df):
    """Save summaries DataFrame to CSV and JSON files."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(summary_results_file), exist_ok=True)
        
        # Save to CSV
        summary_df.to_csv(summary_results_file, index=False)
        print(f"Summarization results successfully saved to {summary_results_file}")
        
        # Save to JSON
        summary_df.to_json(summary_results_json_file, orient='records', lines=True, indent=4)
        print(f"Summarization results successfully saved to {summary_results_json_file}")
    except Exception as e:
        print(f"Error saving summaries: {e}")
        raise

def run_summarization():
    """Run summarization based on identification and text extraction results."""
    try:
        identification_df, text_extraction_df = load_csv_files()
        
        if identification_df.empty or text_extraction_df.empty:
            print("One or both required CSV files are empty or could not be loaded.")
            return

        summary_df = generate_summaries(identification_df, text_extraction_df)
        save_summaries(summary_df)
        print(f"Summarization complete. Results saved to {summary_results_file} and {summary_results_json_file}.")
    except Exception as e:
        print(f"Error during summarization: {e}")

if __name__ == '__main__':
    run_summarization()
