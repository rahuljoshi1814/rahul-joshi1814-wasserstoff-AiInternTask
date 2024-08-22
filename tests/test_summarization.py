import unittest
import pandas as pd
import json
import os
import csv
import sys
import os

# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.summarization_model import load_csv_files, preprocess_dataframes, generate_summaries, save_summaries

class TestSummarization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Define file paths for test data and results
        cls.test_identification_file = 'data/test_identification.csv'
        cls.test_text_extraction_file = 'data/test_text_extraction.csv'
        cls.test_summary_file_csv = 'data/summaries.csv'
        cls.test_summary_file_json = 'data/summaries.json'

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(cls.test_identification_file), exist_ok=True)
        os.makedirs(os.path.dirname(cls.test_text_extraction_file), exist_ok=True)

        # Create dummy CSV files for testing
        identification_data = [
            {'master_id': 1, 'object_id': 1, 'file_path': 'data/segmented_objects/1_1.jpg', 'description': 'test object'},
            {'master_id': 1, 'object_id': 2, 'file_path': 'data/segmented_objects/1_2.jpg', 'description': 'another object'}
        ]
        text_extraction_data = [
            {'Image': '1_1.jpg', 'BBox': [[0, 0, 1, 1]], 'Text': 'Test', 'Confidence': 0.99},
            {'Image': '1_2.jpg', 'BBox': [[0, 0, 1, 1]], 'Text': 'Another Test', 'Confidence': 0.95}
        ]
        pd.DataFrame(identification_data).to_csv(cls.test_identification_file, index=False)
        pd.DataFrame(text_extraction_data).to_csv(cls.test_text_extraction_file, index=False)

    @classmethod
    def tearDownClass(cls):
        # Clean up files created during tests
        if os.path.isfile(cls.test_identification_file):
            os.remove(cls.test_identification_file)
        if os.path.isfile(cls.test_text_extraction_file):
            os.remove(cls.test_text_extraction_file)
        if os.path.isfile(cls.test_summary_file_csv):
            os.remove(cls.test_summary_file_csv)
        if os.path.isfile(cls.test_summary_file_json):
            os.remove(cls.test_summary_file_json)

    def test_load_csv_files(self):
        # Test loading CSV files
        identification_df, text_extraction_df = load_csv_files()

        self.assertFalse(identification_df.empty, "Identification DataFrame should not be empty")
        self.assertFalse(text_extraction_df.empty, "Text Extraction DataFrame should not be empty")

        self.assertIn('master_id', identification_df.columns, "Identification DataFrame should contain 'master_id'")
        self.assertIn('Image', text_extraction_df.columns, "Text Extraction DataFrame should contain 'Image'")

    def test_preprocess_dataframes(self):
        # Test preprocessing of dataframes
        identification_df, text_extraction_df = load_csv_files()
        preprocessed_df = preprocess_dataframes(identification_df, text_extraction_df)

        self.assertIn('Image', preprocessed_df.columns, "Preprocessed Identification DataFrame should contain 'Image'")

    def test_generate_summaries(self):
        # Test generation of summaries
        identification_df, text_extraction_df = load_csv_files()
        summary_df = generate_summaries(identification_df, text_extraction_df)

        self.assertFalse(summary_df.empty, "Summary DataFrame should not be empty")
        self.assertIn('master_id', summary_df.columns, "Summary DataFrame should contain 'master_id'")
        self.assertIn('Text', summary_df.columns, "Summary DataFrame should contain 'Text'")

    def test_save_summaries(self):
        # Test saving summaries
        identification_df, text_extraction_df = load_csv_files()
        summary_df = generate_summaries(identification_df, text_extraction_df)
        save_summaries(summary_df)

        # Check if results CSV and JSON files are created
        self.assertTrue(os.path.isfile(self.test_summary_file_csv), "Summary CSV file was not created")
        self.assertTrue(os.path.isfile(self.test_summary_file_json), "Summary JSON file was not created")

        # Validate CSV file content
        results_df = pd.read_csv(self.test_summary_file_csv)
        self.assertGreater(len(results_df), 0, "Summary CSV is empty")

        # Validate JSON file content
        with open(self.test_summary_file_json, 'r') as json_file:
            results_json = json.load(json_file)
        self.assertGreater(len(results_json), 0, "Summary JSON is empty")
        self.assertIsInstance(results_json[0], dict, "JSON records should be dictionaries")

if __name__ == '__main__':
    unittest.main()
