import unittest
import os
import pandas as pd
import json
import cv2
import numpy as np
import sys
import os

# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.text_extraction_model import extract_text, process_images_and_save_results

class TestTextExtraction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Define paths for test images and results
        cls.test_image_path = 'data/input_images/image_1.jpg'
        cls.test_text_extraction_results_csv = 'data/text_extraction_results.csv'
        cls.test_text_extraction_results_json = 'data/text_extraction_results.json'

        # Create directories and dummy image
        os.makedirs(os.path.dirname(cls.test_image_path), exist_ok=True)
        dummy_image = (np.ones((100, 100, 3), dtype=np.uint8) * 255)  # White image
        cv2.imwrite(cls.test_image_path, dummy_image)

    @classmethod
    def tearDownClass(cls):
        # Clean up files created during tests
        if os.path.isfile(cls.test_image_path):
            os.remove(cls.test_image_path)
        if os.path.isfile(cls.test_text_extraction_results_csv):
            os.remove(cls.test_text_extraction_results_csv)
        if os.path.isfile(cls.test_text_extraction_results_json):
            os.remove(cls.test_text_extraction_results_json)

    def test_extract_text(self):
        # Test text extraction functionality
        results = extract_text(self.test_image_path)
        
        # Since the image is white, we expect no text
        self.assertEqual(len(results), 0, "Text extraction results should be empty for a blank image")

    def test_process_images_and_save_results(self):
        # Test processing and saving results
        process_images_and_save_results()

        # Check if results CSV and JSON files are created
        self.assertTrue(os.path.isfile(self.test_text_extraction_results_csv), "Text extraction CSV file was not created")
        self.assertTrue(os.path.isfile(self.test_text_extraction_results_json), "Text extraction JSON file was not created")

        # Validate CSV file content
        results_df = pd.read_csv(self.test_text_extraction_results_csv)
        self.assertGreater(len(results_df), 0, "Text extraction CSV is empty")

        # Validate JSON file content
        with open(self.test_text_extraction_results_json, 'r') as json_file:
            results_json = json.load(json_file)
        self.assertGreater(len(results_json), 0, "Text extraction JSON is empty")
        self.assertIsInstance(results_json[0], dict, "JSON records should be dictionaries")

if __name__ == '__main__':
    unittest.main()
