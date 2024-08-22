import unittest
import os
import pandas as pd
import json
import sys


# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.identification_model import identify_and_describe_object, process_all_segmented_objects


class TestIdentification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.metadata_file = 'data/metadata.csv'
        cls.descriptions_file = 'data/descriptions.csv'
        cls.descriptions_json_file = 'data/descriptions.json'
        cls.test_image = 'data/input_images/image_1.jpg'
        
        # Ensure test image exists
        if not os.path.isfile(cls.test_image):
            raise FileNotFoundError(f"Test image file not found: {cls.test_image}")

    def test_identify_and_describe_object(self):
        description = identify_and_describe_object(self.test_image)
        self.assertIsInstance(description, str, "Description should be a string")
        self.assertGreater(len(description), 0, "Description should not be empty")

    def test_process_all_segmented_objects(self):
        # Run the processing function
        process_all_segmented_objects()
        
        # Check if the descriptions CSV and JSON files are created
        self.assertTrue(os.path.isfile(self.descriptions_file), "Descriptions CSV file was not created")
        self.assertTrue(os.path.isfile(self.descriptions_json_file), "Descriptions JSON file was not created")
        
        # Check if descriptions CSV has content
        descriptions_df = pd.read_csv(self.descriptions_file)
        self.assertGreater(len(descriptions_df), 0, "Descriptions CSV file is empty")
        
        # Check if descriptions JSON has content
        with open(self.descriptions_json_file, 'r') as json_file:
            descriptions_data = json.load(json_file)
        self.assertGreater(len(descriptions_data), 0, "Descriptions JSON file is empty")

        # Validate descriptions in both CSV and JSON
        for file_path in descriptions_df['file_path']:
            self.assertTrue(os.path.isfile(file_path), f"Image file {file_path} not found")
        
        for item in descriptions_data:
            self.assertIn('description', item, "Description field is missing in JSON")
            self.assertIsInstance(item['description'], str, "Description should be a string")
            self.assertGreater(len(item['description']), 0, "Description should not be empty")

if __name__ == '__main__':
    unittest.main()

