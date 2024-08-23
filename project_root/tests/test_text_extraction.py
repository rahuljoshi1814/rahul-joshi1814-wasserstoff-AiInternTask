import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
from models.text_extraction_model import extract_text, process_images_and_save_results

class TestTextExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the test environment and directories."""
        cls.segmented_objects_dir = 'data/test_segmented_objects'
        cls.text_extraction_results_file_csv = 'data/test_text_extraction_results.csv'
        cls.text_extraction_results_file_json = 'data/test_text_extraction_results.json'
        if not os.path.exists(cls.segmented_objects_dir):
            os.makedirs(cls.segmented_objects_dir)
        # Create dummy files for testing
        with open(cls.text_extraction_results_file_csv, 'w') as f:
            f.write('id,text\n1,test_text\n')
        with open(cls.text_extraction_results_file_json, 'w') as f:
            f.write('{"1": {"text": "test_text"}}\n')

    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        if os.path.exists(cls.text_extraction_results_file_csv):
            os.remove(cls.text_extraction_results_file_csv)
        if os.path.exists(cls.text_extraction_results_file_json):
            os.remove(cls.text_extraction_results_file_json)
        if os.path.exists(cls.segmented_objects_dir):
            shutil.rmtree(cls.segmented_objects_dir)

    @patch('models.text_extraction_model.extract_text')
    def test_extract_text(self, mock_extract_text):
        """Test the extract_text function."""
        mock_extract_text.return_value = None  # Mocking return value

        try:
            extract_text(self.segmented_objects_dir, self.text_extraction_results_file_csv, self.text_extraction_results_file_json)
            mock_extract_text.assert_called_once_with(self.segmented_objects_dir, self.text_extraction_results_file_csv, self.text_extraction_results_file_json)
        except Exception as e:
            self.fail(f"extract_text raised an exception: {e}")

    @patch('models.text_extraction_model.process_images_and_save_results')
    def test_process_images_and_save_results(self, mock_process_images_and_save_results):
        """Test the process_images_and_save_results function."""
        mock_process_images_and_save_results.return_value = None  # Mocking return value

        try:
            process_images_and_save_results()
            mock_process_images_and_save_results.assert_called_once()
        except Exception as e:
            self.fail(f"process_images_and_save_results raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()

