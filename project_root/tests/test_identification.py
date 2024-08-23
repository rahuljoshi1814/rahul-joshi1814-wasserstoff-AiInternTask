import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
from models.identification_model import identify_and_describe_object, process_all_segmented_objects

class TestIdentification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the test environment and directories."""
        cls.test_dir = 'data/test_identified_objects'
        cls.test_metadata_file = 'data/test_metadata.csv'
        cls.test_descriptions_file = 'data/test_descriptions.csv'
        cls.test_descriptions_json_file = 'data/test_descriptions.json'
        if not os.path.exists(cls.test_dir):
            os.makedirs(cls.test_dir)
        # Create dummy files for testing
        with open(cls.test_metadata_file, 'w') as f:
            f.write('id,name\n1,test_object\n')
        with open(cls.test_descriptions_file, 'w') as f:
            f.write('id,description\n1,test_description\n')
        with open(cls.test_descriptions_json_file, 'w') as f:
            f.write('{"1": {"description": "test_description"}}\n')

    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        if os.path.exists(cls.test_metadata_file):
            os.remove(cls.test_metadata_file)
        if os.path.exists(cls.test_descriptions_file):
            os.remove(cls.test_descriptions_file)
        if os.path.exists(cls.test_descriptions_json_file):
            os.remove(cls.test_descriptions_json_file)
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    @patch('models.identification_model.identify_and_describe_object')
    def test_identify_and_describe_object(self, mock_identify_and_describe_object):
        """Test the identify_and_describe_object function."""
        mock_identify_and_describe_object.return_value = None  # Mocking return value

        # Call the function with test data
        try:
            identify_and_describe_object(self.test_metadata_file, self.test_descriptions_file, self.test_descriptions_json_file)
            mock_identify_and_describe_object.assert_called_once_with(self.test_metadata_file, self.test_descriptions_file, self.test_descriptions_json_file)
        except Exception as e:
            self.fail(f"identify_and_describe_object raised an exception: {e}")

    @patch('models.identification_model.process_all_segmented_objects')
    def test_process_all_segmented_objects(self, mock_process_all_segmented_objects):
        """Test the process_all_segmented_objects function."""
        mock_process_all_segmented_objects.return_value = None  # Mocking return value

        # Call the function
        try:
            process_all_segmented_objects()
            mock_process_all_segmented_objects.assert_called_once()
        except Exception as e:
            self.fail(f"process_all_segmented_objects raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()

