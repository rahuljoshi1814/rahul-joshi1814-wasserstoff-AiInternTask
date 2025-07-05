import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
from models.segmentation_model import extract_and_save_objects, process_all_images

class TestSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the test environment and directories."""
        cls.test_dir = 'data/test_segmented_objects'
        cls.test_image_path = 'data/test_image.jpg'
        if not os.path.exists(cls.test_dir):
            os.makedirs(cls.test_dir)
        # Create a dummy image file for testing
        with open(cls.test_image_path, 'wb') as f:
            f.write(b'\x00\x00\x00\x00')  # Dummy content for the image

    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        if os.path.exists(cls.test_image_path):
            os.remove(cls.test_image_path)
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    @patch('models.segmentation_model.extract_and_save_objects')
    def test_extract_and_save_objects(self, mock_extract_and_save_objects):
        """Test the extract_and_save_objects function."""
        mock_extract_and_save_objects.return_value = None  # Mocking return value

        # Call the function with test data
        try:
            extract_and_save_objects(self.test_image_path, 'test_master_id')
            mock_extract_and_save_objects.assert_called_once_with(self.test_image_path, 'test_master_id')
        except Exception as e:
            self.fail(f"extract_and_save_objects raised an exception: {e}")

    @patch('models.segmentation_model.process_all_images')
    def test_process_all_images(self, mock_process_all_images):
        """Test the process_all_images function."""
        mock_process_all_images.return_value = None  # Mocking return value

        # Call the function
        try:
            process_all_images()
            mock_process_all_images.assert_called_once()
        except Exception as e:
            self.fail(f"process_all_images raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
