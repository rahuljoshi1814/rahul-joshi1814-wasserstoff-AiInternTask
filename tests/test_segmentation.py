import unittest
import os
import cv2
import numpy as np
import pandas as pd
import torch
import sys
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn

# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.segmentation_model import extract_and_save_objects, process_all_images
from utils.preprocessing import preprocess_image
from utils.postprocessing import save_image

class TestSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up any necessary files or directories
        cls.test_image_path = 'data/input_images/test_image.jpg'
        cls.test_segmented_image_path = 'data/segmented_objects/test_image_1.jpg'
        cls.test_metadata = 'data/metadata.csv'
        cls.master_id = 'test_image'
        
        # Create a dummy image for testing
        os.makedirs(os.path.dirname(cls.test_image_path), exist_ok=True)
        dummy_image = np.ones((800, 800, 3), dtype=np.uint8) * 255  # White image, size adjusted to 800x800
        cv2.imwrite(cls.test_image_path, dummy_image)

        # Initialize and configure the model
        cls.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        cls.model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).to(cls.device)
        cls.model.eval()

    @classmethod
    def tearDownClass(cls):
        # Clean up any files created during tests, except metadata
        if os.path.isfile(cls.test_image_path):
            os.remove(cls.test_image_path)
        if os.path.isfile(cls.test_segmented_image_path):
            os.remove(cls.test_segmented_image_path)

    def test_preprocess_image(self):
        # Test the preprocessing function
        image_tensor = preprocess_image(self.test_image_path, self.device)
        self.assertIsNotNone(image_tensor)
        self.assertEqual(image_tensor.shape[1:], (3, 800, 800), "Image tensor shape mismatch")

    def test_segmentation(self):
        # Ensure metadata file does not exist before running the test
        if os.path.exists(self.test_metadata):
            os.remove(self.test_metadata)

        # Test the extraction and saving of objects
        metadata = extract_and_save_objects(self.test_image_path, self.master_id)
        
        print(f"Metadata returned: {metadata}")
        
        self.assertGreater(len(metadata), 0, "No metadata returned")
        self.assertTrue(os.path.isfile(self.test_segmented_image_path), "Segmented image not saved")

        # Verify metadata file
        self.assertTrue(os.path.exists(self.test_metadata), "Metadata file was not created")
        
        metadata_df = pd.read_csv(self.test_metadata)
        self.assertGreater(len(metadata_df), 0, "Metadata CSV is empty")
        self.assertIn('file_path', metadata_df.columns, "Expected 'file_path' column not found in metadata")

        # Additional debugging statements
        print(f"Metadata DataFrame: \n{metadata_df.head()}")
        print(f"Segmented Image Path: {self.test_segmented_image_path}")

    def test_save_image(self):
        # Test saving of image
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        save_image(dummy_image, self.test_segmented_image_path)
        saved_image = cv2.imread(self.test_segmented_image_path)
        self.assertIsNotNone(saved_image, "Image was not saved correctly")

    def test_process_all_images(self):
        # Ensure metadata file does not exist before running the test
        if os.path.exists(self.test_metadata):
            os.remove(self.test_metadata)

        # Test processing all images
        process_all_images()
        self.assertTrue(os.path.isfile(self.test_metadata), "Metadata file was not created")

        # Verify the metadata content
        metadata_df = pd.read_csv(self.test_metadata)
        self.assertGreater(len(metadata_df), 0, "Metadata CSV is empty")
        self.assertIn('file_path', metadata_df.columns, "Expected 'file_path' column not found in metadata")

if __name__ == '__main__':
    unittest.main()
