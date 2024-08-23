import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
import pandas as pd
from models.summarization_model import load_csv_files, preprocess_dataframes, generate_summaries, save_summaries

class TestSummarization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the test environment and directories."""
        cls.identification_file = 'data/test_identification.csv'
        cls.text_extraction_file = 'data/test_text_extraction.csv'
        cls.summaries_file = 'data/test_summaries.csv'
        cls.summaries_json_file = 'data/test_summaries.json'
        if not os.path.exists('data'):
            os.makedirs('data')
        # Create dummy files for testing
        pd.DataFrame({'id': [1], 'description': ['test_description']}).to_csv(cls.identification_file, index=False)
        pd.DataFrame({'id': [1], 'text': ['test_text']}).to_csv(cls.text_extraction_file, index=False)
        with open(cls.summaries_file, 'w') as f:
            f.write('id,summary\n1,test_summary\n')
        with open(cls.summaries_json_file, 'w') as f:
            f.write('{"1": {"summary": "test_summary"}}\n')

    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        if os.path.exists(cls.identification_file):
            os.remove(cls.identification_file)
        if os.path.exists(cls.text_extraction_file):
            os.remove(cls.text_extraction_file)
        if os.path.exists(cls.summaries_file):
            os.remove(cls.summaries_file)
        if os.path.exists(cls.summaries_json_file):
            os.remove(cls.summaries_json_file)

    @patch('models.summarization_model.load_csv_files')
    def test_load_csv_files(self, mock_load_csv_files):
        """Test the load_csv_files function."""
        mock_load_csv_files.return_value = (pd.DataFrame({'id': [1], 'description': ['test_description']}), pd.DataFrame({'id': [1], 'text': ['test_text']}))

        try:
            identification_df, text_extraction_df = load_csv_files()
            mock_load_csv_files.assert_called_once()
            self.assertEqual(len(identification_df), 1)
            self.assertEqual(len(text_extraction_df), 1)
        except Exception as e:
            self.fail(f"load_csv_files raised an exception: {e}")

    @patch('models.summarization_model.preprocess_dataframes')
    def test_preprocess_dataframes(self, mock_preprocess_dataframes):
        """Test the preprocess_dataframes function."""
        mock_preprocess_dataframes.return_value = pd.DataFrame({'id': [1], 'summary': ['test_summary']})

        try:
            df = mock_preprocess_dataframes(pd.DataFrame({'id': [1], 'description': ['test_description']}), pd.DataFrame({'id': [1], 'text': ['test_text']}))
            mock_preprocess_dataframes.assert_called_once()
            self.assertEqual(len(df), 1)
        except Exception as e:
            self.fail(f"preprocess_dataframes raised an exception: {e}")

    @patch('models.summarization_model.generate_summaries')
    def test_generate_summaries(self, mock_generate_summaries):
        """Test the generate_summaries function."""
        mock_generate_summaries.return_value = pd.DataFrame({'id': [1], 'summary': ['test_summary']})

        try:
            df = mock_generate_summaries(pd.DataFrame({'id': [1], 'description': ['test_description']}), pd.DataFrame({'id': [1], 'text': ['test_text']}))
            mock_generate_summaries.assert_called_once()
            self.assertEqual(len(df), 1)
        except Exception as e:
            self.fail(f"generate_summaries raised an exception: {e}")

    @patch('models.summarization_model.save_summaries')
    def test_save_summaries(self, mock_save_summaries):
        """Test the save_summaries function."""
        mock_save_summaries.return_value = None

        try:
            save_summaries(pd.DataFrame({'id': [1], 'summary': ['test_summary']}))
            mock_save_summaries.assert_called_once_with(pd.DataFrame({'id': [1], 'summary': ['test_summary']}))
        except Exception as e:
            self.fail(f"save_summaries raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
