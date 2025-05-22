import unittest
from src.data.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.data_loader = DataLoader()

    def test_load_data(self):
        # Assuming load_data returns a DataFrame or similar structure
        data = self.data_loader.load_data('path/to/data.csv')
        self.assertIsNotNone(data)
        self.assertTrue(hasattr(data, 'shape'))  # Check if it's a DataFrame

    def test_preprocess_data(self):
        raw_data = self.data_loader.load_data('path/to/data.csv')
        processed_data = self.data_loader.preprocess_data(raw_data)
        self.assertIsNotNone(processed_data)
        self.assertNotEqual(len(processed_data), 0)  # Ensure data is not empty

if __name__ == '__main__':
    unittest.main()