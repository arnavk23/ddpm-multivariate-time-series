import unittest
from src.training.train import train_model
from src.models.ddpm import DDPM
from src.data.data_loader import DataLoader

class TestTrainModel(unittest.TestCase):

    def setUp(self):
        self.data_loader = DataLoader()
        self.model = DDPM()

    def test_train_model(self):
        # Load and preprocess data
        data = self.data_loader.load_data('path/to/dataset')
        preprocessed_data = self.data_loader.preprocess_data(data)

        # Train the model
        result = train_model(self.model, preprocessed_data)

        # Check if the model has been trained successfully
        self.assertIsNotNone(result)
        self.assertTrue(result['loss'] < 1.0)  # Example condition for a successful training

if __name__ == '__main__':
    unittest.main()