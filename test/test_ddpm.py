import unittest
from src.models.ddpm import DDPM

class TestDDPM(unittest.TestCase):

    def setUp(self):
        self.ddpm = DDPM()

    def test_train(self):
        # Assuming train method returns a boolean indicating success
        result = self.ddpm.train()
        self.assertTrue(result)

    def test_generate(self):
        # Assuming generate method returns generated samples
        samples = self.ddpm.generate()
        self.assertIsNotNone(samples)
        self.assertGreater(len(samples), 0)

    def test_save_model(self):
        # Assuming save_model method returns a boolean indicating success
        result = self.ddpm.save_model("test_model.pth")
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
