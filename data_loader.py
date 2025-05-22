class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        import pandas as pd
        data = pd.read_csv(self.file_path)
        return data

    def preprocess_data(self, data):
        # Example preprocessing steps
        data.fillna(method='ffill', inplace=True)
        data = (data - data.mean()) / data.std()
        return data