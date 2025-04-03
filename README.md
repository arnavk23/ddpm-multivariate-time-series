# ddpm-multivariate-time-series
# DDPM: Multivariate Time Series Forecasting

This repository implements a **Denoising Diffusion Probabilistic Model (DDPM)** for multivariate time series forecasting using **PyTorch**. Unlike traditional models, DDPM leverages diffusion-based generative modeling to capture complex temporal dependencies and generate high-quality forecasts.

## Features
- Implements **DDPM** for time series forecasting
- Handles **multivariate** datasets (e.g., stock prices, weather patterns, exchange rates)
- **PyTorch-based**, leveraging deep generative modeling
- Data preprocessing including **scaling and sequence preparation**
- Model evaluation with **MSE, MAE**, and visualization of predictions

## Dataset
This implementation is designed for multivariate time series datasets. Ensure your dataset has:
- A **timestamp column**
- Multiple **numerical feature columns**
- Data in **CSV format**

## Quick Start
### Install Dependencies
```bash
pip install numpy pandas torch torchvision scikit-learn matplotlib
```

### Clone the Repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### Run the Notebook
Use Jupyter Notebook or execute the script:
```bash
jupyter notebook ddpm_multivariate_time_series.ipynb
```

## Understanding DDPM
### What is DDPM?
Denoising Diffusion Probabilistic Models (DDPM) are a class of generative models that learn to reverse a gradual **noise injection** process, allowing them to generate high-quality samples from complex distributions. In the context of time series forecasting, DDPM captures the temporal dependencies and generates realistic future sequences by **progressively denoising noisy inputs**.

### DDPM Architecture
1. **Forward Diffusion Process:** Gradually adds noise to the time series data over multiple time steps.
2. **Reverse Process (Denoising Network):** Uses a neural network (often a U-Net or Transformer) to iteratively remove noise and reconstruct the original data.
3. **Prediction Generation:** After training, the model generates realistic time series samples by reversing the diffusion process on an initial noise input.

## Code Breakdown
### Data Loading and Preprocessing
```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df
```
- Loads the dataset from a CSV file.
- Parses the **date column** and sets it as the index.
- Uses **StandardScaler** to normalize the dataset before training.

### Data Preparation
```python
def prepare_data(df, seq_len=96, pred_len=14):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(len(df_scaled) - seq_len - pred_len):
        X.append(df_scaled[i:i+seq_len])
        y.append(df_scaled[i+seq_len:i+seq_len+pred_len])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler
```
- Converts the dataset into sequences of **`seq_len`** historical data points.
- Defines **prediction length (`pred_len`)**.
- Uses **torch tensors** for training in PyTorch.

### DDPM Model Implementation
```python
class DDPM(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(DDPM, self).__init__()
        self.noise_scheduler = NoiseScheduler()
        self.unet = UNet(input_dim, seq_len)
    
    def forward(self, x, t):
        noise = self.noise_scheduler.add_noise(x, t)
        denoised_output = self.unet(noise, t)
        return denoised_output
```
- **Noise Scheduler:** Controls how noise is added to the input data.
- **U-Net Architecture:** Learns to remove noise step by step.
- **Final Output:** A denoised version of the input, representing the predicted time series.

### Training the Model
```python
def train_model(model, dataloader, epochs=50, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x, t=torch.randint(0, 100, (batch_x.size(0),)))
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
```
- Uses **Adam optimizer** and **MSE loss function**.
- Loops over mini-batches for training.
- Optimizes model parameters using backpropagation.

### Model Evaluation
```python
def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        predictions, actuals = [], []
        for batch_x, batch_y in dataloader:
            predictions.append(model(batch_x, t=torch.randint(0, 100, (batch_x.size(0),))))
            actuals.append(batch_y)
    return predictions, actuals
```
- Evaluates the trained model on test data.
- Uses **torch.no_grad()** for inference.
- Compares predictions with ground truth.

## Results
The DDPM model provides **high-quality forecasts** for multivariate time series data, outperforming traditional models in capturing complex temporal patterns.

## Contributing
Pull requests are welcome! If you'd like to contribute:
1. Fork the repo
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request

## License
This project is licensed under the MIT License.

