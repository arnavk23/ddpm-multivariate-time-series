# DDPM Multivariate Time Series

This project implements a Denoising Diffusion Probabilistic Model (DDPM) for multivariate time series data. The goal is to provide a framework for training, evaluating, and generating synthetic time series data using diffusion models.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [License](#license)

## Overview

Denoising Diffusion Probabilistic Models have shown great promise in generating high-quality samples from complex distributions. This project focuses on applying DDPMs to multivariate time series data, allowing for the generation of realistic synthetic time series that can be used for various applications, including forecasting and anomaly detection.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/arnavk23/ddpm-multivariate-time-series.git
cd ddpm-multivariate-time-series
pip install -r requirements.txt
```

## Usage

1. **Loading Data**: Use the `DataLoader` class from `src/data/data_loader.py` to load and preprocess your time series data.
2. **Training the Model**: Call the `train_model` function from `src/training/train.py` to start training the DDPM on your dataset.
3. **Evaluating the Model**: Use the `evaluate_model` function from `src/evaluation/evaluate.py` to assess the performance of the trained model.
4. **Generating Samples**: Utilize the `generate` method from the `DDPM` class in `src/models/ddpm.py` to create new synthetic time series data.

## Directory Structure

```
ddpm-multivariate-time-series
├── src
│   ├── data
│   ├── models
│   ├── training
│   ├── evaluation
│   └── utils
├── tests
├── requirements.txt
├── setup.py
├── .gitignore
├── README.md
└── LICENSE
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.