# Stock Price Prediction

This repository contains a Python script that uses a Random Forest Regressor to predict the stock prices for the next week and visualizes the most recent 30 days of historical stock prices along with the predictions.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Contributing](#contributing)
- [License](#license)

## Overview

The script reads historical stock prices from a CSV file, trains a Random Forest model, and predicts stock prices for the next 7 days. It also visualizes the most recent 30 days of historical prices and the predicted prices on a graph.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/stock-price-prediction.git
    cd stock-price-prediction
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have a CSV file named `aapl.csv` in the same directory with the following structure:
    ```plaintext
    Date,Close/Last
    12/24/2024,$258.20
    12/23/2024,$255.27
    ...
    ```

## Usage

Run the script to train the model and visualize the stock price predictions:
```bash
python stock_predictor.py


The script will:

Train a Random Forest model using the historical data.

Predict stock prices for the next 7 days.

Plot the most recent 30 days of historical prices along with the predictions.

Files
stock_predictor.py: Main script for reading data, training the model, predicting prices, and visualizing the results.

aapl.csv: Example CSV file containing historical stock prices.

Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or suggestions.

License
This project is licensed under the MIT License - see the LICENSE file for details.

