# Stock Price Prediction Dashboard

## Overview
This repository contains a Dash-based application for stock price prediction using various machine learning and deep learning models. The application fetches stock data from Yahoo Finance, performs hyperparameter tuning, and visualizes predictions and model performance.

The models integrated into the application include:
- **Deep Learning Models**: LSTM, GRU, Simple RNN, and Dense Neural Networks.
- **Machine Learning Models**: Decision Tree, Random Forest, and Linear Regression.

The user-friendly interface allows analysts and traders to input stock tickers, select metrics (Open, High, Low, Close), and choose predictive models to analyze historical trends and make future predictions.

## Features
### 1. Stock Data Retrieval
- Fetches historical stock data from Yahoo Finance for the past year.
- Users can specify stock tickers and select metrics (Close, Open, High, or Low).

### 2. Model Selection
- Offers a variety of predictive models:
  - LSTM: Long Short-Term Memory Networks
  - GRU: Gated Recurrent Units
  - Simple RNN: Basic Recurrent Neural Networks
  - Dense: Fully Connected Neural Network
  - Decision Tree
  - Random Forest
  - Linear Regression

### 3. Hyperparameter Tuning
- Performs grid search to optimize model parameters (e.g., units, activation functions, batch sizes for deep learning; max depth for Decision Tree; n_estimators for Random Forest).
- Displays RMSE (Root Mean Square Error) for different hyperparameter configurations.

### 4. Visualization
- **Historical Data Graph**: Visualizes past stock performance.
- **Prediction Graph**: Shows future predictions over the next 90 days.
- **Hyperparameter Performance Graph**: Compares RMSE values across different parameter settings.

## How to Run the Application
### Prerequisites
- Python 3.8+
- Required Python Libraries:
  - `numpy`
  - `pandas`
  - `yfinance`
  - `dash`
  - `plotly`
  - `keras`
  - `scikit-learn`

### Steps
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://127.0.0.1:8050/
   ```

## Application Structure
- **Input Fields**: Allows users to input stock tickers, select metrics, and choose predictive models.
- **Graphs**:
  - Historical data trends.
  - Predictions for the next 90 days.
  - Hyperparameter performance for selected models.

## Code Walkthrough
1. **Data Preprocessing**:
   - The `create_timeseries_data` function generates time-series data for training.
   - `MinMaxScaler` is used to normalize stock price values for improved model performance.

2. **Model Training**:
   - Neural networks (LSTM, GRU, RNN, Dense) are trained on sequential data.
   - Non-neural models (Decision Tree, Random Forest, Linear Regression) use flattened data for compatibility.
   - Hyperparameter tuning is applied to select the best model configuration based on RMSE.

3. **Visualization**:
   - The Dash app integrates Plotly for interactive visualizations.
   - Separate graphs for historical data, predictions, and hyperparameter performance.

4. **User Interaction**:
   - Dynamic callbacks update graphs based on user inputs.

## Use Cases
### For Analysts
- **Historical Data Analysis**: Quickly visualize stock trends to identify patterns.
- **Performance Comparison**: Evaluate multiple models to determine the most accurate prediction method.

### For Stock Traders
- **Short-Term Predictions**: Use 90-day forecasts to make informed trading decisions.
- **Model Customization**: Experiment with different algorithms to tailor predictions for specific stocks.

## Possible Extensions
- **Sentiment Analysis**: Integrate news and social media sentiment data for enhanced predictions.
- **Portfolio Management**: Add functionality for tracking and predicting multiple stocks.
- **Multi-Metric Predictions**: Combine metrics like volume, volatility, and price for comprehensive analysis.

## Screenshots
- **Historical Data Graph**: Shows the stock's past performance.
- **Prediction Graph**: Displays model-generated predictions for the next 90 days.
- **Hyperparameter Performance Graph**: Highlights the effect of different parameters on model accuracy.

## Acknowledgments
- **Yahoo Finance**: For providing stock market data.
- **Dash and Plotly**: For creating an interactive dashboard environment.
- **Keras and scikit-learn**: For powerful machine learning and deep learning tools.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

