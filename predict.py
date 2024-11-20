import mlflow
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_latest_data(symbol, days=30):
    """Get and prepare latest stock data for prediction."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    stock = yf.download(symbol, start=start_date, end=end_date)
    df = stock.copy()
    
    # Create features matching training data
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Return'].rolling(window=20).std()
    
    features = ['Close', 'Return', 'MA5', 'MA20', 'Volatility']
    return df[features].iloc[-1:], df['Close'].iloc[-1]

def make_prediction(model, data):
    """Make prediction and calculate confidence interval."""
    prediction = model.predict(data)[0]
    
    # Calculate simple confidence interval based on model's feature importances
    importance_sum = sum(model.feature_importances_)
    confidence = importance_sum / len(model.feature_importances_)
    
    return prediction, confidence

def main():
    # Load model from MLflow
    RUN_ID = "YOUR_RUN_ID_HERE"  # Replace with actual run ID
    try:
        model = mlflow.xgboost.load_model(f"runs:/{RUN_ID}/xgboost_model")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Stocks to predict
    symbols = ['MSFT', 'AAPL', 'GOOGL']
    
    for symbol in symbols:
        try:
            # Get latest data
            latest_data, current_price = get_latest_data(symbol)
            
            # Make prediction
            predicted_price, confidence = make_prediction(model, latest_data)
            
            # Calculate metrics
            change = ((predicted_price - current_price) / current_price) * 100
            
            # Print results
            print(f"\n{symbol} Prediction:")
            print(f"Current Price: ${current_price:.2f}")
            print(f"Predicted Next Price: ${predicted_price:.2f}")
            print(f"Predicted Change: {change:.2f}%")
            print(f"Prediction Confidence: {confidence:.2%}")
            
            # Log prediction
            with open('predictions.txt', 'a') as f:
                f.write(f"{datetime.now()}, {symbol}, {current_price:.2f}, {predicted_price:.2f}, {change:.2f}%\n")
                
        except Exception as e:
            print(f"Error predicting {symbol}: {e}")

if __name__ == "__main__":
    main()
