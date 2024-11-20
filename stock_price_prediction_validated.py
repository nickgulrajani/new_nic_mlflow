import mlflow
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

def prepare_data(symbol, start_date, end_date):
    stock = yf.download(symbol, start=start_date, end=end_date)
    df = stock.copy()
    
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Return'].rolling(window=20).std()
    df['Target'] = df['Close'].shift(-1)
    
    df = df.dropna()
    
    features = ['Close', 'Return', 'MA5', 'MA20', 'Volatility']
    return df[features], df['Target']

def train_and_log_model():
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365*2)
    
    symbol = 'MSFT'
    X, y = prepare_data(symbol, start_date, end_date)
    
    with mlflow.start_run():
        model = XGBRegressor(n_estimators=100, max_depth=3)
        model.fit(X, y)
        
        # Log parameters and metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 3)
        
        predictions = model.predict(X)
        rmse = np.sqrt(np.mean((y - predictions) ** 2))
        r2 = 1 - np.sum((y - predictions) ** 2) / np.sum((y - y.mean()) ** 2)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        mlflow.xgboost.log_model(model, "model")

if __name__ == "__main__":
    mlflow.set_experiment("stock_prediction")
    train_and_log_model()