import pytest
import pandas as pd
from datetime import datetime, timedelta
import mlflow
from stock_price_prediction_validated import prepare_data

def test_data_preparation():
    # Set test dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Test data preparation
    X, y = prepare_data('MSFT', start_date, end_date)
    
    # Assertions
    assert not X.isnull().any().any(), "Features contain null values"
    assert not y.isnull().any(), "Target contains null values"
    assert len(X) == len(y), "Feature and target lengths don't match"
    assert all(col in X.columns for col in ['Close', 'Return', 'MA5', 'MA20', 'Volatility'])

def test_model_metrics():
    mlflow.set_experiment("stock_prediction_test")
    
    # Get latest run
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=['1'])
    if runs:
        latest_run = runs[0]
        metrics = latest_run.data.metrics
        
        # Test metrics
        assert metrics.get('rmse', float('inf')) < 10.0, "RMSE too high"
        assert metrics.get('r2', 0) > 0.6, "R² too low"