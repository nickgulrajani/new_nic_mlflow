import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mlflow
from stock_price_prediction_validated import prepare_data, train_and_log_model

def test_data_preparation():
    """Test data preparation function"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    X, y = prepare_data('MSFT', start_date, end_date)
    
    # Test data structure
    assert isinstance(X, pd.DataFrame), "Features should be a DataFrame"
    assert isinstance(y, pd.Series), "Target should be a Series"
    
    # Test for null values
    assert not X.isnull().any().any(), "Features contain null values"
    assert not y.isnull().any(), "Target contains null values"
    
    # Test data alignment
    assert len(X) == len(y), "Feature and target lengths don't match"
    
    # Test features presence
    required_features = ['Close', 'Return', 'MA5', 'MA20', 'Volatility']
    assert all(col in X.columns for col in required_features), f"Missing required features. Expected {required_features}"
    
    # Test data types
    assert X.dtypes.all() == np.float64, "All features should be float64"
    assert y.dtype == np.float64, "Target should be float64"

def test_data_values():
    """Test data values are within expected ranges"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    X, y = prepare_data('MSFT', start_date, end_date)
    
    # Test Return calculations
    assert abs(X['Return'].mean()) < 1.0, "Average returns seem unrealistic"
    assert X['Return'].std() < 0.1, "Return volatility seems unrealistic"
    
    # Test Moving Averages
    assert (X['MA5'] >= 0).all(), "MA5 contains negative values"
    assert (X['MA20'] >= 0).all(), "MA20 contains negative values"
    
    # Test Volatility
    assert (X['Volatility'] >= 0).all(), "Volatility cannot be negative"
    assert X['Volatility'].mean() < 0.5, "Volatility seems unrealistic"

def test_model_metrics():
    """Test model performance metrics"""
    mlflow.set_experiment("stock_prediction_test")
    
    with mlflow.start_run():
        train_and_log_model()
        
        # Get metrics from current run
        run = mlflow.active_run()
        metrics = mlflow.get_run(run.info.run_id).data.metrics
        
        # Test RMSE
        assert 'rmse' in metrics, "RMSE not logged"
        assert metrics['rmse'] > 0, "RMSE cannot be negative"
        assert metrics['rmse'] < 50.0, "RMSE too high"
        
        # Test R²
        assert 'r2' in metrics, "R² not logged"
        assert metrics['r2'] >= 0, "R² cannot be negative"
        assert metrics['r2'] <= 1.0, "R² cannot be greater than 1"

def test_model_parameters():
    """Test model parameter logging"""
    mlflow.set_experiment("stock_prediction_test")
    
    with mlflow.start_run():
        train_and_log_model()
        
        # Get parameters from current run
        run = mlflow.active_run()
        params = mlflow.get_run(run.info.run_id).data.params
        
        # Test parameters
        assert 'n_estimators' in params, "n_estimators not logged"
        assert 'max_depth' in params, "max_depth not logged"
        assert int(params['n_estimators']) > 0, "Invalid n_estimators"
        assert int(params['max_depth']) > 0, "Invalid max_depth"

if __name__ == "__main__":
    pytest.main([__file__, '-v'])