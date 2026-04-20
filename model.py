"""
Machine Learning models for Energy AI Monitor.
Handles anomaly detection and consumption prediction.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class AnomalyDetector:
    """
    Anomaly detection using Isolation Forest algorithm.
    Detects unusual energy consumption patterns.
    """
    
    def __init__(self, contamination=0.05, random_state=42):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers (default: 0.05)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )
        
    def fit(self, df):
        """
        Train the anomaly detection model.
        
        Args:
            df: DataFrame with 'consumption' and 'temperature' columns
        """
        features = df[['consumption', 'temperature']]
        self.model.fit(features)
        
    def predict(self, df):
        """
        Predict anomalies in the dataset.
        
        Args:
            df: DataFrame with 'consumption' and 'temperature' columns
            
        Returns:
            DataFrame with added 'anomaly' column (1=anomaly, 0=normal)
        """
        features = df[['consumption', 'temperature']]
        predictions = self.model.predict(features)
        
        # Convert predictions: 1 for anomaly, 0 for normal
        # IsolationForest returns -1 for anomalies, 1 for normal
        df['anomaly'] = np.where(predictions == -1, 1, 0)
        
        return df
    
    def get_anomaly_count(self, df):
        """
        Count the number of anomalies detected.
        
        Args:
            df: DataFrame with 'anomaly' column
            
        Returns:
            Integer count of anomalies
        """
        return df['anomaly'].sum()
    
    def save(self, filepath='anomaly_detector.pkl'):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath='anomaly_detector.pkl'):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded AnomalyDetector instance
        """
        if os.path.exists(filepath):
            return joblib.load(filepath)
        return None


class LOFDetector:
    """
    Anomaly detection using Local Outlier Factor (LOF) algorithm.
    Detects anomalies based on local density deviation.
    Good for detecting local anomalies in clusters.
    """
    
    def __init__(self, n_neighbors=20, contamination=0.05):
        """
        Initialize the LOF detector.
        
        Args:
            n_neighbors: Number of neighbors to use (default: 20)
            contamination: Expected proportion of outliers (default: 0.05)
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True
        )
        
    def fit(self, df):
        """
        Train the LOF detector.
        
        Args:
            df: DataFrame with 'consumption' and 'temperature' columns
        """
        features = df[['consumption', 'temperature']]
        self.model.fit(features)
        
    def predict(self, df):
        """
        Predict anomalies in the dataset.
        
        Args:
            df: DataFrame with 'consumption' and 'temperature' columns
            
        Returns:
            DataFrame with added 'anomaly' column (1=anomaly, 0=normal)
        """
        features = df[['consumption', 'temperature']]
        predictions = self.model.predict(features)
        
        # Convert predictions: 1 for anomaly, 0 for normal
        # LOF returns -1 for anomalies, 1 for normal
        df['anomaly'] = np.where(predictions == -1, 1, 0)
        
        return df
    
    def get_anomaly_count(self, df):
        """
        Count the number of anomalies detected.
        
        Args:
            df: DataFrame with 'anomaly' column
            
        Returns:
            Integer count of anomalies
        """
        return df['anomaly'].sum()


class SVMDetector:
    """
    Anomaly detection using One-Class SVM algorithm.
    Learns a decision boundary around normal data.
    Good for detecting global anomalies.
    """
    
    def __init__(self, nu=0.05, kernel='rbf'):
        """
        Initialize the One-Class SVM detector.
        
        Args:
            nu: Expected proportion of outliers (default: 0.05)
            kernel: Kernel type (default: 'rbf')
        """
        self.nu = nu
        self.kernel = kernel
        self.model = OneClassSVM(
            nu=nu,
            kernel=kernel
        )
        
    def fit(self, df):
        """
        Train the One-Class SVM detector.
        
        Args:
            df: DataFrame with 'consumption' and 'temperature' columns
        """
        features = df[['consumption', 'temperature']]
        self.model.fit(features)
        
    def predict(self, df):
        """
        Predict anomalies in the dataset.
        
        Args:
            df: DataFrame with 'consumption' and 'temperature' columns
            
        Returns:
            DataFrame with added 'anomaly' column (1=anomaly, 0=normal)
        """
        features = df[['consumption', 'temperature']]
        predictions = self.model.predict(features)
        
        # Convert predictions: 1 for anomaly, 0 for normal
        # OneClassSVM returns -1 for anomalies, 1 for normal
        df['anomaly'] = np.where(predictions == -1, 1, 0)
        
        return df
    
    def get_anomaly_count(self, df):
        """
        Count the number of anomalies detected.
        
        Args:
            df: DataFrame with 'anomaly' column
            
        Returns:
            Integer count of anomalies
        """
        return df['anomaly'].sum()


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detection combining multiple algorithms.
    Only flags anomalies when most algorithms agree.
    Significantly reduces false positives.
    """
    
    def __init__(self, contamination=0.05):
        """
        Initialize the ensemble detector.
        
        Args:
            contamination: Expected proportion of outliers (default: 0.05)
        """
        self.contamination = contamination
        self.detectors = [
            AnomalyDetector(contamination=contamination),
            LOFDetector(contamination=contamination),
            SVMDetector(nu=contamination)
        ]
        
    def fit(self, df):
        """
        Train all detectors in the ensemble.
        
        Args:
            df: DataFrame with 'consumption' and 'temperature' columns
        """
        for detector in self.detectors:
            detector.fit(df)
        
    def predict(self, df):
        """
        Predict anomalies using ensemble voting.
        Only flags as anomaly if majority of detectors agree.
        
        Args:
            df: DataFrame with 'consumption' and 'temperature' columns
            
        Returns:
            DataFrame with added 'anomaly' column (1=anomaly, 0=normal)
        """
        # Get predictions from all detectors
        all_predictions = []
        for detector in self.detectors:
            df_copy = df.copy()
            df_copy = detector.predict(df_copy)
            all_predictions.append(df_copy['anomaly'].values)
        
        # Unanimous voting: flag as anomaly only if all 3 algorithms agree
        predictions_array = np.array(all_predictions)
        ensemble_prediction = np.where(predictions_array.sum(axis=0) >= 3, 1, 0)
        
        df['anomaly'] = ensemble_prediction
        
        return df
    
    def get_anomaly_count(self, df):
        """
        Count the number of anomalies detected.
        
        Args:
            df: DataFrame with 'anomaly' column
            
        Returns:
            Integer count of anomalies
        """
        return df['anomaly'].sum()


class ConsumptionPredictor:
    """
    Energy consumption prediction using Random Forest Regressor.
    Predicts consumption based on temperature and time features.
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the prediction model.
        
        Args:
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.metrics = {}
        
    def train(self, df):
        """
        Train the prediction model.
        
        Args:
            df: DataFrame with 'temperature', 'hour', 'consumption' columns
        """
        features = df[['temperature', 'hour']]
        target = df['consumption']
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=self.random_state
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        self.metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
    def predict(self, temperature, hour):
        """
        Predict energy consumption for given conditions.
        
        Args:
            temperature: Temperature value
            hour: Hour of the day (0-23)
            
        Returns:
            Predicted consumption value
        """
        # Create feature array
        features = np.array([[temperature, hour]])
        prediction = self.model.predict(features)[0]
        
        return prediction
    
    def get_metrics(self):
        """
        Get model performance metrics.
        
        Returns:
            Dictionary with MSE and R2 scores
        """
        return self.metrics
    
    def save(self, filepath='consumption_predictor.pkl'):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath='consumption_predictor.pkl'):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded ConsumptionPredictor instance
        """
        if os.path.exists(filepath):
            return joblib.load(filepath)
        return None
