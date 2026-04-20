"""
Data processing utilities for Energy AI Monitor.
Handles data cleaning, feature engineering, and basic statistics.
"""

import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Load energy consumption data from CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        pandas DataFrame with energy data
    """
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    """
    Clean the dataset by handling missing values.
    
    Args:
        df: Raw pandas DataFrame
        
    Returns:
        Cleaned pandas DataFrame
    """
    # Drop rows with missing values
    df_clean = df.dropna()
    
    # Reset index after dropping rows
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean


def convert_timestamp(df):
    """
    Convert timestamp column to datetime format.
    
    Args:
        df: pandas DataFrame with 'timestamp' column
        
    Returns:
        DataFrame with datetime timestamp
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def create_features(df):
    """
    Create additional time-based features from timestamp.
    
    Args:
        df: pandas DataFrame with datetime 'timestamp' column
        
    Returns:
        DataFrame with additional features: hour, day
    """
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    
    return df


def calculate_metrics(df):
    """
    Calculate basic statistics for the dataset.
    
    Args:
        df: pandas DataFrame with 'consumption' column
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'mean_consumption': df['consumption'].mean(),
        'max_consumption': df['consumption'].max(),
        'min_consumption': df['consumption'].min(),
        'std_consumption': df['consumption'].std(),
        'total_records': len(df)
    }
    
    return metrics


def preprocess_pipeline(df):
    """
    Complete preprocessing pipeline: clean, convert, and create features.
    
    Args:
        df: Raw pandas DataFrame
        
    Returns:
        Fully processed DataFrame
    """
    # Clean missing values
    df = clean_data(df)
    
    # Convert timestamp
    df = convert_timestamp(df)
    
    # Create features
    df = create_features(df)
    
    return df
