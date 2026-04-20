# Energy AI Monitor ⚡

A complete end-to-end web application for energy consumption analysis, anomaly detection using Machine Learning, and consumption prediction.

## Features

- **Data Upload**: Upload CSV files with energy consumption data
- **Data Processing**: Automatic cleaning, timestamp conversion, and feature engineering
- **Anomaly Detection**: Uses Isolation Forest to detect unusual consumption patterns
- **Prediction Model**: Random Forest Regressor predicts future consumption based on temperature and time
- **Interactive Dashboard**: Clean Streamlit UI with charts, metrics, and predictions

## Tech Stack

- Python
- pandas
- scikit-learn
- Streamlit
- Plotly

## Project Structure

```
energy-ai/
├── app.py              # Streamlit dashboard
├── model.py            # ML models (AnomalyDetector, ConsumptionPredictor)
├── utils.py            # Data processing utilities
├── requirements.txt    # Python dependencies
├── example_data.csv    # Sample dataset
└── README.md          # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## CSV Format

Upload CSV files with the following columns:

| timestamp | consumption | temperature |
|-----------|-------------|-------------|
| 2024-01-01 00:00:00 | 150.5 | 18.2 |
| 2024-01-01 01:00:00 | 142.3 | 17.8 |

- **timestamp**: Date and time in YYYY-MM-DD HH:MM:SS format
- **consumption**: Energy consumption value (numeric)
- **temperature**: Temperature in Celsius (numeric)

## Example Data

An `example_data.csv` file is included with 8 days of hourly data, including some anomalies for testing.

## ML Models

### Anomaly Detection
- Algorithm: Isolation Forest
- Contamination: 0.05 (5% expected outliers)
- Features: consumption, temperature

### Prediction Model
- Algorithm: Random Forest Regressor
- Features: temperature, hour of day
- Target: consumption
- Metrics: R² score, MSE displayed in dashboard

## Dashboard Features

- **Dataset Overview**: View uploaded data with key metrics
- **Anomaly Detection**: Visual chart with anomalies highlighted in red
- **Detected Anomalies Table**: Detailed list of anomalous readings
- **Prediction Interface**: Input temperature and hour to get predicted consumption
- **Model Performance**: R² and MSE metrics displayed

## Notes

- The application is designed for demonstration and educational purposes
- No deep learning models are used - focuses on practical, business-oriented ML
- Clean, modular code structure suitable for student projects or job interviews
