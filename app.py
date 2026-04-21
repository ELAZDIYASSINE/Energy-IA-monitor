"""
Energy AI Monitor - Streamlit Dashboard
Interactive web application for energy consumption analysis,
anomaly detection, and prediction.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import preprocess_pipeline, calculate_metrics
from model import AnomalyDetector, ConsumptionPredictor, EnsembleAnomalyDetector


def calculate_dynamic_contamination(dataset_size):
    """
    Calculate optimal contamination based on dataset size.
    Uses a formula that scales appropriately for different dataset sizes.
    
    Args:
        dataset_size: Number of data points in the dataset
        
    Returns:
        Optimal contamination value between 0.01 and 0.10
    """
    # Formula: scales with dataset size, bounded between 1% and 10%
    # Small datasets (<50): ~3-5%
    # Medium datasets (50-200): ~2-3%
    # Large datasets (>200): ~1-2%
    if dataset_size < 50:
        contamination = 0.05
    elif dataset_size < 100:
        contamination = 0.03
    elif dataset_size < 200:
        contamination = 0.02
    else:
        contamination = 0.015
    
    # Ensure bounds
    contamination = max(0.01, min(0.10, contamination))
    
    return contamination


# Page configuration
st.set_page_config(
    page_title="Energy AI Monitor",
    page_icon="⚡",
    layout="wide"
)


def main():
    """
    Main application function.
    """
    # Title and header
    st.title("⚡ Energy AI Monitor")
    st.markdown("---")
    st.markdown("### Analyze energy consumption, detect anomalies, and predict future usage")
    
    # File uploader
    st.subheader("📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with columns: timestamp, consumption, temperature",
        type=['csv']
    )
    
    if uploaded_file is not None:
        # Load and preprocess data
        df = pd.read_csv(uploaded_file)
        df_processed = preprocess_pipeline(df)
        
        # Display dataset
        st.subheader("📊 Dataset Overview")
        st.dataframe(df_processed.head(10))
        
        # Calculate and display metrics
        metrics = calculate_metrics(df_processed)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Consumption", f"{metrics['mean_consumption']:.2f}")
        with col2:
            st.metric("Max Consumption", f"{metrics['max_consumption']:.2f}")
        with col3:
            st.metric("Min Consumption", f"{metrics['min_consumption']:.2f}")
        with col4:
            st.metric("Total Records", metrics['total_records'])
        
        st.markdown("---")
        
        # Calculate dynamic contamination based on dataset size
        contamination = calculate_dynamic_contamination(len(df_processed))
        
        # Show calculated contamination
        st.info(f"🎯 Auto-calculated contamination: {contamination*100:.1f}% (based on {len(df_processed)} data points)")
        
        # Check if saved models exist
        saved_anomaly_detector = AnomalyDetector.load()
        saved_predictor = ConsumptionPredictor.load()
        
        # Show retrain option if models exist
        if saved_anomaly_detector is not None and saved_predictor is not None:
            st.info("💾 Saved models found from previous training")
            retrain = st.checkbox("🔄 Retrain models on new data (uncheck to use existing models)", value=True)
        else:
            retrain = True
        
        # Anomaly Detection
        st.subheader("🔍 Anomaly Detection")
        
        # Algorithm selection
        algorithm = st.radio(
            "Select Anomaly Detection Algorithm:",
            ["Isolation Forest (Single)", "Ensemble (3 Algorithms - Recommended)"],
            help="Ensemble combines Isolation Forest, LOF, and SVM for better accuracy"
        )
        
        if retrain or saved_anomaly_detector is None:
            if algorithm == "Ensemble (3 Algorithms - Recommended)":
                anomaly_detector = EnsembleAnomalyDetector(contamination=contamination)
                anomaly_detector.fit(df_processed)
                st.info("✅ Ensemble detector trained (Isolation Forest + LOF + SVM)")
            else:
                anomaly_detector = AnomalyDetector(contamination=contamination)
                anomaly_detector.fit(df_processed)
                st.info("✅ Anomaly detector trained on new data")
        else:
            anomaly_detector = saved_anomaly_detector
            st.info("📦 Using previously trained anomaly detector")
        
        df_with_anomalies = anomaly_detector.predict(df_processed)
        
        anomaly_count = anomaly_detector.get_anomaly_count(df_with_anomalies)
        
        # Display anomaly count
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Detected Anomalies", anomaly_count)
        with col2:
            if anomaly_count > 0:
                st.warning(f"⚠️ {anomaly_count} anomalies detected in the dataset!")
            else:
                st.success("✅ No anomalies detected")
        
        # Line chart with anomalies highlighted
        st.subheader("📈 Consumption Over Time")
        
        # Create figure
        fig = go.Figure()
        
        # Add normal consumption line
        normal_data = df_with_anomalies[df_with_anomalies['anomaly'] == 0]
        fig.add_trace(go.Scatter(
            x=normal_data['timestamp'],
            y=normal_data['consumption'],
            mode='lines+markers',
            name='Normal',
            line=dict(color='blue'),
            marker=dict(size=4)
        ))
        
        # Add anomalies in red
        anomaly_data = df_with_anomalies[df_with_anomalies['anomaly'] == 1]
        if len(anomaly_data) > 0:
            fig.add_trace(go.Scatter(
                x=anomaly_data['timestamp'],
                y=anomaly_data['consumption'],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=10, symbol='x')
            ))
        
        fig.update_layout(
            title="Energy Consumption with Anomaly Detection",
            xaxis_title="Timestamp",
            yaxis_title="Consumption",
            hovermode='x unified',
            legend=dict(x=0, y=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detected anomalies table
        if anomaly_count > 0:
            st.subheader("🚨 Detected Anomalies")
            anomalies_df = df_with_anomalies[df_with_anomalies['anomaly'] == 1][
                ['timestamp', 'consumption', 'temperature']
            ].copy()
            anomalies_df = anomalies_df.sort_values('timestamp', ascending=False)
            st.dataframe(anomalies_df)
        
        st.markdown("---")
        
        # Prediction Model
        st.subheader("🔮 Consumption Prediction")
        
        # Train prediction model
        if retrain or saved_predictor is None:
            predictor = ConsumptionPredictor()
            predictor.train(df_processed)
            predictor.save()
            st.info("✅ Prediction model trained on new data and saved")
        else:
            predictor = saved_predictor
            st.info("📦 Using previously trained prediction model")
        
        # Display model metrics
        model_metrics = predictor.get_metrics()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model R² Score", f"{model_metrics['r2']:.4f}")
        with col2:
            st.metric("Model MSE", f"{model_metrics['mse']:.4f}")
        
        # Prediction interface
        st.markdown("#### Predict consumption for specific conditions:")
        
        col1, col2 = st.columns(2)
        with col1:
            temperature_input = st.number_input(
                "Temperature (°C)",
                min_value=-50.0,
                max_value=50.0,
                value=20.0,
                step=0.5
            )
        with col2:
            hour_input = st.slider(
                "Hour of day",
                min_value=0,
                max_value=23,
                value=12
            )
        
        # Make prediction
        if st.button("Predict Consumption"):
            prediction = predictor.predict(temperature_input, hour_input)
            st.success(f"📊 Predicted Consumption: **{prediction:.2f}**")
        
    else:
        # Display instructions when no file is uploaded
        st.info("👆 Please upload a CSV file to begin analysis")
        
        # Show example format
        st.markdown("### Expected CSV Format:")
        st.markdown("""
        | timestamp | consumption | temperature |
        |-----------|-------------|-------------|
        | 2024-01-01 00:00:00 | 150.5 | 18.2 |
        | 2024-01-01 01:00:00 | 142.3 | 17.8 |
        | 2024-01-01 02:00:00 | 138.1 | 17.5 |
        """)


if __name__ == "__main__":
    main()
