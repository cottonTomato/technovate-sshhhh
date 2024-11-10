import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from typing import Dict, List, Union, Tuple, Optional
import joblib
import base64
from io import BytesIO
from catboost import CatBoostRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesPreprocessor:
    """Handles time series data preprocessing and feature engineering"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.scaler = StandardScaler()
        
    def create_temporal_features(self) -> pd.DataFrame:
        """Create temporal features from datetime index"""
        df = self.data.copy()
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        return df
        
    def create_lag_features(self, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """Create lagged features for specified columns"""
        df = self.data.copy()
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df
        
    def create_rolling_features(self, columns: List[str], windows: List[int]) -> pd.DataFrame:
        """Create rolling statistics features"""
        df = self.data.copy()
        for col in columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
        return df

    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess data by creating temporal, lag, and rolling features"""
        df = self.create_temporal_features()
        df = self.create_lag_features(columns=['total_power_consumption'], lags=[1, 2, 3])
        df = self.create_rolling_features(columns=['total_power_consumption'], windows=[3, 7])
        df.dropna(inplace=True)
        return df

class ConsumptionPredictor:
    """Handles time series prediction using various models"""
    
    def __init__(self):
        self.prophet_model = None
        self.lstm_model = None
        self.rf_model = None
        self.catboost_model = None
        
    def train_prophet(self, data: pd.DataFrame, target_col: str) -> Dict:
        """Train Facebook Prophet model for time series prediction"""
        try:
            df = data.reset_index()
            df = df.rename(columns={'datetime': 'ds', target_col: 'y'})
            
            self.prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True
            )
            self.prophet_model.fit(df)
            
            # Generate forecast
            future = self.prophet_model.make_future_dataframe(periods=30)
            forecast = self.prophet_model.predict(future)
            
            return {
                'status': 'success',
                'forecast': forecast.to_dict('records'),
                'components': self.prophet_model.plot_components(forecast).to_dict()
            }
        except Exception as e:
            logger.error(f"Prophet training error: {str(e)}")
            return {'status': 'error', 'message': str(e)}
            
    def train_lstm(self, data: pd.DataFrame, target_col: str, lookback: int = 24) -> Dict:
        """Train LSTM model for time series prediction"""
        try:
            # Prepare sequences
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data[[target_col]])
            
            X, y = [], []
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i])
                y.append(scaled_data[i])
            X, y = np.array(X), np.array(y)
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build model
            self.lstm_model = Sequential([
                LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
                Dropout(0.2),
                LSTM(50, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            self.lstm_model.compile(optimizer='adam', loss='mse')
            
            # Train model
            history = self.lstm_model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.1,
                verbose=0
            )
            
            # Evaluate
            y_pred = self.lstm_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            return {
                'status': 'success',
                'mse': mse,
                'history': history.history,
                'scaler': scaler
            }
        except Exception as e:
            logger.error(f"LSTM training error: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def train_arima(self, data: pd.DataFrame, target_col: str) -> Dict:
        """Train ARIMA model for time series prediction"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            model = ARIMA(data[target_col], order=(5, 1, 0))
            model_fit = model.fit()
            
            forecast = model_fit.forecast(steps=30)
            
            return {
                'status': 'success',
                'forecast': forecast.tolist()
            }
        except Exception as e:
            logger.error(f"ARIMA training error: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def train_catboost(self, data: pd.DataFrame, target_col: str) -> Dict:
        """Train CatBoost model for time series prediction"""
        try:
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.catboost_model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, verbose=0)
            self.catboost_model.fit(X_train, y_train)
            
            y_pred = self.catboost_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            return {
                'status': 'success',
                'mse': mse,
                'feature_importances': self.catboost_model.get_feature_importance().tolist()
            }
        except Exception as e:
            logger.error(f"CatBoost training error: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def cluster_consumption_patterns(self, data: pd.DataFrame, n_clusters: int = 3) -> Dict:
        """Cluster consumption patterns using KMeans"""
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(data)
            
            return {
                'status': 'success',
                'clusters': clusters.tolist()
            }
        except Exception as e:
            logger.error(f"Clustering error: {str(e)}")
            return {'status': 'error', 'message': str(e)}

class AnomalyDetector:
    """Handles anomaly detection in consumption patterns"""
    
    def __init__(self):
        self.model = None
        
    def detect_anomalies(self, data: pd.DataFrame, contamination: float = 0.1) -> Dict:
        """Detect anomalies using Isolation Forest"""
        try:
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42
            )
            
            # Fit and predict
            predictions = self.model.fit_predict(data)
            anomaly_mask = predictions == -1
            
            return {
                'status': 'success',
                'anomalies': {
                    'indices': data.index[anomaly_mask].tolist(),
                    'values': data[anomaly_mask].to_dict('records')
                }
            }
        except Exception as e:
            logger.error(f"Anomaly detection error: {str(e)}")
            return {'status': 'error', 'message': str(e)}

class EcoScoreCalculator:
    """Calculates Eco Score and provides optimization suggestions"""
    
    def __init__(self):
        self.model = None
        
    def calculate_eco_score(self, data: pd.DataFrame, benchmarks: Dict[str, float]) -> Dict:
        """Calculate Eco Score based on benchmarks"""
        try:
            scores = {}
            for col in data.columns:
                if col in benchmarks:
                    scores[col] = 100 - ((data[col].mean() / benchmarks[col]) * 100)
                else:
                    logger.warning(f"Benchmark for {col} not provided.")
            
            return {
                'status': 'success',
                'eco_scores': scores
            }
        except Exception as e:
            logger.error(f"Eco Score calculation error: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def provide_optimization_suggestions(self, data: pd.DataFrame) -> Dict:
        """Provide optimization suggestions based on usage patterns"""
        try:
            suggestions = []
            if data['consumption'].mean() > data['consumption'].quantile(0.75):
                suggestions.append("Consider reducing peak hour usage.")
            if data['consumption'].std() > data['consumption'].mean() * 0.5:
                suggestions.append("Usage is highly variable, consider more consistent usage patterns.")
            
            return {
                'status': 'success',
                'suggestions': suggestions
            }
        except Exception as e:
            logger.error(f"Optimization suggestions error: {str(e)}")
            return {'status': 'error', 'message': str(e)}

class BehavioralSegmentation:
    """Segments users based on their consumption habits"""
    
    def __init__(self):
        self.model = None
        
    def segment_users(self, data: pd.DataFrame, n_segments: int = 3) -> Dict:
        """Segment users using KMeans clustering"""
        try:
            kmeans = KMeans(n_segments, random_state=42)
            segments = kmeans.fit_predict(data)
            
            return {
                'status': 'success',
                'segments': segments.tolist()
            }
        except Exception as e:
            logger.error(f"Segmentation error: {str(e)}")
            return {'status': 'error', 'message': str(e)}

# Commenting out the Flask part
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     df = pd.DataFrame(data['data'])
#     target_col = data['target_col']
    
#     predictor = ConsumptionPredictor()
#     result = predictor.train_prophet(df, target_col)
#     return jsonify(result)

# @app.route('/anomaly', methods=['POST'])
# def anomaly():
#     data = request.json
#     df = pd.DataFrame(data['data'])
    
#     detector = AnomalyDetector()
#     result = detector.detect_anomalies(df)
#     return jsonify(result)

# @app.route('/eco_score', methods=['POST'])
# def eco_score():
#     data = request.json
#     df = pd.DataFrame(data['data'])
#     benchmarks = data['benchmarks']
    
#     calculator = EcoScoreCalculator()
#     result = calculator.calculate_eco_score(df, benchmarks)
#     return jsonify(result)

# @app.route('/optimize', methods=['POST'])
# def optimize():
#     data = request.json
#     df = pd.DataFrame(data['data'])
    
#     calculator = EcoScoreCalculator()
#     result = calculator.provide_optimization_suggestions(df)
#     return jsonify(result)

# @app.route('/segment', methods=['POST'])
# def segment():
#     data = request.json
#     df = pd.DataFrame(data['data'])
    
#     segmenter = BehavioralSegmentation()
#     result = segmenter.segment_users(df)
#     return jsonify(result)

# @app.route('/predict_catboost', methods=['POST'])
# def predict_catboost():
#     data = request.json
#     df = pd.DataFrame(data['data'])
#     target_col = data['target_col']
    
#     predictor = ConsumptionPredictor()
#     result = predictor.train_catboost(df, target_col)
#     return jsonify(result)

# @app.route('/individual_dash', methods=['POST'])
# def individual_dash():
#     data = request.json
#     df = pd.DataFrame(data['data'])
#     target_col = data['target_col']
    
#     predictor = ConsumptionPredictor()
#     result = predictor.train_prophet(df, target_col)
#     return jsonify(result)

# @app.route('/family_dash', methods=['POST'])
# def family_dash():
#     data = request.json
#     df = pd.DataFrame(data['data'])
    
#     segmenter = BehavioralSegmentation()
#     result = segmenter.segment_users(df)
#     return jsonify(result)

# @app.route('/appliance_dash', methods=['POST'])
# def appliance_dash():
#     data = request.json
#     df = pd.DataFrame(data['data'])
#     target_col = data['target_col']
    
#     predictor = ConsumptionPredictor()
#     result = predictor.train_lstm(df, target_col)
#     return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True)

# Analysis and Graph Generation
def convert_timestamps_to_strings(data: Union[Dict, List]) -> Union[Dict, List]:
    """Convert all Timestamp objects in a nested dictionary or list to strings."""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, pd.Timestamp):
                data[key] = value.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(value, (dict, list)):
                data[key] = convert_timestamps_to_strings(value)
    elif isinstance(data, list):
        for i in range(len(data)):
            if isinstance(data[i], pd.Timestamp):
                data[i] = data[i].strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(data[i], (dict, list)):
                data[i] = convert_timestamps_to_strings(data[i])
    return data

def analyze_individuals(data: pd.DataFrame, target_col: str):
    """Analyze and generate graphs for each individual"""
    individuals = data['individual_id'].unique()
    results = []
    for individual in individuals:
        individual_data = data[data['individual_id'] == individual]
        
        # Generate consumption graph
        plt.figure(figsize=(10, 6))
        plt.plot(individual_data.index, individual_data[target_col], label=f'{individual} Consumption')
        plt.title(f'{individual} Consumption Over Time')
        plt.xlabel('Time')
        plt.ylabel('Consumption')
        plt.legend()
        plt.savefig(f'OUTPUT_2/{individual}_consumption.png')
        plt.close()
        
        # Predict consumption using Prophet
        predictor = ConsumptionPredictor()
        prediction_result = predictor.train_prophet(individual_data, target_col)
        
        # Detect anomalies
        detector = AnomalyDetector()
        anomaly_result = detector.detect_anomalies(individual_data[[target_col]])
        
        results.append({
            'individual': individual,
            'prediction': prediction_result,
            'anomalies': anomaly_result
        })
    
    # Convert Timestamps to strings for JSON serialization
    results = convert_timestamps_to_strings(results)
    
    with open('OUTPUT_2/individual_analysis.json', 'w') as f:
        json.dump(results, f, indent=4)

def analyze_family(data: pd.DataFrame, target_col: str):
    """Analyze and generate graphs for the entire family"""
    plt.figure(figsize=(10, 6))
    for individual in data['individual_id'].unique():
        individual_data = data[data['individual_id'] == individual]
        plt.plot(individual_data.index, individual_data[target_col], label=f'{individual} Consumption')
    plt.title('Family Consumption Over Time')
    plt.xlabel('Time')
    plt.ylabel('Consumption')
    plt.legend()
    plt.savefig('OUTPUT_2/family_consumption.png')
    plt.close()
    
    # Predict consumption using Prophet
    predictor = ConsumptionPredictor()
    prediction_result = predictor.train_prophet(data, target_col)
    
    # Detect anomalies
    detector = AnomalyDetector()
    anomaly_result = detector.detect_anomalies(data[[target_col]])
    
    results = {
        'prediction': prediction_result,
        'anomalies': anomaly_result
    }
    
    # Convert Timestamps to strings for JSON serialization
    results = convert_timestamps_to_strings(results)
    
    with open('OUTPUT_2/family_analysis.json', 'w') as f:
        json.dump(results, f, indent=4)

def analyze_appliances(data: pd.DataFrame, appliances: List[str]):
    """Analyze and generate graphs for top appliances"""
    results = []
    for appliance in appliances:
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data[appliance], label=f'{appliance} Usage')
        plt.title(f'{appliance} Usage Over Time')
        plt.xlabel('Time')
        plt.ylabel('Usage')
        plt.legend()
        plt.savefig(f'OUTPUT_2/{appliance}_usage.png')
        plt.close()
        
        # Predict usage using Prophet
        predictor = ConsumptionPredictor()
        prediction_result = predictor.train_prophet(data, appliance)
        
        # Detect anomalies
        detector = AnomalyDetector()
        anomaly_result = detector.detect_anomalies(data[[appliance]])
        
        results.append({
            'appliance': appliance,
            'prediction': prediction_result,
            'anomalies': anomaly_result
        })
    
    # Convert Timestamps to strings for JSON serialization
    results = convert_timestamps_to_strings(results)
    
    with open('OUTPUT_2/appliance_analysis.json', 'w') as f:
        json.dump(results, f, indent=4)

# Example usage
if __name__ == '__main__':
    # Load your data here
    data = pd.read_csv(r'output/indian_family_lifestyle_data.csv', index_col='datetime', parse_dates=True)
    
    # Preprocess data
    preprocessor = TimeSeriesPreprocessor(data)
    processed_data = preprocessor.preprocess_data()
    
    # Analyze individuals
    analyze_individuals(processed_data, target_col='total_power_consumption')
    
    # Analyze family
    analyze_family(processed_data, target_col='total_power_consumption')
    
    # Analyze top appliances
    top_appliances = ['mixer_grinder_usage', 'pressure_cooker_usage', 'microwave_usage']
    analyze_appliances(processed_data, top_appliances)

def plot_pie_chart(data: pd.Series, title: str, filename: str):
    """Plot a pie chart and save it to a file."""
    plt.figure(figsize=(8, 8))
    data.value_counts().plot.pie(autopct='%1.1f%%')
    plt.title(title)
    plt.ylabel('')
    plt.savefig(filename)
    plt.close()

def plot_bar_chart(data: pd.Series, title: str, xlabel: str, ylabel: str, filename: str):
    """Plot a bar chart and save it to a file."""
    plt.figure(figsize=(10, 6))
    data.value_counts().plot.bar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

def plot_line_chart(data: pd.DataFrame, title: str, xlabel: str, ylabel: str, filename: str):
    """Plot a line chart and save it to a file."""
    plt.figure(figsize=(10, 6))
    for col in data.columns:
        plt.plot(data.index, data[col], label=col)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_3d_chart(data: pd.DataFrame, x_col: str, y_col: str, z_col: str, title: str, filename: str):
    """Plot a 3D chart and save it to a file."""
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[x_col], data[y_col], data[z_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    plt.savefig(filename)
    plt.close()

def analyze_individuals(data: pd.DataFrame, target_col: str):
    """Analyze and generate graphs for each individual"""
    individuals = data['individual_id'].unique()
    results = []
    for individual in individuals:
        individual_data = data[data['individual_id'] == individual]
        
        # Generate consumption graph
        plot_line_chart(individual_data[[target_col]], f'{individual} Consumption Over Time', 'Time', 'Consumption', f'OUTPUT_2/{individual}_consumption.png')
        
        # Predict consumption using Prophet
        predictor = ConsumptionPredictor()
        prediction_result = predictor.train_prophet(individual_data, target_col)
        
        # Detect anomalies
        detector = AnomalyDetector()
        anomaly_result = detector.detect_anomalies(individual_data[[target_col]])
        
        results.append({
            'individual': individual,
            'prediction': prediction_result,
            'anomalies': anomaly_result
        })
    
    # Convert Timestamps to strings for JSON serialization
    results = convert_timestamps_to_strings(results)
    
    with open('OUTPUT_2/individual_analysis.json', 'w') as f:
        json.dump(results, f, indent=4)

def analyze_family(data: pd.DataFrame, target_col: str):
    """Analyze and generate graphs for the entire family"""
    plot_line_chart(data.pivot(columns='individual_id', values=target_col), 'Family Consumption Over Time', 'Time', 'Consumption', 'OUTPUT_2/family_consumption.png')
    
    # Predict consumption using Prophet
    predictor = ConsumptionPredictor()
    prediction_result = predictor.train_prophet(data, target_col)
    
    # Detect anomalies
    detector = AnomalyDetector()
    anomaly_result = detector.detect_anomalies(data[[target_col]])
    
    results = {
        'prediction': prediction_result,
        'anomalies': anomaly_result
    }
    
    # Convert Timestamps to strings for JSON serialization
    results = convert_timestamps_to_strings(results)
    
    with open('OUTPUT_2/family_analysis.json', 'w') as f:
        json.dump(results, f, indent=4)

def analyze_appliances(data: pd.DataFrame, appliances: List[str]):
    """Analyze and generate graphs for top appliances"""
    results = []
    for appliance in appliances:
        plot_line_chart(data[[appliance]], f'{appliance} Usage Over Time', 'Time', 'Usage', f'OUTPUT_2/{appliance}_usage.png')
        
        # Predict usage using Prophet
        predictor = ConsumptionPredictor()
        prediction_result = predictor.train_prophet(data, appliance)
        
        # Detect anomalies
        detector = AnomalyDetector()
        anomaly_result = detector.detect_anomalies(data[[appliance]])
        
        results.append({
            'appliance': appliance,
            'prediction': prediction_result,
            'anomalies': anomaly_result
        })
    
    # Convert Timestamps to strings for JSON serialization
    results = convert_timestamps_to_strings(results)
    
    with open('OUTPUT_2/appliance_analysis.json', 'w') as f:
        json.dump(results, f, indent=4)

def analyze_behavioral_segmentation(data: pd.DataFrame):
    """Analyze and generate graphs for behavioral segmentation"""
    segmenter = BehavioralSegmentation()
    numeric_data = data.select_dtypes(include=[np.number])  # Ensure data is numeric
    segmentation_result = segmenter.segment_users(numeric_data)
    
    # Plot segmentation results
    plot_pie_chart(pd.Series(segmentation_result['segments']), 'Behavioral Segmentation', 'OUTPUT_2/behavioral_segmentation.png')
    
    return segmentation_result

def calculate_eco_score(data: pd.DataFrame, benchmarks: Dict[str, float]):
    """Calculate and generate graphs for Eco Score"""
    calculator = EcoScoreCalculator()
    eco_score_result = calculator.calculate_eco_score(data, benchmarks)
    
    # Plot Eco Score results
    if 'eco_scores' in eco_score_result:
        plot_bar_chart(pd.Series(eco_score_result['eco_scores']), 'Eco Scores', 'Category', 'Score', 'OUTPUT_2/eco_scores.png')
    
    return eco_score_result

def provide_optimization_suggestions(data: pd.DataFrame):
    """Provide optimization suggestions based on usage patterns"""
    calculator = EcoScoreCalculator()
    suggestions_result = calculator.provide_optimization_suggestions(data)
    
    return suggestions_result

# Example usage
if __name__ == '__main__':
    # Load your data here
    data = pd.read_csv(r'output/indian_family_lifestyle_data.csv', index_col='datetime', parse_dates=True)
    
    # Preprocess data
    preprocessor = TimeSeriesPreprocessor(data)
    processed_data = preprocessor.preprocess_data()
    
    # Analyze individuals
    analyze_individuals(processed_data, target_col='total_power_consumption')
    
    # Analyze family
    analyze_family(processed_data, target_col='total_power_consumption')
    
    # Analyze top appliances
    top_appliances = ['mixer_grinder_usage', 'pressure_cooker_usage', 'microwave_usage']
    analyze_appliances(processed_data, top_appliances)
    
    # Behavioral segmentation
    segmentation_result = analyze_behavioral_segmentation(processed_data)
    
    # Calculate Eco Score
    benchmarks = {'total_power_consumption': 1000, 'mixer_grinder_usage': 50, 'pressure_cooker_usage': 30, 'microwave_usage': 20}
    eco_score_result = calculate_eco_score(processed_data, benchmarks)
    
    # Provide optimization suggestions
    suggestions_result = provide_optimization_suggestions(processed_data)
    
    # Store results in JSON
    results = {
        'behavioral_segmentation': segmentation_result,
        'eco_score': eco_score_result,
        'optimization_suggestions': suggestions_result
    }
    
    with open('OUTPUT_2/analysis_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print results
    print("Behavioral Segmentation Result:", segmentation_result)
    print("Eco Score Result:", eco_score_result)
    print("Optimization Suggestions:", suggestions_result)