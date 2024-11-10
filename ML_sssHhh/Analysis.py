import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict
from catboost import CatBoostRegressor
import logging
import seaborn as sns
import google.generativeai as genai
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure output directory exists
output_dir = 'output_3'
os.makedirs(output_dir, exist_ok=True)

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
    def aggregate_daily_data(self) -> pd.DataFrame:
        """Aggregate data to daily summaries"""
        df = self.data.copy()
        df['date'] = df.index.date
        daily_data = df.groupby(['date', 'individual_id']).sum().reset_index()
        return daily_data


class ConsumptionPredictor:
    """Handles time series prediction using various models"""
    
    def __init__(self):
        self.prophet_model = None
        self.lstm_model = None
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
                'forecast': forecast
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
                'feature_importances': self.catboost_model.get_feature_importance()
            }
        except Exception as e:
            logger.error(f"CatBoost training error: {str(e)}")
            return {'status': 'error', 'message': str(e)}


class AnomalyDetector:
    """Handles anomaly detection in consumption patterns"""
    
    def __init__(self):
        self.model = None
        
    def detect_anomalies(self, data: pd.DataFrame, target_col: str, threshold: float = 20.0) -> Dict:
        """Detect anomalies based on deviation from mean consumption"""
        try:
            anomalies = []
            mean_consumption = data[target_col].mean()
            
            for index, row in data.iterrows():
                if abs(row[target_col] - mean_consumption) > threshold:
                    anomalies.append({
                        'date': index,
                        'value': row[target_col],
                        'deviation': row[target_col] - mean_consumption
                    })
            
            # Generate natural language summary
            summary = f"Detected {len(anomalies)} anomalies in the last 7 days."
            for anomaly in anomalies:
                summary += f"\nOn {anomaly['date']}, the consumption was {anomaly['value']} which deviated by {anomaly['deviation']} from the mean."
            
            # Save summary to a text file
            with open('anomalies_summary.txt', 'w') as file:
                file.write(summary)
            
            return {
                'status': 'success',
                'summary': summary
            }
        except Exception as e:
            logger.error(f"Anomaly detection error: {str(e)}")
            return {'status': 'error', 'message': str(e)}


class EcoScoreCalculator:
    """Calculates Eco Score and provides optimization suggestions"""
    
    def __init__(self):
        self.model = None
        
    def calculate_eco_score(self, data: pd.DataFrame, benchmarks: Dict[str, float]) -> Dict:
        """Calculate daily Eco Score based on benchmarks and last 7-day history"""
        try:
            scores = {}
            for col in data.columns:
                if (col in benchmarks) and (benchmarks[col] != 0):  # Ensure benchmark is not zero
                    daily_scores = []
                    for i in range(len(data)):
                        daily_score = 100 - ((data[col].iloc[i] / benchmarks[col]) * 100)
                        daily_scores.append(daily_score)
                    scores[col] = daily_scores
                else:
                    logger.warning(f"Benchmark for {col} not provided or is zero.")
            
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
            if data['total_power_consumption'].mean() > data['total_power_consumption'].quantile(0.75):
                suggestions.append("Consider reducing peak hour usage.")
            if data['total_power_consumption'].std() > data['total_power_consumption'].mean() * 0.5:
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
                'segments': segments
            }
        except Exception as e:
            logger.error(f"Segmentation error: {str(e)}")
            return {'status': 'error', 'message': str(e)}


class EnhancedAnomalyDetector:
    """Enhanced anomaly detection with Gemini integration for natural language insights"""
    
    def __init__(self, gemini_api_key: str):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def detect_anomalies(self, data: pd.DataFrame, target_col: str) -> Dict:
        """
        Detect anomalies and generate natural language insights using Gemini
        """
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data[[target_col]])
            
            # Detect anomalies using Isolation Forest
            anomaly_labels = self.isolation_forest.fit_predict(scaled_data)
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            # Extract anomaly information
            anomalies = []
            for idx in anomaly_indices:
                anomalies.append({
                    'date': data.index[idx],
                    'value': data[target_col].iloc[idx],
                    'deviation': abs(data[target_col].iloc[idx] - data[target_col].mean())
                })
            
            # Prepare context for Gemini
            context = self._prepare_anomaly_context(data, anomalies, target_col)
            
            # Generate natural language insights using Gemini
            insights = self._generate_insights(context, data, target_col)
            
            return {
                'status': 'success',
                'anomalies': anomalies,
                'insights': insights
            }
        except Exception as e:
            logger.error(f"Anomaly detection error: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _prepare_anomaly_context(self, data: pd.DataFrame, anomalies: List[Dict], target_col: str) -> str:
        """Prepare context for Gemini analysis"""
        context = f"""
        Analyze the following energy consumption data:
        
        Time period: {data.index.min()} to {data.index.max()}
        Average consumption: {data[target_col].mean():.2f}
        Number of anomalies detected: {len(anomalies)}
        
        Anomaly details:
        """
        
        for anomaly in anomalies:
            context += f"\nDate: {anomaly['date']}, Value: {anomaly['value']:.2f}, Deviation: {anomaly['deviation']:.2f}"
        
        # Include family member names
        family_members = data['individual_id'].unique()
        context += f"\nFamily members: {', '.join(family_members)}"
        
        return context
    
    def _generate_insights(self, context: str, data: pd.DataFrame, target_col: str) -> str:
        """Generate natural language insights using Gemini"""
        # Generate future predictions using Prophet
        predictor = ConsumptionPredictor()
        prediction_result = predictor.train_prophet(data, target_col)
        
        if prediction_result['status'] == 'success':
            forecast = prediction_result['forecast']
            future_predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_dict(orient='records')
        else:
            future_predictions = []
        
        prompt = f"""
        {context}
        
        Please provide a detailed analysis of the energy consumption patterns and anomalies. Include:
        1. Personalized insights about the carbon footprint and activities of each family member throughout the week.
        2. Access to all the database, including historical data and future predictions.
        3. Analysis of the EcoScore and identification of cases where the footprint is increasing or spiking.
        4. Include the future predictions for the next 30 days.
        5. The insights must be in normal English language that users can understand.
        6. Output in JSON OBJECT format.
        
        Format the response in clear, natural language suitable for non-technical users.
        
        Future Predictions:
        {json.dumps(future_predictions, indent=2)}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": "Due to API limitations, detailed insights are not available at the moment. Please try again later."
            })


class EnhancedFamilyAnalyzer:
    """Enhanced family carbon footprint analysis with improved visualizations"""
    
    def __init__(self):
        self.plt_style = {
            'figure.figsize': (12, 6),
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        }
    
    def plot_family_carbon_footprint(self, data: pd.DataFrame, target_col: str):
        """
        Create enhanced visualization of family carbon footprint
        """
        plt.style.use('ggplot')  # Use a valid matplotlib style
        for key, value in self.plt_style.items():
            plt.rcParams[key] = value
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Enhanced histogram
        sns.histplot(data=data, x=target_col, hue='individual_id', 
                    multiple="layer", alpha=0.6, ax=ax1)
        ax1.set_title('Family Carbon Footprint Distribution')
        ax1.set_xlabel('Carbon Footprint (kg CO2)')
        ax1.set_ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, 'family_carbon_footprint_distribution.svg'))
        
        # Box plot for comparison
        sns.boxplot(data=data, x='individual_id', y=target_col, ax=ax2)
        ax2.set_title('Carbon Footprint Comparison')
        ax2.set_xlabel('Family Member')
        ax2.set_ylabel('Carbon Footprint (kg CO2)')
        plt.savefig(os.path.join(output_dir, 'family_carbon_footprint_comparison.svg'))
        
        plt.tight_layout()
        plt.close(fig)
        return fig


# Analysis and Graph Generation


def analyze_individuals(data: pd.DataFrame, target_col: str):
    """Analyze and generate graphs for each individual"""
    individuals = data['individual_id'].unique()
    for individual in individuals:
        individual_data = data[data['individual_id'] == individual]
        individual_data.set_index(pd.to_datetime(individual_data['date']), inplace=True)  # Ensure DatetimeIndex
        last_7_days = individual_data.last('7D')
        
        # Carbon footprint bar graph for last 7 days
        plt.figure(figsize=(10, 6))
        last_7_days[target_col].plot(kind='bar')
        plt.title(f'{individual} Carbon Footprint (Last 7 Days)')
        plt.xlabel('Date')
        plt.ylabel('Carbon Footprint')
        plt.savefig(os.path.join(output_dir, f'{individual}_carbon_footprint.svg'))
        plt.close()
        
        # Anomaly detection
        detector = AnomalyDetector()
        anomaly_result = detector.detect_anomalies(last_7_days, target_col)
        
        # Anomaly summary
        if anomaly_result['status'] == 'success':
            print(f"Anomalies for {individual}:")
            print(anomaly_result['summary'])
        
        # Eco Score calculation
        calculator = EcoScoreCalculator()
        eco_score_result = calculator.calculate_eco_score(last_7_days, {'total_power_consumption': 1000})
        
        # Eco Score line graph
        plt.figure(figsize=(10, 6))
        pd.Series(eco_score_result['eco_scores']['total_power_consumption']).plot(kind='line')
        plt.title(f'{individual} Eco Score (Last 7 Days)')
        plt.xlabel('Date')
        plt.ylabel('Eco Score')
        plt.savefig(os.path.join(output_dir, f'{individual}_eco_score.svg'))
        plt.close()


def analyze_family(data: pd.DataFrame, target_col: str):
    """Analyze and generate graphs for the entire family"""
    data.set_index(pd.to_datetime(data['date']), inplace=True)  # Ensure DatetimeIndex
    last_7_days = data.last('7D')
    
    # Major activities with high carbon footprint (multiple ring graph)
    if 'activity' in last_7_days.columns:
        activity_data = last_7_days.groupby('activity')[target_col].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(activity_data, labels=activity_data.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
    plt.savefig(os.path.join(output_dir, 'family_activities_carbon_footprint.svg'))
    plt.close()
    
    # Family member comparison of carbon footprint (double histogram)
    plt.figure(figsize=(10, 6))
    for individual in data['individual_id'].unique():
        individual_data = data[data['individual_id'] == individual]
        plt.hist(individual_data[target_col], alpha=0.5, label=individual, bins=15)
    plt.title('Family Member Comparison of Carbon Footprint (Last 7 Days)')
    plt.xlabel('Carbon Footprint')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'family_comparison_carbon_footprint.svg'))
    plt.close()
    
    # Provide benchmarks for all required columns
    benchmarks = {
        'total_power_consumption': 1000,
        'mixer_grinder_usage': 100,
        'pressure_cooker_usage': 100,
        'microwave_usage': 100,
        'refrigerator_usage': 100,
        'water_purifier_usage': 100,
        'tv_usage': 100,
        'smartphone_usage': 100,
        'laptop_usage': 100,
        'tablet_usage': 100,
        'air_conditioner_usage': 100,
        'fan_usage': 100,
        'water_heater_usage': 100,
        'eating_habits': 100,
        'vehicle_monthly_distance_km': 100,
        'frequency_of_traveling_by_air': 100,
        'vehicle_type': 100,
        'new_clothes_monthly': 100
    }
    
    calculator = EcoScoreCalculator()
    eco_score_result = calculator.calculate_eco_score(last_7_days, benchmarks)
    
    plt.figure(figsize=(10, 6))
    pd.Series(eco_score_result['eco_scores']['total_power_consumption']).plot(kind='line')
    plt.title('Consolidated Eco Score of the Family (Last 7 Days)')
    plt.xlabel('Date')
    plt.ylabel('Eco Score')
    plt.savefig(os.path.join(output_dir, 'family_eco_score.svg'))
    plt.close()


def analyze_appliances(data: pd.DataFrame, appliances: List[str]):
    """Analyze and generate graphs for top appliances"""
    data.set_index(pd.to_datetime(data['date']), inplace=True)  # Ensure DatetimeIndex
    last_7_days = data.last('7D')
    
    for appliance in appliances:
        # Comparative graph of electricity used and carbon footprint
        plt.figure(figsize=(10, 6))
        last_7_days[appliance].plot(kind='bar', color='blue', label='Electricity Used')
        last_7_days['total_power_consumption'].plot(kind='bar', color='green', alpha=0.5, label='Carbon Footprint')
        plt.title(f'{appliance} Electricity Used vs Carbon Footprint (Last 7 Days)')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{appliance}_electricity_vs_carbon.svg'))
        plt.close()
        
        # Pie chart of appliance usage by various users
        plt.figure(figsize=(8, 8))
        last_7_days.groupby('individual_id')[appliance].sum().plot(kind='pie', autopct='%1.1f%%')
        plt.title(f'{appliance} Usage by Various Users (Last 7 Days)')
        plt.ylabel('')
        plt.savefig(os.path.join(output_dir, f'{appliance}_usage_by_users.svg'))
        plt.close()


# Example usage


if __name__ == '__main__':
    # Load your data here
    data = pd.read_csv(r'output/indian_family_lifestyle_data.csv', index_col='datetime', parse_dates=True)
    
    # Preprocess data
    preprocessor = TimeSeriesPreprocessor(data)
    processed_data = preprocessor.preprocess_data()
    daily_data = preprocessor.aggregate_daily_data()
    
    # Save preprocessed data to CSV
    processed_data.to_csv(os.path.join(output_dir, 'processed_data.csv'))
    daily_data.to_csv(os.path.join(output_dir, 'daily_data.csv'))
    
    # Analyze individuals
    analyze_individuals(daily_data, target_col='total_power_consumption')
    
    # Analyze family
    analyze_family(daily_data, target_col='total_power_consumption')
    
    # Analyze top appliances
    top_appliances = ['mixer_grinder_usage', 'pressure_cooker_usage', 'microwave_usage']
    analyze_appliances(daily_data, top_appliances)
    
    # Initialize enhanced analyzers
    anomaly_detector = EnhancedAnomalyDetector(gemini_api_key='AIzaSyAqfzrYDqh7lRvaX7YIZrCAosyMerJmHnY')
    family_analyzer = EnhancedFamilyAnalyzer()
    
    # Analyze family carbon footprint with enhanced visualizations
    fig = family_analyzer.plot_family_carbon_footprint(daily_data, 'total_power_consumption')
    plt.show()
    
    # Detect anomalies and generate insights
    analysis_result = anomaly_detector.detect_anomalies(daily_data, 'total_power_consumption')
    
    if analysis_result['status'] == 'success':
        print("\nAI-Generated Insights:")
        print(analysis_result['insights'])
        
        # Save Gemini output to text files
        with open(os.path.join(output_dir, 'gemini_alerts.txt'), 'w') as alert_file:
            alert_file.write(analysis_result['insights'])
    else:
        print(f"Error in analysis: {analysis_result['message']}")





