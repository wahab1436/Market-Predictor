import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class AnomalyDetection:
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.anomaly_scores = None
        
    def detect_anomalies(self, df_features):
        """Detect anomalies in financial data"""
        # Select features for anomaly detection
        anomaly_features = ['close', 'volume', 'daily_return', 'volatility_20', 'rsi_14']
        
        # Ensure features exist
        available_features = [f for f in anomaly_features if f in df_features.columns]
        
        if len(available_features) < 2:
            available_features = [col for col in df_features.columns if col not in ['date']]
            available_features = available_features[:5]  # Take first 5 numeric columns
        
        X = df_features[available_features].fillna(0).values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit and predict
        self.model.fit(X_scaled)
        anomaly_predictions = self.model.predict(X_scaled)
        anomaly_scores = self.model.decision_function(X_scaled)
        
        # -1 = anomaly, 1 = normal
        df_anomalies = df_features.copy()
        df_anomalies['is_anomaly'] = (anomaly_predictions == -1).astype(int)
        df_anomalies['anomaly_score'] = anomaly_scores
        df_anomalies['anomaly_confidence'] = 1 - (1 / (1 + np.exp(-np.abs(anomaly_scores))))
        
        # Mark anomaly types
        df_anomalies['anomaly_type'] = self._classify_anomalies(df_anomalies)
        
        self.anomaly_scores = df_anomalies
        return df_anomalies
    
    def _classify_anomalies(self, df_anomalies):
        """Classify anomalies into types"""
        anomaly_types = []
        
        for idx, row in df_anomalies.iterrows():
            if row['is_anomaly'] == 1:
                # Check for price anomalies
                if row.get('daily_return', 0) > 0.05:  # Large positive return
                    anomaly_types.append('Price Spike')
                elif row.get('daily_return', 0) < -0.05:  # Large negative return
                    anomaly_types.append('Price Crash')
                elif row.get('volume_ratio', 1) > 2:  # High volume
                    anomaly_types.append('Volume Surge')
                elif row.get('rsi_14', 50) > 80:  # Overbought
                    anomaly_types.append('Overbought')
                elif row.get('rsi_14', 50) < 20:  # Oversold
                    anomaly_types.append('Oversold')
                else:
                    anomaly_types.append('General Anomaly')
            else:
                anomaly_types.append('Normal')
        
        return anomaly_types
    
    def get_anomaly_summary(self, df_anomalies):
        """Generate summary of anomalies"""
        total_points = len(df_anomalies)
        anomaly_points = df_anomalies['is_anomaly'].sum()
        
        summary = {
            'total_data_points': total_points,
            'anomaly_count': int(anomaly_points),
            'anomaly_percentage': round((anomaly_points / total_points) * 100, 2),
            'most_common_type': df_anomalies[df_anomalies['is_anomaly'] == 1]['anomaly_type'].mode().iloc[0] if anomaly_points > 0 else 'None',
            'avg_anomaly_score': df_anomalies['anomaly_confidence'].mean(),
            'max_anomaly_score': df_anomalies['anomaly_confidence'].max()
        }
        
        return summary
