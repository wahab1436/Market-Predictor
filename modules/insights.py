import pandas as pd
import numpy as np
from datetime import datetime

class InsightsGenerator:
    def __init__(self):
        self.insights = []
        
    def generate_insights(self, df_features, model_results, anomaly_results):
        """Generate AI-driven insights from data"""
        self.insights = []
        
        # 1. Price trend insights
        self._generate_price_insights(df_features)
        
        # 2. Volume insights
        self._generate_volume_insights(df_features)
        
        # 3. Technical indicator insights
        self._generate_technical_insights(df_features)
        
        # 4. Model performance insights
        self._generate_model_insights(model_results)
        
        # 5. Anomaly insights
        self._generate_anomaly_insights(anomaly_results)
        
        # 6. Market condition insights
        self._generate_market_insights(df_features)
        
        return self.insights
    
    def _generate_price_insights(self, df_features):
        """Generate insights about price movements"""
        if 'close' not in df_features.columns:
            return
        
        # Calculate recent trends
        recent_days = min(5, len(df_features))
        recent_prices = df_features['close'].iloc[-recent_days:]
        price_change = ((recent_prices.iloc[-1] / recent_prices.iloc[0]) - 1) * 100
        
        if price_change > 5:
            self.insights.append(f"The price increased by {price_change:.1f}% over the last {recent_days} days, indicating strong positive momentum.")
        elif price_change < -5:
            self.insights.append(f"The price decreased by {abs(price_change):.1f}% over the last {recent_days} days, showing negative pressure.")
        else:
            self.insights.append(f"The price remained relatively stable over the last {recent_days} days with a {price_change:.1f}% change.")
        
        # Volatility insight
        if 'volatility_20' in df_features.columns:
            current_vol = df_features['volatility_20'].iloc[-1]
            avg_vol = df_features['volatility_20'].mean()
            
            if current_vol > avg_vol * 1.5:
                self.insights.append(f"Current volatility ({current_vol:.4f}) is significantly higher than the average ({avg_vol:.4f}), indicating increased market uncertainty.")
    
    def _generate_volume_insights(self, df_features):
        """Generate insights about trading volume"""
        if 'volume' not in df_features.columns:
            return
        
        recent_volume = df_features['volume'].iloc[-5:].mean()
        avg_volume = df_features['volume'].mean()
        volume_ratio = recent_volume / avg_volume
        
        if volume_ratio > 1.5:
            self.insights.append(f"Recent trading volume is {volume_ratio:.1f}x higher than average, suggesting increased investor interest.")
        elif volume_ratio < 0.5:
            self.insights.append(f"Recent trading volume is {volume_ratio:.1f}x lower than average, indicating reduced market activity.")
        
        # Volume spikes
        if 'volume_ratio' in df_features.columns:
            recent_spikes = (df_features['volume_ratio'].iloc[-10:] > 2).sum()
            if recent_spikes > 0:
                self.insights.append(f"Detected {recent_spikes} volume spikes in the last 10 days, potentially indicating significant news or events.")
    
    def _generate_technical_insights(self, df_features):
        """Generate insights from technical indicators"""
        if 'rsi_14' in df_features.columns:
            current_rsi = df_features['rsi_14'].iloc[-1]
            
            if current_rsi > 70:
                self.insights.append(f"RSI is at {current_rsi:.1f}, indicating overbought conditions which may suggest a potential pullback.")
            elif current_rsi < 30:
                self.insights.append(f"RSI is at {current_rsi:.1f}, indicating oversold conditions which may suggest a potential rebound.")
        
        if 'sma_5' in df_features.columns and 'sma_20' in df_features.columns:
            sma_5 = df_features['sma_5'].iloc[-1]
            sma_20 = df_features['sma_20'].iloc[-1]
            
            if sma_5 > sma_20:
                self.insights.append(f"Short-term moving average (5-day) is above long-term average (20-day), suggesting bullish momentum.")
            else:
                self.insights.append(f"Short-term moving average (5-day) is below long-term average (20-day), suggesting bearish pressure.")
    
    def _generate_model_insights(self, model_results):
        """Generate insights from model performance"""
        if not model_results:
            return
        
        # Find best performing model
        best_model = None
        best_metric = -np.inf
        
        for model_name, results in model_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
                # Use test R² for regression, accuracy for classification
                if 'test_r2' in metrics:
                    if metrics['test_r2'] > best_metric:
                        best_metric = metrics['test_r2']
                        best_model = model_name
                elif 'test_accuracy' in metrics:
                    if metrics['test_accuracy'] > best_metric:
                        best_metric = metrics['test_accuracy']
                        best_model = model_name
        
        if best_model:
            self.insights.append(f"The {best_model.replace('_', ' ').title()} model achieved the best performance with a test metric of {best_metric:.3f}.")
    
    def _generate_anomaly_insights(self, anomaly_results):
        """Generate insights from anomaly detection"""
        if anomaly_results is None or 'is_anomaly' not in anomaly_results.columns:
            return
        
        anomaly_count = anomaly_results['is_anomaly'].sum()
        total_days = len(anomaly_results)
        anomaly_percentage = (anomaly_count / total_days) * 100
        
        if anomaly_percentage > 10:
            self.insights.append(f"High anomaly rate detected: {anomaly_count} anomalies ({anomaly_percentage:.1f}% of data), suggesting unusual market conditions.")
        elif anomaly_count > 0:
            self.insights.append(f"Detected {anomaly_count} anomalies ({anomaly_percentage:.1f}% of data) in the dataset.")
        
        # Most recent anomaly
        recent_anomalies = anomaly_results[anomaly_results['is_anomaly'] == 1].tail(1)
        if not recent_anomalies.empty:
            self.insights.append(f"Most recent anomaly detected with confidence {recent_anomalies['anomaly_confidence'].iloc[0]:.1%}.")
    
    def _generate_market_insights(self, df_features):
        """Generate general market condition insights"""
        if 'daily_return' not in df_features.columns:
            return
        
        # Market trend
        positive_days = (df_features['daily_return'] > 0).sum()
        negative_days = (df_features['daily_return'] < 0).sum()
        total_days = len(df_features)
        
        if positive_days / total_days > 0.6:
            self.insights.append(f"The market showed positive momentum with {positive_days} up days out of {total_days}.")
        elif negative_days / total_days > 0.6:
            self.insights.append(f"The market showed negative pressure with {negative_days} down days out of {total_days}.")
        
        # Risk assessment
        if 'volatility_20' in df_features.columns:
            current_risk = df_features['volatility_20'].iloc[-1]
            if current_risk > 0.03:
                self.insights.append(f"High volatility environment detected, suggesting increased market risk.")
