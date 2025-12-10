import pandas as pd
import numpy as np
import ta

class FeatureEngineer:
    def __init__(self):
        pass
    
    def create_features(self, df):
        """Create technical indicators and features"""
        df_features = df.copy()
        
        # Ensure we have required price data
        if 'close' not in df_features.columns:
            raise ValueError("Data must contain 'close' column")
        
        # Calculate returns
        df_features['returns'] = df_features['close'].pct_change()
        df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df_features[f'sma_{window}'] = df_features['close'].rolling(window=window).mean()
            df_features[f'ema_{window}'] = df_features['close'].ewm(span=window, adjust=False).mean()
        
        # Volatility
        df_features['volatility_5'] = df_features['returns'].rolling(window=5).std()
        df_features['volatility_20'] = df_features['returns'].rolling(window=20).std()
        
        # RSI
        try:
            df_features['rsi_14'] = ta.momentum.RSIIndicator(df_features['close'], window=14).rsi()
        except:
            pass
        
        # MACD
        try:
            macd = ta.trend.MACD(df_features['close'])
            df_features['macd'] = macd.macd()
            df_features['macd_signal'] = macd.macd_signal()
            df_features['macd_diff'] = macd.macd_diff()
        except:
            pass
        
        # Bollinger Bands
        try:
            bb = ta.volatility.BollingerBands(df_features['close'], window=20, window_dev=2)
            df_features['bb_upper'] = bb.bollinger_hband()
            df_features['bb_middle'] = bb.bollinger_mavg()
            df_features['bb_lower'] = bb.bollinger_lband()
        except:
            pass
        
        # Volume indicators (if volume exists)
        if 'volume' in df_features.columns:
            df_features['volume_sma_20'] = df_features['volume'].rolling(window=20).mean()
            df_features['volume_ratio'] = df_features['volume'] / df_features['volume_sma_20']
        
        # Lag features
        for lag in [1, 2, 3, 5, 7]:
            df_features[f'close_lag_{lag}'] = df_features['close'].shift(lag)
            df_features[f'returns_lag_{lag}'] = df_features['returns'].shift(lag)
        
        # Target variables for ML
        df_features['target_next_close'] = df_features['close'].shift(-1)
        df_features['target_direction'] = (df_features['target_next_close'] > df_features['close']).astype(int)
        
        # Drop NaN values
        df_features = df_features.dropna()
        
        return df_features
