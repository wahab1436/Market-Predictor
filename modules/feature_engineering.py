import pandas as pd
import numpy as np
import ta

class FeatureEngineering:
    def __init__(self):
        self.features_df = None
        
    def create_features(self, df):
        """Create technical indicators and features"""
        df_features = df.copy()
        
        # Ensure we have required columns
        required_cols = ['close', 'high', 'low', 'volume']
        if not all(col in df_features.columns for col in required_cols):
            # Try to find similar columns
            col_mapping = {}
            for req in required_cols:
                for col in df_features.columns:
                    if req in col.lower():
                        col_mapping[req] = col
                        break
            
            # Rename columns for consistency
            df_features = df_features.rename(columns={v: k for k, v in col_mapping.items()})
        
        # Calculate returns
        df_features['daily_return'] = df_features['close'].pct_change()
        df_features['log_return'] = np.log(df_features['close'] / df_features['close'].shift(1))
        
        # Moving averages
        df_features['sma_5'] = df_features['close'].rolling(window=5).mean()
        df_features['sma_20'] = df_features['close'].rolling(window=20).mean()
        df_features['ema_12'] = df_features['close'].ewm(span=12, adjust=False).mean()
        df_features['ema_26'] = df_features['close'].ewm(span=26, adjust=False).mean()
        
        # Volatility
        df_features['volatility_5'] = df_features['daily_return'].rolling(window=5).std()
        df_features['volatility_20'] = df_features['daily_return'].rolling(window=20).std()
        
        # RSI
        df_features['rsi_14'] = ta.momentum.RSIIndicator(df_features['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df_features['close'])
        df_features['macd'] = macd.macd()
        df_features['macd_signal'] = macd.macd_signal()
        df_features['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df_features['close'], window=20, window_dev=2)
        df_features['bb_upper'] = bb.bollinger_hband()
        df_features['bb_lower'] = bb.bollinger_lband()
        df_features['bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['close']
        
        # Volume indicators
        df_features['volume_sma'] = df_features['volume'].rolling(window=20).mean()
        df_features['volume_ratio'] = df_features['volume'] / df_features['volume_sma']
        
        # Lag features
        for lag in [1, 3, 7]:
            df_features[f'close_lag_{lag}'] = df_features['close'].shift(lag)
            df_features[f'return_lag_{lag}'] = df_features['daily_return'].shift(lag)
            df_features[f'volume_lag_{lag}'] = df_features['volume'].shift(lag)
        
        # Price position
        df_features['high_low_range'] = (df_features['high'] - df_features['low']) / df_features['close']
        df_features['close_position'] = (df_features['close'] - df_features['low']) / (df_features['high'] - df_features['low'])
        
        # Drop NaN values from rolling calculations
        df_features = df_features.dropna()
        
        self.features_df = df_features
        return df_features
    
    def get_target_variables(self, df_features):
        """Create target variables for ML models"""
        df_targets = df_features.copy()
        
        # Next day price (regression target)
        df_targets['next_day_close'] = df_targets['close'].shift(-1)
        
        # Next day return classification
        df_targets['next_day_return'] = df_targets['close'].shift(-1) / df_targets['close'] - 1
        df_targets['next_day_direction'] = (df_targets['next_day_return'] > 0).astype(int)
        
        # Remove the last row (no target)
        df_targets = df_targets.iloc[:-1]
        
        return df_targets
    
    def get_feature_list(self):
        """Get list of engineered features"""
        base_features = ['close', 'high', 'low', 'volume', 'daily_return']
        technical_features = ['sma_5', 'sma_20', 'ema_12', 'ema_26', 
                             'volatility_5', 'volatility_20', 'rsi_14',
                             'macd', 'macd_signal', 'macd_diff',
                             'bb_upper', 'bb_lower', 'bb_width']
        lag_features = [f'close_lag_{i}' for i in [1, 3, 7]] + \
                      [f'return_lag_{i}' for i in [1, 3, 7]] + \
                      [f'volume_lag_{i}' for i in [1, 3, 7]]
        other_features = ['volume_sma', 'volume_ratio', 'high_low_range', 'close_position']
        
        return base_features + technical_features + lag_features + other_features
