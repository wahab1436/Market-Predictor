import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config FIRST
st.set_page_config(
    page_title="Financial Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional design
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    
    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0B3D91;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #0B3D91;
        padding-bottom: 0.5rem;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0B3D91;
        margin: 1.5rem 0 1rem 0;
    }
    
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #0B3D91;
        margin-bottom: 1rem;
    }
    
    .insight-card {
        background: #fff;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid #dee2e6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .upload-area {
        border: 2px dashed #dee2e6;
        border-radius: 10px;
        padding: 3rem 1rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
        transition: all 0.3s;
    }
    
    .upload-area:hover {
        border-color: #0B3D91;
        background: #e9ecef;
    }
    
    .stTabs [aria-selected="true"] {
        background: #0B3D91 !important;
        color: white !important;
    }
    
    .stButton > button {
        background-color: #0B3D91;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #0A3179;
    }
    
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #dee2e6;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Define all processing classes directly in app.py
class DataProcessor:
    def process(self, df):
        """Process raw dataframe"""
        try:
            df_clean = df.copy()
            
            # Standardize column names
            df_clean.columns = [self._standardize_column_name(col) for col in df_clean.columns]
            
            # Identify date column
            date_col = self._identify_date_column(df_clean)
            if date_col:
                try:
                    df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
                    df_clean = df_clean.sort_values(date_col).reset_index(drop=True)
                except:
                    # If date conversion fails, use index as date
                    df_clean['date'] = range(len(df_clean))
            
            # Ensure we have a date column
            if 'date' not in df_clean.columns:
                df_clean['date'] = range(len(df_clean))
            
            # Identify and convert numeric columns
            for col in df_clean.columns:
                if col != 'date':
                    try:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    except:
                        pass
            
            # Handle missing values
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].ffill().bfill()
            
            # Remove any remaining nulls
            df_clean = df_clean.dropna()
            
            # Ensure we have at least some data
            if len(df_clean) == 0:
                raise ValueError("No valid data found after cleaning")
                
            return df_clean
            
        except Exception as e:
            st.error(f"Error in data processing: {str(e)}")
            return df
    
    def _standardize_column_name(self, col_name):
        """Standardize column names"""
        if not isinstance(col_name, str):
            col_name = str(col_name)
        
        col_name = col_name.strip().lower()
        
        # Common mappings
        mappings = {
            'timestamp': 'date',
            'datetime': 'date',
            'time': 'date',
            'adj close': 'close',
            'last': 'close',
            'price': 'close',
            'closing': 'close',
            'close price': 'close',
            'qty': 'volume',
            'amount': 'volume',
            'quantity': 'volume'
        }
        
        for key, value in mappings.items():
            if key in col_name:
                return value
        
        # Replace spaces with underscores
        col_name = col_name.replace(' ', '_').replace('.', '_').replace('(', '').replace(')', '')
        
        return col_name
    
    def _identify_date_column(self, df):
        """Identify date/time column"""
        date_keywords = ['date', 'time', 'timestamp', 'datetime']
        
        for col in df.columns:
            col_lower = str(col).lower()
            for keyword in date_keywords:
                if keyword in col_lower:
                    return col
        
        # Check if first column looks like dates
        if len(df) > 0:
            first_col = df.columns[0]
            try:
                sample_value = str(df[first_col].iloc[0])
                # Simple date pattern check
                date_patterns = ['-', '/', '202', '201', '200', 'jan', 'feb', 'mar']
                if any(pattern in sample_value.lower() for pattern in date_patterns):
                    return first_col
            except:
                pass
        
        return None

class FeatureEngineer:
    def create_features(self, df):
        """Create technical indicators and features"""
        try:
            df_features = df.copy()
            
            # Ensure we have required price data
            if 'close' not in df_features.columns:
                # Try to find price column
                for col in df_features.columns:
                    if col not in ['date'] and df_features[col].dtype in [np.float64, np.int64]:
                        df_features = df_features.rename(columns={col: 'close'})
                        break
            
            if 'close' not in df_features.columns or len(df_features) < 5:
                st.warning("Insufficient data for feature engineering. Using basic features only.")
                return df_features
            
            # Calculate returns safely
            if len(df_features) > 1:
                df_features['returns'] = df_features['close'].pct_change()
                df_features['returns'] = df_features['returns'].fillna(0)
                
                # Moving averages (only if we have enough data)
                for window in [5, 10, 20]:
                    if len(df_features) >= window:
                        df_features[f'sma_{window}'] = df_features['close'].rolling(window=window).mean()
                        df_features[f'ema_{window}'] = df_features['close'].ewm(span=window, adjust=False).mean()
                
                # Volatility
                if 'returns' in df_features.columns and len(df_features) >= 5:
                    df_features['volatility_5'] = df_features['returns'].rolling(window=5).std()
                    df_features['volatility_20'] = df_features['returns'].rolling(window=20).std()
                
                # RSI (simplified calculation)
                if len(df_features) >= 14:
                    delta = df_features['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df_features['rsi'] = 100 - (100 / (1 + rs))
                    df_features['rsi'] = df_features['rsi'].fillna(50)  # Default to neutral
                
                # MACD (simplified)
                if len(df_features) >= 26:
                    exp1 = df_features['close'].ewm(span=12, adjust=False).mean()
                    exp2 = df_features['close'].ewm(span=26, adjust=False).mean()
                    df_features['macd'] = exp1 - exp2
                    df_features['macd_signal'] = df_features['macd'].ewm(span=9, adjust=False).mean()
                
                # Volume indicators
                if 'volume' in df_features.columns and len(df_features) >= 20:
                    df_features['volume_sma'] = df_features['volume'].rolling(window=20).mean()
                    df_features['volume_ratio'] = df_features['volume'] / df_features['volume_sma'].replace(0, 1)
                
                # Lag features
                for lag in [1, 2, 3]:
                    df_features[f'close_lag_{lag}'] = df_features['close'].shift(lag)
                    df_features[f'returns_lag_{lag}'] = df_features['returns'].shift(lag)
                
                # Target variables for ML
                df_features['target_next_close'] = df_features['close'].shift(-1)
                if 'target_next_close' in df_features.columns:
                    df_features['target_direction'] = (df_features['target_next_close'] > df_features['close']).astype(int)
            
            # Drop NaN values
            df_features = df_features.dropna()
            
            # If we lost all data, return original
            if len(df_features) == 0:
                st.warning("Feature engineering resulted in empty dataframe. Using original data.")
                return df
            
            return df_features
            
        except Exception as e:
            st.warning(f"Feature engineering had issues: {str(e)}. Using original data.")
            return df

class MLPipeline:
    def __init__(self):
        self.results = {}
        self.feature_importance = None
    
    def prepare_data(self, df_features):
        """Prepare data for ML training"""
        try:
            if df_features is None or len(df_features) < 20:
                st.warning("Insufficient data for machine learning. Need at least 20 data points.")
                return None, None, None, None, None, None, []
            
            # Select features (exclude date and target columns)
            exclude_cols = ['date', 'target_next_close', 'target_direction']
            feature_cols = [col for col in df_features.columns 
                           if col not in exclude_cols and df_features[col].dtype in [np.float64, np.int64]]
            
            if not feature_cols:
                feature_cols = [col for col in df_features.columns if df_features[col].dtype in [np.float64, np.int64]]
            
            if len(feature_cols) == 0:
                st.warning("No numeric features found for ML training.")
                return None, None, None, None, None, None, []
            
            X = df_features[feature_cols].fillna(0)
            
            # Check if we have target columns
            if 'target_next_close' not in df_features.columns:
                st.warning("Target column not found. Creating from close prices.")
                df_features['target_next_close'] = df_features['close'].shift(-1)
                df_features['target_direction'] = (df_features['target_next_close'] > df_features['close']).astype(int)
            
            y_reg = df_features['target_next_close'].fillna(0).values
            y_clf = df_features['target_direction'].fillna(0).values
            
            # Ensure we have valid targets
            if len(y_reg) == 0 or len(y_clf) == 0:
                st.warning("No valid target values found.")
                return None, None, None, None, None, None, []
            
            # Train-test split (time-series aware) - use at least 10 points for testing
            min_test_size = 10
            if len(X) < min_test_size * 2:
                split_idx = len(X) // 2
            else:
                split_idx = int(0.8 * len(X))
            
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train_reg, y_test_reg = y_reg[:split_idx], y_reg[split_idx:]
            y_train_clf, y_test_clf = y_clf[:split_idx], y_clf[split_idx:]
            
            # Ensure we have test data
            if len(X_test) == 0 or len(y_test_reg) == 0:
                st.warning("Not enough data for testing.")
                return None, None, None, None, None, None, []
            
            return X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf, feature_cols
            
        except Exception as e:
            st.error(f"Data preparation error: {str(e)}")
            return None, None, None, None, None, None, []
    
    def train_all_models(self, X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf):
        """Train all ML models"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
            
            # Check if we have data
            if X_train is None or len(X_train) == 0:
                st.error("No training data available.")
                return False
            
            # Linear Regression
            try:
                lr_model = LinearRegression()
                lr_model.fit(X_train, y_train_reg)
                y_pred_lr = lr_model.predict(X_test)
                
                # Calculate metrics safely
                if len(y_test_reg) > 0 and len(y_pred_lr) > 0:
                    self.results['linear_regression'] = {
                        'metrics': {
                            'rmse': np.sqrt(mean_squared_error(y_test_reg, y_pred_lr)),
                            'mae': mean_absolute_error(y_test_reg, y_pred_lr),
                            'r2': r2_score(y_test_reg, y_pred_lr)
                        },
                        'predictions': y_pred_lr,
                        'actual': y_test_reg
                    }
            except Exception as e:
                st.warning(f"Linear Regression failed: {str(e)}")
            
            # KNN Classifier
            try:
                knn_model = KNeighborsClassifier(n_neighbors=min(5, len(X_train)))
                knn_model.fit(X_train, y_train_clf)
                y_pred_knn = knn_model.predict(X_test)
                
                if len(y_test_clf) > 0 and len(y_pred_knn) > 0:
                    self.results['knn'] = {
                        'metrics': {
                            'accuracy': accuracy_score(y_test_clf, y_pred_knn),
                            'f1': f1_score(y_test_clf, y_pred_knn)
                        },
                        'predictions': y_pred_knn,
                        'actual': y_test_clf
                    }
            except Exception as e:
                st.warning(f"KNN Classifier failed: {str(e)}")
            
            # Gradient Boosted Trees
            try:
                n_estimators = min(100, len(X_train) // 2)
                gb_model = GradientBoostingRegressor(
                    n_estimators=n_estimators, 
                    learning_rate=0.1, 
                    max_depth=3, 
                    random_state=42
                )
                gb_model.fit(X_train, y_train_reg)
                y_pred_gb = gb_model.predict(X_test)
                
                if len(y_test_reg) > 0 and len(y_pred_gb) > 0:
                    self.results['gradient_boosted'] = {
                        'metrics': {
                            'rmse': np.sqrt(mean_squared_error(y_test_reg, y_pred_gb)),
                            'mae': mean_absolute_error(y_test_reg, y_pred_gb),
                            'r2': r2_score(y_test_reg, y_pred_gb)
                        },
                        'predictions': y_pred_gb,
                        'actual': y_test_reg
                    }
                    
                    # Feature importance
                    if hasattr(gb_model, 'feature_importances_'):
                        self.feature_importance = pd.DataFrame({
                            'feature': X_train.columns,
                            'importance': gb_model.feature_importances_
                        }).sort_values('importance', ascending=False)
            except Exception as e:
                st.warning(f"Gradient Boosted Trees failed: {str(e)}")
            
            return len(self.results) > 0
            
        except Exception as e:
            st.error(f"Model training error: {str(e)}")
            return False
    
    def get_results(self):
        return self.results
    
    def get_feature_importance(self):
        return self.feature_importance

class AnomalyDetector:
    def detect(self, df_features):
        """Detect anomalies in data"""
        try:
            from sklearn.ensemble import IsolationForest
            
            if df_features is None or len(df_features) < 10:
                st.warning("Insufficient data for anomaly detection.")
                df_features = df_features.copy()
                df_features['is_anomaly'] = 0
                df_features['anomaly_score'] = 0
                df_features['anomaly_confidence'] = 0
                df_features['anomaly_type'] = 'Normal'
                return df_features
            
            # Select features for anomaly detection
            feature_cols = ['close', 'returns', 'volatility_20']
            available_features = [f for f in feature_cols if f in df_features.columns]
            
            if len(available_features) == 0:
                available_features = df_features.select_dtypes(include=[np.number]).columns.tolist()[:3]
            
            if len(available_features) == 0:
                st.warning("No features available for anomaly detection.")
                df_features = df_features.copy()
                df_features['is_anomaly'] = 0
                df_features['anomaly_score'] = 0
                df_features['anomaly_confidence'] = 0
                df_features['anomaly_type'] = 'Normal'
                return df_features
            
            X = df_features[available_features].fillna(0).values
            
            # Fit and predict
            contamination = min(0.1, 10/len(X))  # Adjust contamination based on data size
            model = IsolationForest(
                contamination=contamination, 
                random_state=42,
                n_estimators=min(100, len(X))
            )
            predictions = model.fit_predict(X)
            scores = model.decision_function(X)
            
            # Add anomaly information to dataframe
            df_anomalies = df_features.copy()
            df_anomalies['is_anomaly'] = (predictions == -1).astype(int)
            df_anomalies['anomaly_score'] = scores
            
            # Calculate confidence safely
            anomaly_scores_abs = np.abs(scores)
            if np.max(anomaly_scores_abs) > 0:
                df_anomalies['anomaly_confidence'] = anomaly_scores_abs / np.max(anomaly_scores_abs)
            else:
                df_anomalies['anomaly_confidence'] = 0
            
            # Classify anomaly types
            anomaly_types = []
            for idx, row in df_anomalies.iterrows():
                if row['is_anomaly'] == 1:
                    if 'returns' in df_anomalies.columns and abs(row.get('returns', 0)) > 0.05:
                        anomaly_types.append('Extreme Return')
                    elif 'volume_ratio' in df_anomalies.columns and row.get('volume_ratio', 1) > 2.5:
                        anomaly_types.append('Volume Spike')
                    elif 'rsi' in df_anomalies.columns and row.get('rsi', 50) > 80:
                        anomaly_types.append('Overbought')
                    elif 'rsi' in df_anomalies.columns and row.get('rsi', 50) < 20:
                        anomaly_types.append('Oversold')
                    else:
                        anomaly_types.append('General Anomaly')
                else:
                    anomaly_types.append('Normal')
            
            df_anomalies['anomaly_type'] = anomaly_types
            
            return df_anomalies
            
        except Exception as e:
            st.error(f"Anomaly detection error: {str(e)}")
            df_features = df_features.copy()
            df_features['is_anomaly'] = 0
            df_features['anomaly_score'] = 0
            df_features['anomaly_confidence'] = 0
            df_features['anomaly_type'] = 'Normal'
            return df_features

class InsightsGenerator:
    def generate_insights(self, df_features, model_results=None):
        """Generate AI insights from data"""
        insights = []
        
        try:
            if df_features is not None and 'close' in df_features.columns and len(df_features) > 0:
                # Price insights
                try:
                    first_price = df_features['close'].iloc[0]
                    last_price = df_features['close'].iloc[-1]
                    
                    if first_price != 0 and not np.isnan(first_price) and not np.isnan(last_price):
                        price_change = ((last_price - first_price) / first_price) * 100
                        
                        if price_change > 10:
                            insights.append(f"Strong bullish trend with {price_change:.1f}% total gain over the period.")
                        elif price_change < -10:
                            insights.append(f"Bearish pressure with {abs(price_change):.1f}% total decline over the period.")
                        else:
                            insights.append(f"Relative stability with {price_change:.1f}% net change over the period.")
                except:
                    insights.append("Price data available for analysis.")
                
                # Recent performance
                if len(df_features) >= 5:
                    try:
                        recent_start = df_features['close'].iloc[-5]
                        recent_end = df_features['close'].iloc[-1]
                        if recent_start != 0:
                            recent_change = ((recent_end - recent_start) / recent_start) * 100
                            if recent_change > 3:
                                insights.append(f"Positive momentum in the last 5 days with {recent_change:.1f}% gain.")
                            elif recent_change < -3:
                                insights.append(f"Negative pressure in the last 5 days with {abs(recent_change):.1f}% decline.")
                    except:
                        pass
                
                # Volatility insights
                if 'volatility_20' in df_features.columns:
                    try:
                        current_vol = df_features['volatility_20'].iloc[-1]
                        if not np.isnan(current_vol):
                            insights.append(f"Current volatility is {current_vol:.4f}.")
                    except:
                        pass
                
                # RSI insights
                if 'rsi' in df_features.columns:
                    try:
                        current_rsi = df_features['rsi'].iloc[-1]
                        if not np.isnan(current_rsi):
                            if current_rsi > 70:
                                insights.append(f"RSI at {current_rsi:.1f} indicates overbought conditions.")
                            elif current_rsi < 30:
                                insights.append(f"RSI at {current_rsi:.1f} indicates oversold conditions.")
                    except:
                        pass
                
                # Volume insights
                if 'volume' in df_features.columns:
                    try:
                        if len(df_features) >= 5:
                            recent_volume = df_features['volume'].iloc[-5:].mean()
                            avg_volume = df_features['volume'].mean()
                            if avg_volume > 0:
                                volume_ratio = recent_volume / avg_volume
                                if volume_ratio > 1.5:
                                    insights.append("Recent trading volume is significantly higher than average.")
                                elif volume_ratio < 0.7:
                                    insights.append("Recent trading activity is lower than average.")
                    except:
                        pass
            
            if model_results:
                # Model performance insights
                best_model = None
                best_score = -np.inf
                
                for model_name, result in model_results.items():
                    metrics = result['metrics']
                    if 'r2' in metrics and not np.isnan(metrics['r2']):
                        if metrics['r2'] > best_score:
                            best_score = metrics['r2']
                            best_model = model_name
                    elif 'accuracy' in metrics and not np.isnan(metrics['accuracy']):
                        if metrics['accuracy'] > best_score:
                            best_score = metrics['accuracy']
                            best_model = model_name
                
                if best_model:
                    model_name_display = best_model.replace('_', ' ').title()
                    insights.append(f"{model_name_display} achieved the best predictive performance.")
        
        except Exception as e:
            insights.append("Generating insights from available data.")
        
        # Ensure we have at least some insights
        if not insights:
            insights.append("Analyzing the uploaded financial data.")
            insights.append("Review technical indicators for trading signals.")
            insights.append("Consider market trends and volume patterns.")
        
        return insights[:5]  # Return top 5 insights

class FinancialAnalyticsDashboard:
    def __init__(self):
        """Initialize dashboard components"""
        self.processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.ml_pipeline = MLPipeline()
        self.anomaly_detector = AnomalyDetector()
        self.insights_gen = InsightsGenerator()
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'df_raw' not in st.session_state:
            st.session_state.df_raw = None
        if 'df_processed' not in st.session_state:
            st.session_state.df_processed = None
        if 'df_features' not in st.session_state:
            st.session_state.df_features = None
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        if 'anomalies_detected' not in st.session_state:
            st.session_state.anomalies_detected = False
        if 'insights' not in st.session_state:
            st.session_state.insights = []
        if 'anomalies' not in st.session_state:
            st.session_state.anomalies = None
    
    def render_sidebar(self):
        """Render sidebar with upload and controls"""
        with st.sidebar:
            # Logo/Title
            st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h3 style='margin: 0.5rem 0; color: #0B3D91;'>Financial Analytics</h3>
                <p style='color: #666; font-size: 0.9rem;'>Enterprise Prediction Platform</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Upload Section
            st.markdown("**Data Upload**")
            
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="Upload time series data with Date and Price columns",
                key="file_uploader"
            )
            
            if uploaded_file is not None:
                if st.button("📊 Process Data", type="primary", use_container_width=True):
                    with st.spinner("Processing data..."):
                        try:
                            # Load data
                            st.session_state.df_raw = pd.read_csv(uploaded_file)
                            
                            # Process data
                            st.session_state.df_processed = self.processor.process(st.session_state.df_raw)
                            
                            if st.session_state.df_processed is not None and len(st.session_state.df_processed) > 0:
                                # Engineer features
                                st.session_state.df_features = self.feature_engineer.create_features(
                                    st.session_state.df_processed
                                )
                                
                                # Generate initial insights
                                st.session_state.insights = self.insights_gen.generate_insights(st.session_state.df_features)
                                
                                st.session_state.data_loaded = True
                                st.success(f"✅ Processed {len(st.session_state.df_processed)} rows successfully")
                                st.rerun()
                            else:
                                st.error("❌ No valid data found after processing")
                                
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
            st.markdown("---")
            
            # Pipeline Status
            if st.session_state.data_loaded:
                st.markdown("**Pipeline Status**")
                
                status_items = [
                    ("Data Loaded", st.session_state.data_loaded),
                    ("Data Processed", st.session_state.df_processed is not None),
                    ("Features Engineered", st.session_state.df_features is not None),
                    ("Models Trained", st.session_state.models_trained),
                    ("Anomalies Detected", st.session_state.anomalies_detected)
                ]
                
                for item, status in status_items:
                    indicator = "✅" if status else "◯"
                    color = "green" if status else "gray"
                    st.markdown(f'<span style="color: {color}; font-weight: bold;">{indicator}</span> {item}', 
                              unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Action Buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🤖 Train Models", use_container_width=True, 
                               disabled=st.session_state.df_features is None or len(st.session_state.df_features) < 20):
                        self.train_models()
                
                with col2:
                    if st.button("🔍 Detect Anomalies", use_container_width=True,
                               disabled=st.session_state.df_features is None):
                        self.detect_anomalies()
            
            st.markdown("---")
            
            # Info
            st.markdown("""
            <div style='font-size: 0.8rem; color: #666;'>
            <p><strong>📋 Supported Format:</strong> CSV</p>
            <p><strong>📊 Required Columns:</strong> Date, Price/Close</p>
            <p><strong>⚙️ Optional Columns:</strong> Open, High, Low, Volume</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sidebar help
            st.markdown("---")
            st.markdown("""
            <div style='font-size: 0.8rem; color: #666;'>
            <p><strong>💡 Tips:</strong></p>
            <p>• Click <strong>></strong> to collapse sidebar</p>
            <p>• Click <strong>☰</strong> to expand sidebar</p>
            <p>• Need at least 20 rows for ML training</p>
            </div>
            """, unsafe_allow_html=True)
    
    def train_models(self):
        """Train machine learning models"""
        with st.spinner("Training ML models..."):
            try:
                if st.session_state.df_features is not None:
                    # Prepare data for ML
                    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf, feature_names = self.ml_pipeline.prepare_data(
                        st.session_state.df_features
                    )
                    
                    if X_train is not None:
                        # Train all models
                        success = self.ml_pipeline.train_all_models(X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf)
                        
                        if success:
                            st.session_state.models_trained = True
                            st.session_state.insights = self.insights_gen.generate_insights(
                                st.session_state.df_features,
                                self.ml_pipeline.get_results()
                            )
                            st.success("✅ Models trained successfully")
                            st.rerun()
                        else:
                            st.warning("⚠️ Model training completed with some issues")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    def detect_anomalies(self):
        """Detect anomalies in the data"""
        with st.spinner("Detecting anomalies..."):
            try:
                if st.session_state.df_features is not None:
                    st.session_state.anomalies = self.anomaly_detector.detect(
                        st.session_state.df_features
                    )
                    st.session_state.anomalies_detected = True
                    st.success("✅ Anomalies detected successfully")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    def render_upload_screen(self):
        """Render clean upload interface"""
        col1, col2 = st.columns([1, 3])
        
        with col2:
            st.markdown('<div class="upload-area">', unsafe_allow_html=True)
            
            st.markdown("""
            <div style='margin-bottom: 2rem;'>
                <h2 style='color: #0B3D91;'>Financial Analytics Platform</h2>
                <p style='color: #666;'>
                    Upload your financial time series data to generate predictions, 
                    detect anomalies, and gain AI-driven insights.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "📁 Drag and drop your CSV file here or click to browse",
                type=['csv'],
                key="main_uploader",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                st.info(f"📄 File uploaded: {uploaded_file.name}")
                
                if st.button("🚀 Begin Analysis", type="primary", use_container_width=True):
                    with st.spinner("Loading and processing data..."):
                        try:
                            st.session_state.df_raw = pd.read_csv(uploaded_file)
                            
                            # Process immediately
                            st.session_state.df_processed = self.processor.process(st.session_state.df_raw)
                            
                            if st.session_state.df_processed is not None and len(st.session_state.df_processed) > 0:
                                st.session_state.df_features = self.feature_engineer.create_features(
                                    st.session_state.df_processed
                                )
                                st.session_state.insights = self.insights_gen.generate_insights(st.session_state.df_features)
                                st.session_state.data_loaded = True
                                st.rerun()
                            else:
                                st.error("❌ No valid data found after processing")
                                
                        except Exception as e:
                            st.error(f"❌ Error reading file: {str(e)}")
            
            st.markdown("""
            <div style='margin-top: 2rem; color: #666; font-size: 0.9rem;'>
                <p><strong>📋 Example CSV format:</strong></p>
                <pre style='background: #f8f9fa; padding: 1rem; border-radius: 5px;'>
Date,Open,High,Low,Close,Volume
2023-01-01,100.0,102.5,99.5,101.2,1000000
2023-01-02,101.2,103.0,100.5,102.8,1200000
2023-01-03,102.8,104.5,102.0,103.5,1150000</pre>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_dashboard(self):
        """Render main dashboard with tabs"""
        # Main header
        st.markdown('<h1 class="dashboard-title">Financial Analytics Dashboard</h1>', unsafe_allow_html=True)
        
        # Summary metrics
        self.render_summary_metrics()
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🤖 Predictions", 
            "📈 Analysis", 
            "🔍 Anomalies",
            "💾 Export"
        ])
        
        with tab1:
            self.render_overview_tab()
        
        with tab2:
            self.render_predictions_tab()
        
        with tab3:
            self.render_analysis_tab()
        
        with tab4:
            self.render_anomalies_tab()
        
        with tab5:
            self.render_export_tab()
    
    def render_summary_metrics(self):
        """Render summary metrics cards - FIXED DIVISION BY ZERO"""
        if st.session_state.df_processed is not None:
            df = st.session_state.df_processed
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📊 Data Points", f"{len(df):,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if 'close' in df.columns and len(df) > 0:
                    try:
                        current = float(df['close'].iloc[-1])
                        first = float(df['close'].iloc[0])
                        
                        # FIXED: Check for division by zero
                        if first != 0 and not np.isnan(first) and not np.isnan(current):
                            change = ((current - first) / first) * 100
                            st.metric("📈 Total Return", f"{change:.1f}%", f"${current:.2f}")
                        else:
                            st.metric("📈 Total Return", "N/A", f"${current:.2f}")
                    except:
                        st.metric("📈 Total Return", "N/A", "N/A")
                else:
                    st.metric("📈 Total Return", "N/A", "N/A")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if 'volume' in df.columns and len(df) > 0:
                    try:
                        avg_volume = float(df['volume'].mean())
                        st.metric("📦 Avg Volume", f"{avg_volume:,.0f}")
                    except:
                        st.metric("📦 Avg Volume", "N/A")
                else:
                    st.metric("📦 Avg Volume", "N/A")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if 'date' in df.columns and len(df) > 0:
                    try:
                        start = df['date'].min()
                        end = df['date'].max()
                        if hasattr(end, 'day') and hasattr(start, 'day'):  # Check if they're datetime objects
                            days = (end - start).days
                            st.metric("📅 Period", f"{days} days")
                        else:
                            st.metric("📅 Period", f"{len(df)} points")
                    except:
                        st.metric("📅 Period", f"{len(df)} points")
                else:
                    st.metric("📅 Period", "N/A")
                st.markdown('</div>', unsafe_allow_html=True)
    
    def render_overview_tab(self):
        """Render overview tab"""
        if st.session_state.df_processed is None:
            st.info("📝 Please upload and process data first")
            return
        
        df = st.session_state.df_processed
        
        # Price Chart
        st.markdown('<h3 class="section-title">📈 Price Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if 'close' in df.columns and len(df) > 0:
                fig = go.Figure()
                
                # Try candlestick if we have OHLC data
                if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    fig.add_trace(go.Candlestick(
                        x=df['date'] if 'date' in df.columns else df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='Price'
                    ))
                else:
                    # Simple line chart
                    fig.add_trace(go.Scatter(
                        x=df['date'] if 'date' in df.columns else df.index,
                        y=df['close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(width=2, color='#0B3D91')
                    ))
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Price",
                    hovermode='x unified',
                    template='plotly_white',
                    showlegend=True,
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown("**📊 Quick Stats**")
            
            if 'close' in df.columns and len(df) > 1:
                try:
                    returns = df['close'].pct_change().dropna()
                    if len(returns) > 0:
                        avg_return = returns.mean() * 100
                        volatility = returns.std() * 100
                        st.markdown(f"**Avg Daily Return:** {avg_return:.2f}%")
                        st.markdown(f"**Volatility:** {volatility:.2f}%")
                        if volatility > 0:
                            sharpe = avg_return / volatility
                            st.markdown(f"**Sharpe Ratio:** {sharpe:.2f}")
                except:
                    st.markdown("**Stats:** Calculating...")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Volume Chart
        if 'volume' in df.columns and len(df) > 0:
            st.markdown('<h3 class="section-title">📦 Volume Analysis</h3>', unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df['volume'],
                name='Volume',
                marker_color='#0B3D91'
            ))
            
            fig.update_layout(
                height=300,
                xaxis_title="Date",
                yaxis_title="Volume",
                hovermode='x unified',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # AI Insights
        if st.session_state.insights:
            st.markdown('<h3 class="section-title">💡 AI Insights</h3>', unsafe_allow_html=True)
            
            for insight in st.session_state.insights[:3]:
                st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
    
    def render_predictions_tab(self):
        """Render predictions tab"""
        if not st.session_state.models_trained:
            st.info("🤖 Please train models first using the sidebar")
            return
        
        results = self.ml_pipeline.get_results()
        
        if not results:
            st.info("📊 No model results available. Please train models again.")
            return
        
        st.markdown('<h3 class="section-title">🤖 Model Performance</h3>', unsafe_allow_html=True)
        
        # Model comparison table
        perf_data = []
        for model_name, result in results.items():
            metrics = result['metrics']
            row = {'Model': model_name.replace('_', ' ').title()}
            
            for metric_name, metric_value in metrics.items():
                if metric_name in ['rmse', 'mae', 'r2', 'accuracy', 'f1']:
                    row[metric_name.upper()] = f"{metric_value:.4f}"
            
            perf_data.append(row)
        
        if perf_data:
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True)
        
        # Prediction visualization
        st.markdown('<h3 class="section-title">📊 Predictions vs Actual</h3>', unsafe_allow_html=True)
        
        model_options = list(results.keys())
        if model_options:
            selected_model = st.selectbox(
                "Select Model to Visualize",
                model_options,
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            if selected_model in results:
                result = results[selected_model]
                
                fig = go.Figure()
                
                # Plot actual vs predicted
                actual_values = result['actual']
                predicted_values = result['predictions']
                
                # Take last 50 points or all if less
                n_points = min(50, len(actual_values))
                
                fig.add_trace(go.Scatter(
                    y=actual_values[-n_points:],
                    mode='lines',
                    name='Actual',
                    line=dict(width=2, color='#0B3D91')
                ))
                
                fig.add_trace(go.Scatter(
                    y=predicted_values[-n_points:],
                    mode='lines',
                    name='Predicted',
                    line=dict(dash='dash', width=2, color='#FFD700')
                ))
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Time Step",
                    yaxis_title="Value",
                    hovermode='x unified',
                    template='plotly_white',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_analysis_tab(self):
        """Render analysis tab"""
        if st.session_state.df_features is None:
            st.info("📝 Please process data first")
            return
        
        df = st.session_state.df_features
        
        st.markdown('<h3 class="section-title">📈 Technical Indicators</h3>', unsafe_allow_html=True)
        
        # Technical indicators visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI
            if 'rsi' in df.columns and len(df) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['date'] if 'date' in df.columns else df.index,
                    y=df['rsi'],
                    mode='lines',
                    name='RSI',
                    line=dict(width=2, color='#0B3D91')
                ))
                
                # Overbought/oversold lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                
                fig.update_layout(
                    height=300,
                    xaxis_title="Date",
                    yaxis_title="RSI",
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 RSI data not available")
        
        with col2:
            # MACD
            if all(col in df.columns for col in ['macd', 'macd_signal']) and len(df) > 0:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df['date'] if 'date' in df.columns else df.index,
                    y=df['macd'],
                    mode='lines',
                    name='MACD',
                    line=dict(width=2, color='#0B3D91')
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['date'] if 'date' in df.columns else df.index,
                    y=df['macd_signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(width=2, color='#FFD700')
                ))
                
                fig.update_layout(
                    height=300,
                    xaxis_title="Date",
                    yaxis_title="MACD",
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 MACD data not available")
        
        # Feature Importance
        if st.session_state.models_trained:
            st.markdown('<h3 class="section-title">⚖️ Feature Importance</h3>', unsafe_allow_html=True)
            
            importance = self.ml_pipeline.get_feature_importance()
            if importance is not None and len(importance) > 0:
                fig = go.Figure()
                
                # Get top 10 features
                top_features = importance.head(10)
                
                fig.add_trace(go.Bar(
                    x=top_features['importance'],
                    y=top_features['feature'],
                    orientation='h',
                    marker_color='#0B3D91'
                ))
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Feature importance not available")
    
    def render_anomalies_tab(self):
        """Render anomalies tab"""
        if not st.session_state.anomalies_detected:
            st.info("🔍 Please detect anomalies first using the sidebar")
            return
        
        if st.session_state.anomalies is None:
            st.error("❌ Anomalies not detected. Please run detection first.")
            return
        
        df_anomalies = st.session_state.anomalies
        
        st.markdown('<h3 class="section-title">🔍 Anomaly Detection Results</h3>', unsafe_allow_html=True)
        
        # Anomaly summary
        if 'is_anomaly' in df_anomalies.columns:
            anomaly_count = df_anomalies['is_anomaly'].sum()
            total_count = len(df_anomalies)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🚨 Anomalies Detected", f"{anomaly_count}")
            
            with col2:
                if total_count > 0:
                    anomaly_rate = (anomaly_count / total_count) * 100
                    st.metric("📊 Anomaly Rate", f"{anomaly_rate:.1f}%")
                else:
                    st.metric("📊 Anomaly Rate", "N/A")
            
            with col3:
                if 'anomaly_confidence' in df_anomalies.columns and len(df_anomalies) > 0:
                    try:
                        avg_confidence = df_anomalies['anomaly_confidence'].mean()
                        st.metric("🎯 Avg Confidence", f"{avg_confidence:.1%}")
                    except:
                        st.metric("🎯 Avg Confidence", "N/A")
                else:
                    st.metric("🎯 Avg Confidence", "N/A")
        
        # Anomaly visualization
        st.markdown('<h4 style="color: #0B3D91; margin: 1rem 0;">📈 Anomaly Timeline</h4>', unsafe_allow_html=True)
        
        if 'close' in df_anomalies.columns and len(df_anomalies) > 0:
            fig = go.Figure()
            
            # Plot price
            fig.add_trace(go.Scatter(
                x=df_anomalies['date'] if 'date' in df_anomalies.columns else df_anomalies.index,
                y=df_anomalies['close'],
                mode='lines',
                name='Price',
                line=dict(width=1, color='lightgray')
            ))
            
            # Highlight anomalies
            if 'is_anomaly' in df_anomalies.columns:
                anomalies = df_anomalies[df_anomalies['is_anomaly'] == 1]
                if len(anomalies) > 0:
                    fig.add_trace(go.Scatter(
                        x=anomalies['date'] if 'date' in anomalies.columns else anomalies.index,
                        y=anomalies['close'],
                        mode='markers',
                        name='Anomaly',
                        marker=dict(
                            size=10,
                            color='red',
                            symbol='x'
                        )
                    ))
            
            fig.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly details
        st.markdown('<h4 style="color: #0B3D91; margin: 1rem 0;">📋 Anomaly Details</h4>', unsafe_allow_html=True)
        
        if 'is_anomaly' in df_anomalies.columns:
            anomalies = df_anomalies[df_anomalies['is_anomaly'] == 1]
            if len(anomalies) > 0:
                # Select columns to display
                display_cols = []
                for col in ['date', 'close', 'volume', 'returns', 'anomaly_confidence', 'anomaly_type']:
                    if col in anomalies.columns:
                        display_cols.append(col)
                
                if display_cols:
                    st.dataframe(
                        anomalies[display_cols].head(20),
                        use_container_width=True
                    )
            else:
                st.info("✅ No anomalies detected in the data")
    
    def render_export_tab(self):
        """Render export tab"""
        st.markdown('<h3 class="section-title">💾 Export Data</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📁 Available Datasets")
            
            datasets = []
            if st.session_state.df_processed is not None:
                datasets.append(("📊 Processed Data", st.session_state.df_processed))
            if st.session_state.df_features is not None:
                datasets.append(("⚙️ Features Data", st.session_state.df_features))
            if st.session_state.anomalies is not None:
                datasets.append(("🔍 Anomaly Data", st.session_state.anomalies))
            
            for name, df in datasets:
                with st.expander(f"{name} ({len(df)} rows)"):
                    st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.markdown("### ⚙️ Export Options")
            
            export_format = st.selectbox(
                "Select Format",
                ["CSV", "Excel"],
                key="export_format"
            )
            
            if datasets:
                selected_dataset = st.selectbox(
                    "Select Dataset to Export",
                    [name for name, _ in datasets],
                    key="export_dataset"
                )
                
                if st.button("📥 Download Data", use_container_width=True, type="primary"):
                    # Get selected dataframe
                    df_to_export = None
                    for name, df in datasets:
                        if name == selected_dataset:
                            df_to_export = df
                            break
                    
                    if df_to_export is not None:
                        if export_format == "CSV":
                            csv = df_to_export.to_csv(index=False)
                            st.download_button(
                                label="⬇️ Download CSV",
                                data=csv,
                                file_name=f"{selected_dataset.lower().replace(' ', '_').replace('📊', '').replace('⚙️', '').replace('🔍', '')}.csv",
                                mime="text/csv",
                                key="download_csv"
                            )
                        
                        elif export_format == "Excel":
                            # For Excel, we need to use BytesIO
                            import io
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                df_to_export.to_excel(writer, index=False, sheet_name='Data')
                            excel_data = output.getvalue()
                            
                            st.download_button(
                                label="⬇️ Download Excel",
                                data=excel_data,
                                file_name=f"{selected_dataset.lower().replace(' ', '_').replace('📊', '').replace('⚙️', '').replace('🔍', '')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download_excel"
                            )
            else:
                st.info("📝 No data available for export")
    
    def run(self):
        """Main application runner"""
        # Render sidebar
        self.render_sidebar()
        
        # Main content
        if not st.session_state.data_loaded:
            self.render_upload_screen()
        else:
            self.render_dashboard()

# Run the application
if __name__ == "__main__":
    # Initialize and run dashboard
    app = FinancialAnalyticsDashboard()
    app.run()
