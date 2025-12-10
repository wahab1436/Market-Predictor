import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class MLPipeline:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.feature_names = []
    
    def prepare_data(self, df_features):
        """Prepare data for ML training"""
        # Select features (exclude date and target columns)
        exclude_cols = ['date', 'target_next_close', 'target_direction']
        feature_cols = [col for col in df_features.columns 
                       if col not in exclude_cols and df_features[col].dtype in [np.float64, np.int64]]
        
        X = df_features[feature_cols].fillna(0)
        y_reg = df_features['target_next_close'].values
        y_clf = df_features['target_direction'].values
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-test split (time-series aware)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train_reg, y_test_reg = y_reg[:split_idx], y_reg[split_idx:]
        y_train_clf, y_test_clf = y_clf[:split_idx], y_clf[split_idx:]
        
        return X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf, feature_cols
    
    def train_all_models(self, X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf):
        """Train all ML models"""
        # Linear Regression
        self._train_linear_regression(X_train, X_test, y_train_reg, y_test_reg)
        
        # KNN Classifier
        self._train_knn(X_train, X_test, y_train_clf, y_test_clf)
        
        # Gradient Boosted Trees
        self._train_gradient_boosted(X_train, X_test, y_train_reg, y_test_reg)
        
        # Simple Neural Network (simulated with XGBoost for now)
        self._train_xgboost(X_train, X_test, y_train_reg, y_test_reg)
    
    def _train_linear_regression(self, X_train, X_test, y_train, y_test):
        """Train linear regression model"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_r2': r2_score(y_test, y_pred)
        }
        
        self.models['linear_regression'] = model
        self.results['linear_regression'] = {
            'metrics': metrics,
            'predictions': y_pred,
            'actual': y_test
        }
    
    def _train_knn(self, X_train, X_test, y_train, y_test):
        """Train KNN classifier"""
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_f1': f1_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        self.models['knn'] = model
        self.results['knn'] = {
            'metrics': metrics,
            'predictions': y_pred,
            'actual': y_test
        }
    
    def _train_gradient_boosted(self, X_train, X_test, y_train, y_test):
        """Train gradient boosted trees"""
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = {
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_r2': r2_score(y_test, y_pred)
        }
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.models['gradient_boosted'] = model
        self.results['gradient_boosted'] = {
            'metrics': metrics,
            'predictions': y_pred,
            'actual': y_test
        }
    
    def _train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost as neural network proxy"""
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = {
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_r2': r2_score(y_test, y_pred)
        }
        
        self.models['xgboost'] = model
        self.results['xgboost'] = {
            'metrics': metrics,
            'predictions': y_pred,
            'actual': y_test
        }
    
    def get_results(self):
        """Get all model results"""
        return self.results
    
    def get_feature_importance(self):
        """Get feature importance"""
        return self.feature_importance
