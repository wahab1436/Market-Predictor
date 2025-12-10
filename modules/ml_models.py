import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings('ignore')

class MLModels:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
        self.feature_importance = {}
        
    def prepare_data(self, df_features, target_type='regression'):
        """Prepare data for ML models"""
        # Select features
        feature_cols = [col for col in df_features.columns if col not in 
                       ['date', 'next_day_close', 'next_day_return', 'next_day_direction']]
        
        X = df_features[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        
        if target_type == 'regression':
            y = df_features['next_day_close'].values
        elif target_type == 'classification':
            y = df_features['next_day_direction'].values
        else:  # return regression
            y = df_features['next_day_return'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time-series split (last 20% for testing)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test, X.columns.tolist()
    
    def train_linear_regression(self, X_train, X_test, y_train, y_test):
        """Train Linear Regression model"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        # Feature importance (coefficients)
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        self.models['linear_regression'] = model
        self.results['linear_regression'] = {
            'metrics': metrics,
            'predictions': y_pred_test,
            'actual': y_test
        }
        self.feature_importance['linear_regression'] = importance
        
        return metrics, y_pred_test
    
    def train_knn_classifier(self, X_train, X_test, y_train, y_test):
        """Train K-Nearest Neighbors classifier"""
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'train_f1': f1_score(y_train, y_pred_train),
            'test_f1': f1_score(y_test, y_pred_test),
            'test_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.models['knn'] = model
        self.results['knn'] = {
            'metrics': metrics,
            'predictions': y_pred_test,
            'actual': y_test,
            'probabilities': y_pred_proba
        }
        
        return metrics, y_pred_test
    
    class NeuralNet(nn.Module):
        """Simple feedforward neural network"""
        def __init__(self, input_size, hidden_size=64, output_size=1):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc3 = nn.Linear(hidden_size // 2, output_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    def train_neural_network(self, X_train, X_test, y_train, y_test):
        """Train Neural Network for return prediction"""
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
        
        # Create model
        input_size = X_train.shape[1]
        model = self.NeuralNet(input_size=input_size)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        n_epochs = 100
        batch_size = 32
        
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(n_epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Predictions
        with torch.no_grad():
            y_pred_train = model(X_train_tensor).numpy().flatten()
            y_pred_test = model(X_test_tensor).numpy().flatten()
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        self.models['neural_network'] = model
        self.results['neural_network'] = {
            'metrics': metrics,
            'predictions': y_pred_test,
            'actual': y_test
        }
        
        return metrics, y_pred_test
    
    def train_gradient_boosted_trees(self, X_train, X_test, y_train, y_test, model_type='regression'):
        """Train Gradient Boosted Trees"""
        if model_type == 'regression':
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test)
            }
            
            # Feature importance
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
        else:  # classification
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'test_accuracy': accuracy_score(y_test, y_pred_test),
                'train_f1': f1_score(y_train, y_pred_train),
                'test_f1': f1_score(y_test, y_pred_test),
                'test_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Feature importance
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        model_key = 'gradient_boosted_regression' if model_type == 'regression' else 'gradient_boosted_classification'
        self.models[model_key] = model
        self.results[model_key] = {
            'metrics': metrics,
            'predictions': y_pred_test,
            'actual': y_test
        }
        self.feature_importance[model_key] = importance
        
        return metrics, y_pred_test
    
    def calculate_shap_values(self, X_test, model_name='gradient_boosted_regression'):
        """Calculate SHAP values for model explainability"""
        if model_name in self.models:
            model = self.models[model_name]
            
            # Calculate SHAP values
            explainer = shap.Explainer(model, X_test)
            shap_values = explainer(X_test)
            
            return shap_values
        return None
