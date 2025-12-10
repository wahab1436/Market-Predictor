import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        pass
    
    def process(self, df):
        """Process raw dataframe"""
        df_clean = df.copy()
        
        # Standardize column names
        df_clean.columns = [self._standardize_column_name(col) for col in df_clean.columns]
        
        # Identify date column
        date_col = self._identify_date_column(df_clean)
        if date_col:
            df_clean[date_col] = pd.to_datetime(df_clean[date_col])
            df_clean = df_clean.sort_values(date_col).reset_index(drop=True)
        
        # Identify numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # Handle missing values
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                # For time series, forward fill then backward fill
                df_clean[col] = df_clean[col].ffill().bfill()
        
        # Remove any remaining nulls
        df_clean = df_clean.dropna()
        
        return df_clean
    
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
            'qty': 'volume',
            'amount': 'volume'
        }
        
        for key, value in mappings.items():
            if key in col_name:
                return value
        
        # Replace spaces with underscores
        col_name = col_name.replace(' ', '_')
        
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
        first_col = df.columns[0]
        if len(df) > 0:
            sample_value = str(df[first_col].iloc[0])
            
            # Simple date pattern check
            date_patterns = ['-', '/', '202', '201', '200']
            if any(pattern in sample_value for pattern in date_patterns):
                return first_col
        
        return None
