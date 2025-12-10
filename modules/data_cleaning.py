import pandas as pd
import numpy as np
from datetime import datetime

class DataCleaning:
    def __init__(self):
        self.df_cleaned = None
        
    def clean_data(self, df):
        """Clean and preprocess the financial data"""
        df_clean = df.copy()
        
        # Standardize column names
        df_clean.columns = [col.strip().replace(' ', '_').replace('.', '_').lower() for col in df_clean.columns]
        
        # Convert date column
        date_col = self._find_date_column(df_clean)
        if date_col:
            df_clean[date_col] = pd.to_datetime(df_clean[date_col])
            df_clean = df_clean.sort_values(date_col).reset_index(drop=True)
        
        # Identify numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # Handle missing values
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                # Forward fill, then backward fill
                df_clean[col] = df_clean[col].ffill().bfill()
                # If still null, use median
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Cap outliers using IQR method
        for col in numeric_cols:
            if col != 'volume' and col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
                df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
        
        self.df_cleaned = df_clean
        return df_clean
    
    def _find_date_column(self, df):
        """Identify date column in dataframe"""
        date_keywords = ['date', 'time', 'datetime', 'timestamp']
        for col in df.columns:
            col_lower = col.lower()
            for keyword in date_keywords:
                if keyword in col_lower:
                    return col
        return None
    
    def get_cleaning_report(self, df_original, df_cleaned):
        """Generate cleaning report"""
        report = {
            'original_shape': df_original.shape,
            'cleaned_shape': df_cleaned.shape,
            'missing_values_filled': df_original.isnull().sum().sum() - df_cleaned.isnull().sum().sum(),
            'date_column_found': self._find_date_column(df_original) is not None
        }
        return report
