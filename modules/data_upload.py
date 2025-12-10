import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import io

class DataUpload:
    def __init__(self):
        self.df = None
        self.uploaded_file = None
        
    def upload_csv(self):
        """Handle CSV file upload"""
        st.sidebar.header("Data Upload")
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload financial time series data with columns: Date, Open, High, Low, Close, Volume"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV file
                self.df = pd.read_csv(uploaded_file)
                self.uploaded_file = uploaded_file
                
                # Display file info
                st.sidebar.success(f"File uploaded successfully: {uploaded_file.name}")
                st.sidebar.write(f"**Rows:** {self.df.shape[0]}")
                st.sidebar.write(f"**Columns:** {self.df.shape[1]}")
                
                return True, self.df
                
            except Exception as e:
                st.sidebar.error(f"Error reading file: {str(e)}")
                return False, None
        else:
            # Load sample data
            if st.sidebar.checkbox("Use Sample Data", value=True):
                self.df = self._load_sample_data()
                st.sidebar.info("Using sample stock data")
                return True, self.df
                
        return False, None
    
    def _load_sample_data(self):
        """Load sample data if no file uploaded"""
        # Generate sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        base_price = 100
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Add some trends and seasonality
        trend = np.linspace(0, 20, len(dates))
        seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 252)
        
        prices = prices + trend + seasonal
        
        # Create OHLCV data
        data = {
            'Date': dates,
            'Open': prices * 0.998 + np.random.normal(0, 0.5, len(dates)),
            'High': prices * 1.005 + np.random.normal(0, 0.5, len(dates)),
            'Low': prices * 0.995 + np.random.normal(0, 0.5, len(dates)),
            'Close': prices + np.random.normal(0, 0.5, len(dates)),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }
        
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        
        return df
    
    def get_data_info(self, df):
        """Get information about the uploaded data"""
        info = {
            'start_date': df['Date'].min().strftime('%Y-%m-%d'),
            'end_date': df['Date'].max().strftime('%Y-%m-%d'),
            'num_days': len(df),
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict()
        }
        return info
