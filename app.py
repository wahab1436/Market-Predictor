import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for professional, elegant design - NO EMOJIS
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Headers */
    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #0B3D91;
        padding-bottom: 0.5rem;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-color);
        margin: 1.5rem 0 1rem 0;
    }
    
    .subsection-title {
        font-size: 1.2rem;
        font-weight: 500;
        color: var(--text-color);
        margin: 1rem 0 0.5rem 0;
    }
    
    /* Cards */
    .metric-card {
        background: var(--background-color);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #0B3D91;
        margin-bottom: 1rem;
    }
    
    .insight-card {
        background: var(--secondary-background-color);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid var(--border-color);
        transition: transform 0.2s;
    }
    
    .insight-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .anomaly-highlight {
        background: linear-gradient(135deg, #fff3f3 0%, #fff8f8 100%);
        border-left: 4px solid #dc3545;
    }
    
    .success-highlight {
        background: linear-gradient(135deg, #f3fff7 0%, #f8fffb 100%);
        border-left: 4px solid #28a745;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding: 10px 16px;
        background: var(--secondary-background-color);
        border-radius: 6px;
        font-weight: 500;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--background-color);
    }
    
    .stTabs [aria-selected="true"] {
        background: #0B3D91 !important;
        color: white !important;
        border-color: #0B3D91 !important;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed var(--border-color);
        border-radius: 10px;
        padding: 3rem 1rem;
        text-align: center;
        background: var(--secondary-background-color);
        margin: 1rem 0;
        transition: all 0.3s;
    }
    
    .upload-area:hover {
        border-color: #0B3D91;
        background: var(--background-color);
    }
    
    /* Data table */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border-color);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-success {
        background-color: #28a745;
    }
    
    .status-processing {
        background-color: #ffc107;
        animation: pulse 2s infinite;
    }
    
    .status-error {
        background-color: #dc3545;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Import modules
sys.path.append('modules')

# Check if modules exist, if not, create them inline
modules_exist = os.path.exists('modules')
if not modules_exist:
    os.makedirs('modules', exist_ok=True)

# Create necessary module files if they don't exist
module_files = ['data_processing.py', 'feature_engineering.py', 'ml_pipeline.py', 
                'anomaly_detector.py', 'insights_generator.py']

for module in module_files:
    if not os.path.exists(f'modules/{module}'):
        with open(f'modules/{module}', 'w') as f:
            f.write('')

# Now try to import
try:
    from data_processing import DataProcessor
    from feature_engineering import FeatureEngineer
    from ml_pipeline import MLPipeline
    from anomaly_detector import AnomalyDetector
    from insights_generator import InsightsGenerator
except ImportError:
    # Define the classes inline if modules don't exist
    class DataProcessor:
        def process(self, df):
            return df.copy()
    
    class FeatureEngineer:
        def create_features(self, df):
            return df.copy()
    
    class MLPipeline:
        def prepare_data(self, df):
            return None, None, None, None, None, None, []
        
        def train_all_models(self, *args):
            pass
        
        def get_results(self):
            return {}
        
        def get_feature_importance(self):
            return None
    
    class AnomalyDetector:
        def detect(self, df):
            return df.copy()
    
    class InsightsGenerator:
        def generate_insights(self, df, results=None):
            return []

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
    
    def render_sidebar(self):
        """Render elegant sidebar with upload and controls - NO EMOJIS"""
        with st.sidebar:
            # Logo/Title - NO EMOJIS
            st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h3 style='margin: 0.5rem 0; color: #0B3D91;'>Financial Analytics</h3>
                <p style='color: var(--text-color-secondary); font-size: 0.9rem;'>Enterprise Prediction Platform</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Upload Section - NO EMOJIS
            st.markdown("### Data Upload")
            
            uploaded_file = st.file_uploader(
                "Upload your financial CSV file",
                type=['csv'],
                help="Upload time series data with columns like Date, Open, High, Low, Close, Volume",
                key="file_uploader"
            )
            
            if uploaded_file is not None:
                if st.button("Process Data", type="primary", use_container_width=True):
                    with st.spinner("Processing data..."):
                        try:
                            # Load data
                            st.session_state.df_raw = pd.read_csv(uploaded_file)
                            st.session_state.data_loaded = True
                            
                            # Process data
                            st.session_state.df_processed = self.processor.process(st.session_state.df_raw)
                            
                            # Engineer features
                            st.session_state.df_features = self.feature_engineer.create_features(
                                st.session_state.df_processed
                            )
                            
                            st.success("Data processed successfully")
                            
                        except Exception as e:
                            st.error(f"Error processing data: {str(e)}")
            
            st.markdown("---")
            
            # Pipeline Status - NO EMOJIS
            if st.session_state.data_loaded:
                st.markdown("### Pipeline Status")
                
                status_items = [
                    ("Data Loaded", st.session_state.data_loaded),
                    ("Data Processed", st.session_state.df_processed is not None),
                    ("Features Engineered", st.session_state.df_features is not None),
                    ("Models Trained", st.session_state.models_trained),
                    ("Anomalies Detected", st.session_state.anomalies_detected)
                ]
                
                for item, status in status_items:
                    indicator = "●" if status else "○"
                    color = "#28a745" if status else "#6c757d"
                    st.markdown(f'<span style="color: {color}; font-weight: bold;">{indicator}</span> {item}', 
                              unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Action Buttons - NO EMOJIS
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Train Models", use_container_width=True, 
                               disabled=st.session_state.df_features is None):
                        self.train_models()
                
                with col2:
                    if st.button("Detect Anomalies", use_container_width=True,
                               disabled=st.session_state.df_features is None):
                        self.detect_anomalies()
            
            st.markdown("---")
            
            # Info - NO EMOJIS
            st.markdown("""
            <div style='font-size: 0.8rem; color: var(--text-color-secondary);'>
            <p><strong>Supported Format:</strong> CSV</p>
            <p><strong>Required Columns:</strong> Date, Open, High, Low, Close, Volume</p>
            <p><strong>Max Size:</strong> 200,000 rows</p>
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
                    
                    # Train all models
                    self.ml_pipeline.train_all_models(X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf)
                    st.session_state.models_trained = True
                    st.success("Models trained successfully")
                    
                    # Generate insights
                    self.insights = self.insights_gen.generate_insights(
                        st.session_state.df_features,
                        self.ml_pipeline.get_results()
                    )
                    
            except Exception as e:
                st.error(f"Error training models: {str(e)}")
    
    def detect_anomalies(self):
        """Detect anomalies in the data"""
        with st.spinner("Detecting anomalies..."):
            try:
                if st.session_state.df_features is not None:
                    self.anomalies = self.anomaly_detector.detect(
                        st.session_state.df_features
                    )
                    st.session_state.anomalies_detected = True
                    st.success("Anomalies detected successfully")
            except Exception as e:
                st.error(f"Error detecting anomalies: {str(e)}")
    
    def render_upload_screen(self):
        """Render clean upload interface - NO EMOJIS"""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="upload-area">', unsafe_allow_html=True)
            
            st.markdown("""
            <div style='margin-bottom: 2rem;'>
                <h2 style='color: #0B3D91;'>Financial Analytics Platform</h2>
                <p style='color: var(--text-color-secondary);'>
                    Upload your financial time series data to generate predictions, 
                    detect anomalies, and gain AI-driven insights.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Drag and drop your CSV file here",
                type=['csv'],
                key="main_uploader",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                st.info(f"File: {uploaded_file.name} ready for processing")
                
                if st.button("Begin Analysis", type="primary", use_container_width=True):
                    with st.spinner("Loading and processing data..."):
                        try:
                            st.session_state.df_raw = pd.read_csv(uploaded_file)
                            st.session_state.data_loaded = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error reading file: {str(e)}")
            
            st.markdown("""
            <div style='margin-top: 2rem; color: var(--text-color-secondary); font-size: 0.9rem;'>
                <p><strong>Example CSV format:</strong></p>
                <pre style='background: var(--secondary-background-color); padding: 1rem; border-radius: 5px;'>
Date,Open,High,Low,Close,Volume
2023-01-01,100.0,102.5,99.5,101.2,1000000
2023-01-02,101.2,103.0,100.5,102.8,1200000</pre>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_dashboard(self):
        """Render main dashboard with tabs - NO EMOJIS"""
        # Main header
        st.markdown('<h1 class="dashboard-title">Financial Analytics Dashboard</h1>', unsafe_allow_html=True)
        
        # Summary metrics
        self.render_summary_metrics()
        
        # Tabs - NO EMOJIS
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview", 
            "Predictions", 
            "Analysis", 
            "Anomalies",
            "Export"
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
        """Render summary metrics cards"""
        if st.session_state.df_processed is not None:
            df = st.session_state.df_processed
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Data Points", f"{len(df):,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if 'close' in df.columns:
                    current = df['close'].iloc[-1]
                    first = df['close'].iloc[0]
                    change = ((current - first) / first) * 100
                    st.metric("Total Return", f"{change:.1f}%", f"${current:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if 'volume' in df.columns:
                    avg_volume = df['volume'].mean()
                    st.metric("Avg Volume", f"{avg_volume:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if 'date' in df.columns:
                    start = df['date'].min()
                    end = df['date'].max()
                    days = (end - start).days
                    st.metric("Period", f"{days} days")
                st.markdown('</div>', unsafe_allow_html=True)
    
    def render_overview_tab(self):
        """Render overview tab"""
        if st.session_state.df_processed is None:
            st.info("Please upload and process data first")
            return
        
        df = st.session_state.df_processed
        
        # Price Chart
        st.markdown('<h3 class="section-title">Price Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if 'close' in df.columns and 'date' in df.columns:
                fig = go.Figure()
                
                # Candlestick if we have OHLC data
                if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    fig.add_trace(go.Candlestick(
                        x=df['date'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='Price'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=df['date'],
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
            st.markdown('<div class="insight-card success-highlight">', unsafe_allow_html=True)
            st.markdown("### Quick Stats")
            
            if 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                st.markdown(f"**Avg Daily Return:** {returns.mean()*100:.2f}%")
                st.markdown(f"**Volatility:** {returns.std()*100:.2f}%")
                st.markdown(f"**Sharpe Ratio:** {returns.mean()/returns.std():.2f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Volume Chart
        st.markdown('<h3 class="section-title">Volume Analysis</h3>', unsafe_allow_html=True)
        
        if 'volume' in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['date'],
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
        if hasattr(self, 'insights'):
            st.markdown('<h3 class="section-title">AI Insights</h3>', unsafe_allow_html=True)
            
            for insight in self.insights[:3]:  # Show top 3 insights
                st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
    
    def render_predictions_tab(self):
        """Render predictions tab"""
        if not st.session_state.models_trained:
            st.info("Please train models first using the sidebar")
            return
        
        results = self.ml_pipeline.get_results()
        
        st.markdown('<h3 class="section-title">Model Performance</h3>', unsafe_allow_html=True)
        
        # Model comparison table
        perf_data = []
        for model_name, result in results.items():
            metrics = result['metrics']
            perf_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'RMSE': f"{metrics.get('test_rmse', 0):.4f}",
                'MAE': f"{metrics.get('test_mae', 0):.4f}",
                'R²': f"{metrics.get('test_r2', 0):.4f}",
                'Accuracy': f"{metrics.get('test_accuracy', 0):.4f}"
            })
        
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True)
        
        # Prediction visualization
        st.markdown('<h3 class="section-title">Predictions vs Actual</h3>', unsafe_allow_html=True)
        
        model_options = list(results.keys())
        selected_model = st.selectbox(
            "Select Model to Visualize",
            model_options,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if selected_model in results:
            result = results[selected_model]
            
            fig = go.Figure()
            
            # Plot actual vs predicted
            fig.add_trace(go.Scatter(
                y=result['actual'][-50:],  # Last 50 points
                mode='lines',
                name='Actual',
                line=dict(width=2, color='#0B3D91')
            ))
            
            fig.add_trace(go.Scatter(
                y=result['predictions'][-50:],
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
            st.info("Please process data first")
            return
        
        df = st.session_state.df_features
        
        st.markdown('<h3 class="section-title">Technical Indicators</h3>', unsafe_allow_html=True)
        
        # Technical indicators visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI
            if 'rsi_14' in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['date'] if 'date' in df.columns else df.index,
                    y=df['rsi_14'],
                    mode='lines',
                    name='RSI (14)',
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
        
        with col2:
            # MACD
            if all(col in df.columns for col in ['macd', 'macd_signal']):
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
                
                # Histogram
                if 'macd_diff' in df.columns:
                    colors = ['green' if val > 0 else 'red' for val in df['macd_diff']]
                    fig.add_trace(go.Bar(
                        x=df['date'] if 'date' in df.columns else df.index,
                        y=df['macd_diff'],
                        name='Histogram',
                        marker_color=colors,
                        opacity=0.3
                    ))
                
                fig.update_layout(
                    height=300,
                    xaxis_title="Date",
                    yaxis_title="MACD",
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        if st.session_state.models_trained:
            st.markdown('<h3 class="section-title">Feature Importance</h3>', unsafe_allow_html=True)
            
            importance = self.ml_pipeline.get_feature_importance()
            if importance is not None:
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
    
    def render_anomalies_tab(self):
        """Render anomalies tab"""
        if not st.session_state.anomalies_detected:
            st.info("Please detect anomalies first using the sidebar")
            return
        
        if not hasattr(self, 'anomalies'):
            st.error("Anomalies not detected. Please run detection first.")
            return
        
        df_anomalies = self.anomalies
        
        st.markdown('<h3 class="section-title">Anomaly Detection Results</h3>', unsafe_allow_html=True)
        
        # Anomaly summary
        anomaly_count = df_anomalies['is_anomaly'].sum()
        total_count = len(df_anomalies)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Anomalies Detected", f"{anomaly_count}")
        
        with col2:
            st.metric("Anomaly Rate", f"{(anomaly_count/total_count)*100:.1f}%")
        
        with col3:
            if 'anomaly_confidence' in df_anomalies.columns:
                avg_confidence = df_anomalies['anomaly_confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Anomaly visualization
        st.markdown('<h3 class="subsection-title">Anomaly Timeline</h3>', unsafe_allow_html=True)
        
        if 'close' in df_anomalies.columns:
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
            anomalies = df_anomalies[df_anomalies['is_anomaly'] == 1]
            if not anomalies.empty:
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
        st.markdown('<h3 class="subsection-title">Anomaly Details</h3>', unsafe_allow_html=True)
        
        if not anomalies.empty:
            # Select columns to display
            display_cols = []
            for col in ['date', 'close', 'volume', 'daily_return', 'anomaly_confidence', 'anomaly_type']:
                if col in anomalies.columns:
                    display_cols.append(col)
            
            if display_cols:
                st.dataframe(
                    anomalies[display_cols].sort_values('date' if 'date' in anomalies.columns else 'anomaly_confidence', 
                                                      ascending=False),
                    use_container_width=True
                )
    
    def render_export_tab(self):
        """Render export tab - NO EMOJIS"""
        st.markdown('<h3 class="section-title">Export Data</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Available Datasets")
            
            datasets = []
            if st.session_state.df_processed is not None:
                datasets.append(("Processed Data", st.session_state.df_processed))
            if st.session_state.df_features is not None:
                datasets.append(("Features Data", st.session_state.df_features))
            if hasattr(self, 'anomalies'):
                datasets.append(("Anomaly Data", self.anomalies))
            
            for name, df in datasets:
                with st.expander(f"{name} ({len(df)} rows)"):
                    st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.markdown("### Export Options")
            
            export_format = st.selectbox(
                "Select Format",
                ["CSV", "Excel", "JSON"],
                key="export_format"
            )
            
            selected_dataset = st.selectbox(
                "Select Dataset to Export",
                [name for name, _ in datasets],
                key="export_dataset"
            )
            
            if st.button("Download Data", use_container_width=True):
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
                            label="Click to Download CSV",
                            data=csv,
                            file_name=f"{selected_dataset.lower().replace(' ', '_')}.csv",
                            mime="text/csv",
                            key="download_csv"
                        )
                    
                    elif export_format == "Excel":
                        # For Excel, we need to use to_excel
                        excel_buffer = pd.ExcelWriter('temp.xlsx', engine='openpyxl')
                        df_to_export.to_excel(excel_buffer, index=False)
                        excel_buffer.save()
                        
                        with open('temp.xlsx', 'rb') as f:
                            st.download_button(
                                label="Click to Download Excel",
                                data=f,
                                file_name=f"{selected_dataset.lower().replace(' ', '_')}.xlsx",
                                mime="application/vnd.ms-excel",
                                key="download_excel"
                            )
                    
                    elif export_format == "JSON":
                        json_str = df_to_export.to_json(orient='records', indent=2)
                        st.download_button(
                            label="Click to Download JSON",
                            data=json_str,
                            file_name=f"{selected_dataset.lower().replace(' ', '_')}.json",
                            mime="application/json",
                            key="download_json"
                        )
    
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
    # Set page config
    st.set_page_config(
        page_title="Financial Analytics Platform",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize and run dashboard
    app = FinancialAnalyticsDashboard()
    app.run()
