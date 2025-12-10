import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_upload import DataUpload
from modules.data_cleaning import DataCleaning
from modules.feature_engineering import FeatureEngineering
from modules.ml_models import MLModels
from modules.anomaly_detection import AnomalyDetection
from modules.insights import InsightsGenerator

# Page configuration
st.set_page_config(
    page_title="Financial Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0B3D91;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0B3D91;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0B3D91;
        margin-bottom: 1rem;
    }
    .insight-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FFD700;
        margin-bottom: 0.5rem;
    }
    .anomaly-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0B3D91;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class FinancialAnalyticsDashboard:
    def __init__(self):
        self.data_upload = DataUpload()
        self.data_cleaning = DataCleaning()
        self.feature_engineering = FeatureEngineering()
        self.ml_models = MLModels()
        self.anomaly_detection = AnomalyDetection()
        self.insights_generator = InsightsGenerator()
        
        self.df_raw = None
        self.df_clean = None
        self.df_features = None
        self.df_targets = None
        self.df_anomalies = None
        self.model_results = {}
        self.insights = []
        
    def run(self):
        """Main application flow"""
        # Header
        st.markdown('<h1 class="main-header">Financial Analytics Platform</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style='color: #666; margin-bottom: 2rem;'>
        Enterprise-grade platform for financial market prediction and anomaly detection. 
        Upload CSV data to generate predictions, detect anomalies, and gain AI-driven insights.
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.markdown("### Configuration")
            
            # Data upload
            success, df_raw = self.data_upload.upload_csv()
            if success:
                self.df_raw = df_raw
                
                # Data cleaning
                with st.spinner("Cleaning data..."):
                    self.df_clean = self.data_cleaning.clean_data(self.df_raw)
                
                # Feature engineering
                with st.spinner("Generating features..."):
                    self.df_features = self.feature_engineering.create_features(self.df_clean)
                    self.df_targets = self.feature_engineering.get_target_variables(self.df_features)
                
                st.sidebar.success("Data processed successfully")
                
                # Display data info
                st.sidebar.markdown("---")
                st.sidebar.markdown("### Data Summary")
                if 'date' in self.df_clean.columns:
                    st.sidebar.write(f"**Period:** {self.df_clean['date'].min().date()} to {self.df_clean['date'].max().date()}")
                st.sidebar.write(f"**Data Points:** {len(self.df_clean):,}")
                st.sidebar.write(f"**Features Generated:** {len(self.df_features.columns)}")
        
        # Main content tabs
        if self.df_clean is not None:
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Overview", 
                "Model Predictions", 
                "Feature Analysis", 
                "Anomaly Detection",
                "Export Data"
            ])
            
            with tab1:
                self._render_overview_tab()
            
            with tab2:
                self._render_predictions_tab()
            
            with tab3:
                self._render_feature_analysis_tab()
            
            with tab4:
                self._render_anomaly_tab()
            
            with tab5:
                self._render_export_tab()
        else:
            st.info("Please upload a CSV file or use sample data to begin analysis.")
            
            # Display sample data format
            st.markdown("### Expected Data Format")
            st.code("""
            Column names should include:
            - Date (or similar datetime column)
            - Open (opening price)
            - High (daily high)
            - Low (daily low)
            - Close (closing price)
            - Volume (trading volume)
            
            Example:
            Date,Open,High,Low,Close,Volume
            2023-01-01,100.0,102.5,99.5,101.2,1000000
            2023-01-02,101.2,103.0,100.5,102.8,1200000
            """)
    
    def _render_overview_tab(self):
        """Render overview tab with charts and insights"""
        st.markdown('<h2 class="sub-header">Market Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'close' in self.df_clean.columns:
                current_price = self.df_clean['close'].iloc[-1]
                price_change = ((self.df_clean['close'].iloc[-1] / self.df_clean['close'].iloc[0]) - 1) * 100
                st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f}%")
        
        with col2:
            if 'volume' in self.df_clean.columns:
                avg_volume = self.df_clean['volume'].mean()
                st.metric("Avg Daily Volume", f"{avg_volume:,.0f}")
        
        with col3:
            if 'daily_return' in self.df_features.columns:
                avg_return = self.df_features['daily_return'].mean() * 100
                st.metric("Avg Daily Return", f"{avg_return:.2f}%")
        
        with col4:
            if 'volatility_20' in self.df_features.columns:
                current_vol = self.df_features['volatility_20'].iloc[-1] * 100
                st.metric("Current Volatility", f"{current_vol:.2f}%")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Price chart
            st.markdown("#### Price Trend")
            fig = go.Figure()
            
            if 'close' in self.df_clean.columns and 'date' in self.df_clean.columns:
                fig.add_trace(go.Scatter(
                    x=self.df_clean['date'],
                    y=self.df_clean['close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#0B3D91', width=2)
                ))
                
                # Add moving averages if available
                if 'sma_20' in self.df_features.columns:
                    fig.add_trace(go.Scatter(
                        x=self.df_clean['date'],
                        y=self.df_features['sma_20'],
                        mode='lines',
                        name='20-Day SMA',
                        line=dict(color='#FFD700', width=1, dash='dash')
                    ))
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Price",
                    hovermode='x unified',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Volume chart
            st.markdown("#### Trading Volume")
            if 'volume' in self.df_clean.columns:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=self.df_clean['date'],
                    y=self.df_clean['volume'],
                    name='Volume',
                    marker_color='#666'
                ))
                
                if 'volume_sma' in self.df_features.columns:
                    fig.add_trace(go.Scatter(
                        x=self.df_clean['date'],
                        y=self.df_features['volume_sma'],
                        mode='lines',
                        name='20-Day Avg Volume',
                        line=dict(color='#FFD700', width=2)
                    ))
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    hovermode='x unified',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # AI Insights
        st.markdown("---")
        st.markdown('<h2 class="sub-header">AI-Generated Insights</h2>', unsafe_allow_html=True)
        
        # Generate insights if not already generated
        if not self.insights:
            with st.spinner("Generating insights..."):
                # Train models and detect anomalies first
                self._train_models()
                self._detect_anomalies()
                self.insights = self.insights_generator.generate_insights(
                    self.df_features, 
                    self.ml_models.results,
                    self.df_anomalies
                )
        
        # Display insights
        for insight in self.insights[:5]:  # Show top 5 insights
            st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
    
    def _train_models(self):
        """Train all ML models"""
        with st.spinner("Training machine learning models..."):
            # Prepare data
            X_train, X_test, y_train, y_test, feature_names = self.ml_models.prepare_data(
                self.df_targets, target_type='regression'
            )
            self.ml_models.feature_names = feature_names
            
            # Train models
            self.ml_models.train_linear_regression(X_train, X_test, y_train, y_test)
            
            # Prepare classification data
            X_train_clf, X_test_clf, y_train_clf, y_test_clf, _ = self.ml_models.prepare_data(
                self.df_targets, target_type='classification'
            )
            self.ml_models.train_knn_classifier(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
            
            # Train neural network
            X_train_nn, X_test_nn, y_train_nn, y_test_nn, _ = self.ml_models.prepare_data(
                self.df_targets, target_type='return'
            )
            self.ml_models.train_neural_network(X_train_nn, X_test_nn, y_train_nn, y_test_nn)
            
            # Train gradient boosted trees
            self.ml_models.train_gradient_boosted_trees(X_train, X_test, y_train, y_test, 'regression')
    
    def _detect_anomalies(self):
        """Detect anomalies in the data"""
        with st.spinner("Detecting anomalies..."):
            self.df_anomalies = self.anomaly_detection.detect_anomalies(self.df_features)
    
    def _render_predictions_tab(self):
        """Render model predictions tab"""
        st.markdown('<h2 class="sub-header">Model Predictions</h2>', unsafe_allow_html=True)
        
        # Ensure models are trained
        if not self.ml_models.results:
            self._train_models()
        
        # Model selection
        model_options = list(self.ml_models.results.keys())
        selected_model = st.selectbox("Select Model", model_options, format_func=lambda x: x.replace('_', ' ').title())
        
        if selected_model in self.ml_models.results:
            results = self.ml_models.results[selected_model]
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            metrics = results['metrics']
            for metric_name, metric_value in metrics.items():
                if 'test_' in metric_name:
                    display_name = metric_name.replace('test_', '').upper()
                    col1.metric(display_name, f"{metric_value:.4f}")
            
            # Prediction vs Actual chart
            st.markdown("#### Predictions vs Actual")
            
            # Create date range for test predictions
            test_size = len(results['predictions'])
            if 'date' in self.df_clean.columns:
                dates = self.df_clean['date'].iloc[-test_size:]
            else:
                dates = range(test_size)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=results['actual'],
                mode='lines',
                name='Actual',
                line=dict(color='#0B3D91', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=dates,
                y=results['predictions'],
                mode='lines',
                name='Predicted',
                line=dict(color='#FFD700', width=2, dash='dash')
            ))
            
            fig.update_layout(
                height=500,
                xaxis_title="Date",
                yaxis_title="Value",
                hovermode='x unified',
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Residuals plot
            st.markdown("#### Prediction Residuals")
            residuals = results['actual'] - results['predictions']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='#666', size=8)
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            
            fig.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="Residual (Actual - Predicted)",
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_feature_analysis_tab(self):
        """Render feature analysis tab"""
        st.markdown('<h2 class="sub-header">Feature Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature importance
            st.markdown("#### Top Feature Importance")
            
            # Get feature importance from gradient boosted model
            if 'gradient_boosted_regression' in self.ml_models.feature_importance:
                importance_df = self.ml_models.feature_importance['gradient_boosted_regression'].head(10)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=importance_df['importance'],
                    y=importance_df['feature'],
                    orientation='h',
                    marker_color='#0B3D91'
                ))
                
                fig.update_layout(
                    height=500,
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation heatmap
            st.markdown("#### Feature Correlation")
            
            # Select top features for correlation
            numeric_cols = self.df_features.select_dtypes(include=[np.number]).columns.tolist()
            top_features = numeric_cols[:10]  # First 10 numeric features
            
            if len(top_features) > 1:
                corr_matrix = self.df_features[top_features].corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    text=corr_matrix.round(2).values,
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                
                fig.update_layout(
                    height=500,
                    xaxis_title="Features",
                    yaxis_title="Features",
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Rolling returns distribution
        st.markdown("---")
        st.markdown("#### Returns Distribution")
        
        if 'daily_return' in self.df_features.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=self.df_features['daily_return'].dropna(),
                    nbinsx=50,
                    marker_color='#0B3D91',
                    opacity=0.7
                ))
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Daily Return",
                    yaxis_title="Frequency",
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=self.df_features['daily_return'].dropna(),
                    name='Returns',
                    boxpoints='outliers',
                    marker_color='#0B3D91'
                ))
                
                fig.update_layout(
                    height=400,
                    yaxis_title="Daily Return",
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_anomaly_tab(self):
        """Render anomaly detection tab"""
        st.markdown('<h2 class="sub-header">Anomaly Detection</h2>', unsafe_allow_html=True)
        
        # Ensure anomalies are detected
        if self.df_anomalies is None:
            self._detect_anomalies()
        
        # Anomaly summary
        anomaly_summary = self.anomaly_detection.get_anomaly_summary(self.df_anomalies)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Anomalies", f"{anomaly_summary['anomaly_count']}")
        
        with col2:
            st.metric("Anomaly Rate", f"{anomaly_summary['anomaly_percentage']}%")
        
        with col3:
            st.metric("Most Common Type", anomaly_summary['most_common_type'])
        
        with col4:
            st.metric("Max Confidence", f"{anomaly_summary['max_anomaly_score']:.1%}")
        
        # Anomaly visualization
        st.markdown("#### Anomaly Timeline")
        
        if 'date' in self.df_anomalies.columns and 'close' in self.df_anomalies.columns:
            # Create scatter plot with anomalies highlighted
            fig = go.Figure()
            
            # Normal points
            normal_mask = self.df_anomalies['is_anomaly'] == 0
            fig.add_trace(go.Scatter(
                x=self.df_anomalies.loc[normal_mask, 'date'],
                y=self.df_anomalies.loc[normal_mask, 'close'],
                mode='markers',
                name='Normal',
                marker=dict(color='#0B3D91', size=6, opacity=0.6)
            ))
            
            # Anomaly points
            anomaly_mask = self.df_anomalies['is_anomaly'] == 1
            if anomaly_mask.any():
                fig.add_trace(go.Scatter(
                    x=self.df_anomalies.loc[anomaly_mask, 'date'],
                    y=self.df_anomalies.loc[anomaly_mask, 'close'],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='#dc3545', size=10, symbol='x')
                ))
            
            fig.update_layout(
                height=500,
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode='x unified',
                template='plotly_white',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly details table
        st.markdown("#### Anomaly Details")
        
        if anomaly_mask.any():
            anomaly_details = self.df_anomalies[anomaly_mask].copy()
            
            # Select relevant columns
            display_cols = ['date', 'close', 'daily_return', 'volume', 'anomaly_type', 'anomaly_confidence']
            display_cols = [col for col in display_cols if col in anomaly_details.columns]
            
            st.dataframe(
                anomaly_details[display_cols].sort_values('date', ascending=False),
                use_container_width=True
            )
        
        # Anomaly insights
        st.markdown("#### Anomaly Insights")
        
        anomaly_insights = [
            f"Detected {anomaly_summary['anomaly_count']} anomalies representing {anomaly_summary['anomaly_percentage']}% of the dataset.",
            f"The average anomaly confidence score is {anomaly_summary['avg_anomaly_score']:.1%}.",
            f"Most common anomaly type: {anomaly_summary['most_common_type']}."
        ]
        
        for insight in anomaly_insights:
            st.markdown(f'<div class="anomaly-card">{insight}</div>', unsafe_allow_html=True)
    
    def _render_export_tab(self):
        """Render data export tab"""
        st.markdown('<h2 class="sub-header">Export Data</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Export buttons
        with col1:
            if st.button("Export Cleaned Data", use_container_width=True):
                self._download_dataframe(self.df_clean, "cleaned_data.csv")
        
        with col2:
            if st.button("Export Features", use_container_width=True):
                self._download_dataframe(self.df_features, "features_data.csv")
        
        with col3:
            if st.button("Export Anomalies", use_container_width=True) and self.df_anomalies is not None:
                self._download_dataframe(self.df_anomalies, "anomalies_data.csv")
        
        with col4:
            if st.button("Export Model Results", use_container_width=True) and self.ml_models.results:
                self._export_model_results()
        
        # Data preview
        st.markdown("---")
        st.markdown("#### Data Preview")
        
        dataset_option = st.selectbox(
            "Select Dataset to Preview",
            ["Cleaned Data", "Features Data", "Anomaly Data"]
        )
        
        if dataset_option == "Cleaned Data":
            st.dataframe(self.df_clean.head(20), use_container_width=True)
        elif dataset_option == "Features Data":
            st.dataframe(self.df_features.head(20), use_container_width=True)
        elif dataset_option == "Anomaly Data" and self.df_anomalies is not None:
            st.dataframe(self.df_anomalies.head(20), use_container_width=True)
    
    def _download_dataframe(self, df, filename):
        """Helper function to download dataframe as CSV"""
        csv = df.to_csv(index=False)
        st.download_button(
            label="Click to Download",
            data=csv,
            file_name=filename,
            mime="text/csv",
            key=f"download_{filename}"
        )
    
    def _export_model_results(self):
        """Export model results to CSV"""
        results_data = []
        
        for model_name, results in self.ml_models.results.items():
            if 'metrics' in results:
                row = {'Model': model_name}
                row.update(results['metrics'])
                results_data.append(row)
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Click to Download Model Results",
                data=csv,
                file_name="model_results.csv",
                mime="text/csv",
                key="download_model_results"
            )

# Run the application
if __name__ == "__main__":
    app = FinancialAnalyticsDashboard()
    app.run()
