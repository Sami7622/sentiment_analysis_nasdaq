#!/usr/bin/env python3
"""
Streamlit Dashboard for Financial Sentiment Analysis
Allows users to select between XGBoost and FinBERT models for sentiment analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import inference classes
from inference import SentimentPredictor as XGBoostPredictor
from inference_finbert import FinBERTPredictor

# Page configuration
st.set_page_config(
    page_title="Financial Sentiment Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'xgboost_predictor' not in st.session_state:
    st.session_state.xgboost_predictor = None
if 'finbert_predictor' not in st.session_state:
    st.session_state.finbert_predictor = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = None


@st.cache_resource
def load_xgboost_model():
    """Load XGBoost model (cached)"""
    try:
        return XGBoostPredictor()
    except Exception as e:
        st.error(f"Error loading XGBoost model: {e}")
        return None


@st.cache_resource
def load_finbert_model():
    """Load FinBERT model (cached)"""
    try:
        return FinBERTPredictor()
    except Exception as e:
        st.error(f"Error loading FinBERT model: {e}")
        return None


def load_dataset():
    """Load and prepare the dataset"""
    try:
        df = pd.read_csv("raw_dataset.csv")
        df = df.dropna(subset=['description', 'sentiment', 'datetime'])
        
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        
        # Map sentiment labels
        sent_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        df['label'] = df['sentiment'].map(sent_map)
        df = df.dropna(subset=['label'])
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


def predict_sentiment(text, predictor):
    """Predict sentiment for a single text"""
    if predictor is None:
        return None
    try:
        result = predictor.predict(text, return_proba=True)
        return result
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def analyze_dataset(df, predictor):
    """Analyze entire dataset or sample"""
    if df is None or predictor is None:
        return None
    
    predictions = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        text = str(row['description'])
        result = predict_sentiment(text, predictor)
        
        if result:
            predictions.append({
                'datetime': row['datetime'],
                'ticker': row.get('ticker', 'UNKNOWN'),
                'text': text[:100] + '...' if len(text) > 100 else text,
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'positive_prob': result['probabilities'].get('positive', 0),
                'negative_prob': result['probabilities'].get('negative', 0),
                'neutral_prob': result['probabilities'].get('neutral', 0)
            })
        
        # Update progress
        progress = (len(predictions) / len(df))
        progress_bar.progress(progress)
        status_text.text(f"Processing: {len(predictions)}/{len(df)} samples")
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(predictions)


def create_sentiment_distribution_chart(df):
    """Create sentiment distribution pie chart"""
    sentiment_counts = df['sentiment'].value_counts()
    
    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
    color_map = [colors.get(sent, '#95a5a6') for sent in sentiment_counts.index]
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color=sentiment_counts.index,
        color_discrete_map=colors
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        showlegend=True,
        height=400,
        font=dict(size=12)
    )
    return fig


def create_sentiment_bar_chart(df):
    """Create sentiment distribution bar chart"""
    sentiment_counts = df['sentiment'].value_counts().sort_index()
    total = len(df)
    percentages = (sentiment_counts / total * 100).round(2)
    
    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
    color_map = [colors.get(sent, '#95a5a6') for sent in sentiment_counts.index]
    
    fig = go.Figure(data=[
        go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            text=[f'{p}%' for p in percentages],
            textposition='auto',
            marker_color=color_map
        )
    ])
    
    fig.update_layout(
        title="Sentiment Distribution (Counts and Percentages)",
        xaxis_title="Sentiment",
        yaxis_title="Count",
        height=400,
        showlegend=False
    )
    
    return fig


def create_sentiment_over_time_chart(df, ticker=None):
    """Create sentiment over time chart"""
    if ticker:
        df_filtered = df[df['ticker'] == ticker].copy()
        title = f"Sentiment Over Time for {ticker}"
    else:
        df_filtered = df.copy()
        title = "Sentiment Over Time (All Instruments)"
    
    # Group by date and sentiment
    df_filtered['date'] = pd.to_datetime(df_filtered['datetime']).dt.date
    daily_sentiment = df_filtered.groupby(['date', 'sentiment']).size().reset_index(name='count')
    daily_sentiment = daily_sentiment.pivot(index='date', columns='sentiment', values='count').fillna(0)
    
    # Calculate percentages
    daily_sentiment['total'] = daily_sentiment.sum(axis=1)
    for col in ['positive', 'negative', 'neutral']:
        if col in daily_sentiment.columns:
            daily_sentiment[f'{col}_pct'] = (daily_sentiment[col] / daily_sentiment['total'] * 100).round(2)
    
    # Create line chart
    fig = go.Figure()
    
    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment in daily_sentiment.columns:
            fig.add_trace(go.Scatter(
                x=daily_sentiment.index,
                y=daily_sentiment[f'{sentiment}_pct'],
                mode='lines+markers',
                name=sentiment.capitalize(),
                line=dict(color=colors[sentiment], width=2),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Percentage (%)",
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_sentiment_timeline_chart(df, ticker=None):
    """Create timeline chart showing sentiment distribution over time"""
    if ticker:
        df_filtered = df[df['ticker'] == ticker].copy()
        title = f"Sentiment Timeline for {ticker}"
    else:
        df_filtered = df.copy()
        title = "Sentiment Timeline (All Instruments)"
    
    df_filtered['date'] = pd.to_datetime(df_filtered['datetime']).dt.date
    
    # Count sentiments per day
    daily_counts = df_filtered.groupby(['date', 'sentiment']).size().reset_index(name='count')
    
    # Create stacked area chart
    fig = px.area(
        daily_counts,
        x='date',
        y='count',
        color='sentiment',
        title=title,
        color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Mentions",
        height=500,
        hovermode='x unified'
    )
    
    return fig


def create_sentiment_vs_time_points(df, ticker, start_date, end_date):
    """Create a point timeline of individual labels over a selectable date range"""
    df_plot = df.copy()
    df_plot['date'] = pd.to_datetime(df_plot['datetime']).dt.date
    if ticker and ticker != 'All':
        df_plot = df_plot[df_plot['ticker'] == ticker]
    if start_date:
        df_plot = df_plot[df_plot['date'] >= start_date]
    if end_date:
        df_plot = df_plot[df_plot['date'] <= end_date]
    if df_plot.empty:
        return go.Figure()
    
    # Ensure a consistent order for categorical y-axis
    category_order = ['negative', 'neutral', 'positive']
    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
    
    fig = px.scatter(
        df_plot,
        x='date',
        y='sentiment',
        color='sentiment',
        color_discrete_map=colors,
        category_orders={'sentiment': category_order},
        hover_data={'ticker': True, 'confidence': True, 'text': True, 'date': False},
        opacity=0.7
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    fig.update_layout(
        title=f"Sentiment vs Time ({'All' if ticker == 'All' else ticker})",
        xaxis_title="Date",
        yaxis_title="Sentiment Label",
        height=500,
        yaxis=dict(categoryorder='array', categoryarray=category_order)
    )
    return fig


# Main App
def main():
    st.markdown('<h1 class="main-header">📈 Financial Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_option = st.radio(
            "Select Model:",
            ["XGBoost (BERT + XGBoost)", "FinBERT (Fine-tuned)"],
            help="Choose the sentiment analysis model to use"
        )
        
        st.markdown("---")
        
        # Analysis mode
        analysis_mode = st.radio(
            "Analysis Mode:",
            ["Single Text Input", "Dataset Analysis"],
            help="Analyze a single text or process the entire dataset"
        )
        
        st.markdown("---")
        
        # Model info
        st.subheader("ℹ️ Model Information")
        if model_option == "XGBoost (BERT + XGBoost)":
            st.info("""
            **XGBoost Model:**
            - Uses BERT embeddings
            - Trained with Optuna optimization
            - Fast inference
            """)
        else:
            st.info("""
            **FinBERT Model:**
            - Fine-tuned on financial text
            - Domain-specific embeddings
            - Higher accuracy for financial sentiment
            """)
    
    # Main content area
    if model_option == "XGBoost (BERT + XGBoost)":
        if st.session_state.xgboost_predictor is None:
            with st.spinner("Loading XGBoost model..."):
                st.session_state.xgboost_predictor = load_xgboost_model()
        predictor = st.session_state.xgboost_predictor
        model_name = "XGBoost"
    else:
        if st.session_state.finbert_predictor is None:
            with st.spinner("Loading FinBERT model..."):
                st.session_state.finbert_predictor = load_finbert_model()
        predictor = st.session_state.finbert_predictor
        model_name = "FinBERT"
    
    if predictor is None:
        st.error("⚠️ Model could not be loaded. Please ensure the model files exist.")
        st.stop()
    
    # Single Text Input Mode
    if analysis_mode == "Single Text Input":
        st.subheader("📝 Single Text Sentiment Analysis")
        
        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Enter financial news, tweet, or any text related to financial instruments..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            predict_button = st.button("🔍 Analyze Sentiment", type="primary")
        
        if predict_button and text_input:
            with st.spinner("Analyzing sentiment..."):
                result = predict_sentiment(text_input, predictor)
            
            if result:
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sentiment", result['sentiment'].upper())
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.2%}")
                with col3:
                    st.metric("Model", model_name)
                
                # Probabilities
                st.subheader("📊 Sentiment Probabilities")
                prob_df = pd.DataFrame([
                    {'Sentiment': k.capitalize(), 'Probability': v}
                    for k, v in result['probabilities'].items()
                ])
                
                fig_prob = px.bar(
                    prob_df,
                    x='Sentiment',
                    y='Probability',
                    color='Sentiment',
                    color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#f39c12'},
                    text='Probability'
                )
                fig_prob.update_traces(textposition='outside', texttemplate='%{text:.2%}')
                fig_prob.update_layout(
                    height=300,
                    showlegend=False,
                    yaxis_title="Probability",
                    yaxis=dict(tickformat='.0%')
                )
                st.plotly_chart(fig_prob, width='stretch')
        
        elif predict_button:
            st.warning("Please enter some text to analyze.")
    
    # Dataset Analysis Mode
    else:
        st.subheader("📊 Dataset Analysis")
        
        # Load dataset
        df = load_dataset()
        
        if df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                ticker_filter = st.selectbox(
                    "Filter by Ticker (Optional):",
                    options=['All'] + sorted(df['ticker'].unique().tolist()),
                    help="Filter analysis by specific financial instrument"
                )
            
            with col2:
                st.write("")  # Spacing
                analyze_button = st.button("🚀 Analyze Dataset", type="primary")
            
            if analyze_button:
                # Filter by ticker if selected
                if ticker_filter != 'All':
                    df = df[df['ticker'] == ticker_filter]
                    st.info(f"Analyzing {len(df)} samples for ticker: {ticker_filter}")
                
                # Analyze dataset
                with st.spinner(f"Analyzing dataset with {model_name} model..."):
                    predictions_df = analyze_dataset(df, predictor)
                
                if predictions_df is not None and len(predictions_df) > 0:
                    st.session_state.predictions_df = predictions_df
                    
                    # Display summary statistics
                    st.success(f"✅ Successfully analyzed {len(predictions_df)} samples!")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total = len(predictions_df)
                    positive_pct = (predictions_df['sentiment'] == 'positive').sum() / total * 100
                    negative_pct = (predictions_df['sentiment'] == 'negative').sum() / total * 100
                    neutral_pct = (predictions_df['sentiment'] == 'neutral').sum() / total * 100
                    avg_confidence = predictions_df['confidence'].mean() * 100
                    
                    with col1:
                        st.metric("Total Samples", total)
                    with col2:
                        st.metric("Positive", f"{positive_pct:.1f}%")
                    with col3:
                        st.metric("Negative", f"{negative_pct:.1f}%")
                    with col4:
                        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                    
                    # Visualizations
                    st.markdown("---")
                    st.subheader("📈 Visualizations")
                    
                    # Create two columns for charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(
                            create_sentiment_distribution_chart(predictions_df),
                            width='stretch'
                        )
                    
                    with col2:
                        st.plotly_chart(
                            create_sentiment_bar_chart(predictions_df),
                            width='stretch'
                        )
                    
                    # Sentiment over time
                    st.subheader("📅 Sentiment Over Time")
                    
                    # Ticker selection for time series
                    available_tickers = sorted(predictions_df['ticker'].unique().tolist())
                    if len(available_tickers) > 1:
                        selected_ticker = st.selectbox(
                            "Select Ticker for Time Series:",
                            options=['All'] + available_tickers,
                            key="time_series_ticker"
                        )
                    else:
                        selected_ticker = available_tickers[0] if available_tickers else 'All'
                    
                    if selected_ticker == 'All':
                        st.plotly_chart(
                            create_sentiment_over_time_chart(predictions_df),
                            width='stretch'
                        )
                    else:
                        st.plotly_chart(
                            create_sentiment_over_time_chart(predictions_df, ticker=selected_ticker),
                            width='stretch'
                        )
                    
                    # Timeline chart
                    st.subheader("⏱️ Sentiment Timeline")
                    if selected_ticker == 'All':
                        st.plotly_chart(
                            create_sentiment_timeline_chart(predictions_df),
                            width='stretch'
                        )
                    else:
                        st.plotly_chart(
                            create_sentiment_timeline_chart(predictions_df, ticker=selected_ticker),
                            width='stretch'
                        )
                    
                    # Sentiment vs Time (filterable scatter by date range and ticker)
                    st.markdown("---")
                    st.subheader("🎯 Sentiment vs Time (Filterable)")
                    
                    # Compute date bounds from predictions
                    min_date = pd.to_datetime(predictions_df['datetime']).min().date()
                    max_date = pd.to_datetime(predictions_df['datetime']).max().date()
                    
                    c1, c2, c3 = st.columns([2, 2, 2])
                    with c1:
                        svt_ticker = st.selectbox(
                            "Select Ticker:",
                            options=['All'] + sorted(predictions_df['ticker'].unique().tolist()),
                            index=0,
                            key="svt_ticker"
                        )
                    with c2:
                        svt_start = st.date_input(
                            "From date:",
                            value=min_date,
                            min_value=min_date,
                            max_value=max_date,
                            key="svt_start"
                        )
                    with c3:
                        svt_end = st.date_input(
                            "To date:",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date,
                            key="svt_end"
                        )
                    
                    if svt_start > svt_end:
                        st.warning("Start date is after end date. Please adjust the range.")
                    else:
                        st.plotly_chart(
                            create_sentiment_vs_time_points(
                                predictions_df, svt_ticker, svt_start, svt_end
                            ),
                            width='stretch'
                        )
                    
                    # Data table
                    st.markdown("---")
                    st.subheader("📋 Prediction Results")
                    
                    with st.expander("View Detailed Results"):
                        display_df = predictions_df[['datetime', 'ticker', 'sentiment', 'confidence', 'text']].copy()
                        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
                        st.dataframe(display_df, use_container_width=True, height=400)
                        
                        # Download button
                        csv = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"sentiment_analysis_{model_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("No predictions generated. Please check your input.")
        else:
            st.error("Could not load dataset. Please ensure 'raw_dataset.csv' exists.")


if __name__ == "__main__":
    main()

