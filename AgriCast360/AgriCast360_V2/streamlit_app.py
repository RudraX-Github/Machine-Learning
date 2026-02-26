"""
AgriCast360 - Streamlit Application
Comprehensive Crop Price, Yield & Supply Prediction System

Objective:
- Predict Crop Prices by Mandi (historical data + weather)
- Predict Yield & Supply (historical data + weather)
- Use ensemble of trained models with intelligent model selection
- Minimize prediction errors using price-range based model routing
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
import logging
import urllib.request
import os
import tempfile

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

st.set_page_config(
    page_title="AgriCast360 - Crop Price & Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# ENVIRONMENT DETECTION & PATH CONFIGURATION
# ============================================================================

# GitHub repository configuration
GITHUB_USERNAME = "RudraX-Github"
GITHUB_REPO = "Machine-Learning"
GITHUB_BRANCH = "main"
GITHUB_BASE_PATH = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/AgriCast360/AgriCast360_V2"
GITHUB_MODELS_URL = f"{GITHUB_BASE_PATH}/Models"
GITHUB_DATA_URL = f"{GITHUB_BASE_PATH}/Processed_Data"

# Dynamic local paths (fixes hardcoded D:\ path issues)
try:
    ROOT_FOLDER = Path(__file__).parent
except NameError:
    ROOT_FOLDER = Path.cwd()

MODELS_FOLDER = ROOT_FOLDER / "Models"
PROCESSED_DATA_FOLDER = ROOT_FOLDER / "Processed_Data"

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 10px;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.3em;
        color: #558B2F;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #F1F8E9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E7D32;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #F57C00;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2E7D32;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# GITHUB FILE DOWNLOAD UTILITIES
# ============================================================================

@st.cache_resource(show_spinner=False)
def download_file_from_github(file_url, file_type='pickle'):
    """Download a file from GitHub with robust error handling and caching"""
    try:
        # Use temp directory for better cross-platform/cloud compatibility
        cache_dir = Path(tempfile.gettempdir()) / '.agricast360_cache'
        cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Extract filename from URL
        filename = file_url.split('/')[-1]
        cache_path = cache_dir / filename
        
        # Download if not already cached
        if not cache_path.exists():
            logger.info(f"Downloading from GitHub: {filename}")
            req = urllib.request.Request(file_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(cache_path, 'wb') as out_file:
                out_file.write(response.read())
            logger.info(f"✅ Downloaded: {filename}")
        
        # Load the file based on type
        if file_type == 'pickle':
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        elif file_type == 'csv':
            return pd.read_csv(cache_path)
            
    except Exception as e:
        logger.error(f"Error downloading file {file_url}: {str(e)}")
        return None

# ============================================================================
# MODEL LOADING & CACHING
# ============================================================================

@st.cache_resource(show_spinner="Loading predictive models...")
def load_models():
    """Load all trained models and scalers from local or GitHub seamlessly"""
    models = {}
    
    # Define models dictionary
    model_files = {
        '01_linear_regression': 'Linear Regression',
        '02_random_forest': 'Random Forest',
        '03_xgboost': 'XGBoost',
        '05_linear_regression_tuned': 'Linear Regression (Tuned)',
        '06_random_forest_tuned': 'Random Forest (Tuned)',
        '07_xgboost_tuned': 'XGBoost (Tuned)',
        '08_stacking_ensemble': 'Stacking Ensemble',
        '09_voting_ensemble': 'Voting Ensemble'
    }
    
    for file_key, model_name in model_files.items():
        # 1. Try Loading Locally First
        model_path = MODELS_FOLDER / f"{file_key}.pkl"
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
                logger.info(f"✅ Loaded locally: {model_name}")
                continue  # Skip Github fetching if successful locally
            except Exception as e:
                logger.warning(f"⚠️ Failed to load local model {model_name}: {e}")
        
        # 2. Fallback to GitHub Download
        file_url = f"{GITHUB_MODELS_URL}/{file_key}.pkl"
        try:
            model = download_file_from_github(file_url, 'pickle')
            if model is not None:
                models[model_name] = model
        except Exception as e:
            logger.warning(f"Warning downloading {model_name}: {str(e)}")

    # Load scalers
    scalers = {}
    scaler_files = ['scaler_standard.pkl', 'scaler_robust.pkl']
    
    for scaler_file in scaler_files:
        scaler_name = scaler_file.replace('.pkl', '')
        scaler_path = PROCESSED_DATA_FOLDER / scaler_file
        
        # 1. Try Loading Locally First
        if scaler_path.exists():
            try:
                with open(scaler_path, 'rb') as f:
                    scalers[scaler_name] = pickle.load(f)
                logger.info(f"✅ Loaded locally scaler: {scaler_name}")
                continue
            except Exception as e:
                logger.warning(f"⚠️ Failed to load local scaler {scaler_file}: {e}")
                
        # 2. Fallback to GitHub Download
        file_url = f"{GITHUB_DATA_URL}/{scaler_file}"
        try:
            scaler = download_file_from_github(file_url, 'pickle')
            if scaler is not None:
                scalers[scaler_name] = scaler
        except Exception as e:
            logger.warning(f"Warning downloading scaler {scaler_file}: {str(e)}")
            
    return models, scalers

@st.cache_resource(show_spinner="Fetching datasets...")
def load_processed_data():
    """Load processed data and metadata from local or GitHub"""
    features_data = None
    features_scaled = None
    
    # 1. Try Features Raw Data
    raw_path = PROCESSED_DATA_FOLDER / "features_raw.csv"
    if raw_path.exists():
        try:
            features_data = pd.read_csv(raw_path, nrows=1000)
            logger.info("✅ Loaded local features_raw.csv")
        except Exception as e:
            logger.warning(f"Error loading local raw features: {e}")
            
    if features_data is None:
        file_url = f"{GITHUB_DATA_URL}/features_raw.csv"
        df = download_file_from_github(file_url, 'csv')
        if df is not None:
            features_data = df.iloc[:1000]
            logger.info("✅ Loaded features_raw.csv from GitHub")

    # 2. Try Scaled Features Data
    scaled_path = PROCESSED_DATA_FOLDER / "features_scaled_robust.csv"
    if scaled_path.exists():
        try:
            features_scaled = pd.read_csv(scaled_path, nrows=1000)
            logger.info("✅ Loaded local features_scaled_robust.csv")
        except Exception as e:
            logger.warning(f"Error loading local scaled features: {e}")
            
    if features_scaled is None:
        file_url = f"{GITHUB_DATA_URL}/features_scaled_robust.csv"
        df = download_file_from_github(file_url, 'csv')
        if df is not None:
            features_scaled = df.iloc[:1000]
            logger.info("✅ Loaded features_scaled_robust.csv from GitHub")
            
    return features_data, features_scaled

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_available_mandis(data, commodity):
    """Get list of available mandis for a specific commodity"""
    if data is None or data.empty or 'Market' not in data.columns:
        return []
    
    mandis = data[data['Commodity'] == commodity]['Market'].dropna().unique()
    return sorted([str(m) for m in mandis])

def get_feature_columns(features_df):
    
    exclude_cols = ['State', 'District', 'Market', 'Commodity', 'Variety', 
                    'Arrival_Date', 'Min_Price', 'Max_Price', 'Modal_Price',
                    'Commodity_Code', 'Source_Mandi', 'Date', 'Location_Key']
    
    return [col for col in features_df.columns if col not in exclude_cols]

def validate_input_features(input_data, required_features):
    """Validate input features"""
    missing_features = [f for f in required_features if f not in input_data.columns]
    if missing_features:
        return False, f"Missing features: {missing_features}"
    return True, "Valid"

def select_best_model_for_price_range(price_range_low, price_range_high, models_dict):
    """
    Intelligent model selection based on price range
    Different models perform better for different price ranges
    """
    price_midpoint = (price_range_low + price_range_high) / 2
    
    model_strategy = {
        'low': {  # Low price range (< 5000)
            'models': ['Linear Regression (Tuned)', 'Random Forest'],
            'description': 'For low-price commodities'
        },
        'medium': {  # Medium price range (5000-15000)
            'models': ['XGBoost (Tuned)', 'Voting Ensemble'],
            'description': 'For mid-range commodities'
        },
        'high': {  # High price range (> 15000)
            'models': ['Stacking Ensemble', 'XGBoost (Tuned)'],
            'description': 'For high-value commodities'
        }
    }
    
    if price_midpoint < 5000:
        category = 'low'
    elif price_midpoint < 15000:
        category = 'medium'
    else:
        category = 'high'
    
    recommended_models = [m for m in model_strategy[category]['models'] if m in models_dict]
    
    return category, recommended_models, model_strategy[category]['description']

def make_predictions(input_features, models_dict, selected_models=None):
    """Make predictions using selected models with smart feature matching"""
    predictions = {}
    
    if selected_models is None:
        selected_models = list(models_dict.keys())
    
    try:
        # Extract numeric features from input
        if isinstance(input_features, pd.Series):
            input_features = input_features.to_frame().T
        
        if isinstance(input_features, pd.DataFrame):
            # Extract only numeric features
            numeric_df = input_features.select_dtypes(include=[np.number]).copy()
            
            # Define metadata columns to exclude
            exclude_cols = ['State', 'District', 'Market', 'Commodity', 'Variety', 
                           'Arrival_Date', 'Commodity_Code', 'Source_Mandi', 
                           'Date', 'Location_Key']
            
            for col in exclude_cols:
                if col in numeric_df.columns:
                    numeric_df = numeric_df.drop(columns=[col], errors='ignore')
        else:
            numeric_df = pd.DataFrame(input_features)
        
        # Make predictions with adaptive feature handling
        for model_name in selected_models:
            if model_name in models_dict:
                try:
                    model = models_dict[model_name]
                    
                    # Determine expected number of features for this model
                    try:
                        n_features = model.n_features_in_
                    except:
                        try:
                            n_features = model.coef_.shape[0]
                        except:
                            n_features = numeric_df.shape[1]
                    
                    # Prepare data with correct feature count
                    if numeric_df.shape[1] >= n_features:
                        # Use first n_features columns
                        X = numeric_df.iloc[:, :n_features].values
                    else:
                        # Model expects fewer features than available
                        X = numeric_df.values
                    
                    # Reshape if needed
                    if len(X.shape) == 1:
                        X = X.reshape(1, -1)
                    
                    # Make prediction
                    pred = model.predict(X)[0]
                    predictions[model_name] = float(pred)
                    
                except Exception as e:
                    logger.warning(f"Error with {model_name}: {str(e)}")
                    predictions[model_name] = None
        
        return predictions
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {}

def create_prediction_visualization(predictions_dict, title="Model Predictions Comparison"):
    """Create visualization for predictions"""
    if not predictions_dict:
        return None
    
    # Filter out None values
    valid_preds = {k: v for k, v in predictions_dict.items() if v is not None}
    
    if not valid_preds:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(valid_preds.keys())
    predictions = list(valid_preds.values())
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(models)))
    bars = ax.barh(models, predictions, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, pred) in enumerate(zip(bars, predictions)):
        ax.text(pred, bar.get_y() + bar.get_height()/2, 
                f' ₹{pred:,.0f}', 
                va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Predicted Price (₹)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def get_historical_prices(data, commodity, location, days_back=30):
    """Get historical price data for a commodity and location"""
    if data is None or data.empty:
        return None
    
    try:
        # Filter by commodity and location
        filtered_df = data[
            (data['Commodity'] == commodity) & 
            (data['Market'].str.contains(location, case=False, na=False))
        ].copy()
        
        if filtered_df.empty:
            return None
        
        # Convert Arrival_Date to datetime
        if 'Arrival_Date' in filtered_df.columns:
            filtered_df['Arrival_Date'] = pd.to_datetime(filtered_df['Arrival_Date'], errors='coerce')
            filtered_df = filtered_df.dropna(subset=['Arrival_Date'])
            filtered_df = filtered_df.sort_values('Arrival_Date')
            
            # Keep only last N days
            if len(filtered_df) > 0:
                max_date = filtered_df['Arrival_Date'].max()
                cutoff_date = max_date - timedelta(days=days_back)
                filtered_df = filtered_df[filtered_df['Arrival_Date'] >= cutoff_date]
        
        return filtered_df
    except Exception as e:
        logger.warning(f"Error getting historical prices: {str(e)}")
        return None

def plot_historical_with_prediction(historical_data, prediction_price, commodity, location):
    """Plot historical prices with prediction overlay"""
    if historical_data is None or historical_data.empty:
        return None
    
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot historical prices
        if 'Arrival_Date' in historical_data.columns and 'Modal_Price' in historical_data.columns:
            dates = historical_data['Arrival_Date']
            prices = historical_data['Modal_Price']
            
            ax.plot(dates, prices, marker='o', linestyle='-', linewidth=2, 
                   markersize=6, color='#1f77b4', label='Historical Modal Price', alpha=0.8)
            
            # Fill area under the curve
            ax.fill_between(dates, prices, alpha=0.2, color='#1f77b4')
            
            # Add prediction line
            if prediction_price is not None:
                ax.axhline(y=prediction_price, color='#ff7f0e', linestyle='--', 
                          linewidth=2.5, label=f'Predicted Price: ₹{prediction_price:,.0f}')
            
            # Add min/max/avg lines
            min_price = prices.min()
            max_price = prices.max()
            avg_price = prices.mean()
            
            ax.axhline(y=min_price, color='#d62728', linestyle=':', linewidth=1.5, 
                       alpha=0.6, label=f'Historical Min: ₹{min_price:,.0f}')
            ax.axhline(y=max_price, color='#2ca02c', linestyle=':', linewidth=1.5, 
                       alpha=0.6, label=f'Historical Max: ₹{max_price:,.0f}')
            ax.axhline(y=avg_price, color='#9467bd', linestyle=':', linewidth=1.5, 
                       alpha=0.6, label=f'Historical Avg: ₹{avg_price:,.0f}')
            
            # Formatting
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Price (₹)', fontsize=12, fontweight='bold')
            ax.set_title(f'Historical Price Trend & Prediction\n{commodity} at {location}', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.legend(loc='best', fontsize=10, framealpha=0.95)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            return fig
    except Exception as e:
        logger.warning(f"Error plotting historical data: {str(e)}")
        return None

# ============================================================================
# STREAMLIT APP LAYOUT
# ============================================================================

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<div class='main-header'>🌾 AgriCast360</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Intelligent Crop Price & Yield Prediction System</div>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar Configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    app_mode = st.radio(
        "Select Mode:",
        ["🔮 Predictions", "📊 Model Info", "📈 Data Overview", "❓ Help & Documentation"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📋 About AgriCast360")
    st.info("""
    **AgriCast360** leverages:
    - ✅ 9 Pre-trained ML Models
    - ✅ Historical Crop Data
    - ✅ Real Weather Data
    - ✅ Intelligent Model Routing
    - ✅ Ensemble Predictions
    """)

# ============================================================================
# MODE 1: PREDICTIONS
# ============================================================================

if app_mode == "🔮 Predictions":
    st.header("🔮 Crop Price & Yield Predictions")
    
    # Load models
    models, scalers = load_models()
    features_data, features_scaled = load_processed_data()
    
    # Get dynamic commodities from data
    if features_data is not None and 'Commodity' in features_data.columns:
        available_commodities = sorted(features_data['Commodity'].dropna().unique().tolist())
    else:
        available_commodities = ["Ajwan", "Wheat", "Rice", "Cotton", "Tobacco", "Other"]
    
    if not models:
        st.error("❌ No models loaded. Please check the Github connection or Models folder.")
    else:
        st.success(f"✅ Loaded {len(models)} trained models successfully!")
        st.info(f"📊 Available commodities: {len(available_commodities)}")
        
        # Create tabs for different prediction types
        tab1, tab2, tab3 = st.tabs([
            "💰 Crop Price Prediction",
            "🌾 Yield & Supply Prediction",
            "📊 Batch Predictions"
        ])
        
        # ====== TAB 1: CROP PRICE PREDICTION ======
        with tab1:
            st.subheader("💰 Predict Crop Price by Mandi")
            
            col1, col2 = st.columns(2)
            
            with col1:
                commodity = st.selectbox(
                    "Select Commodity:",
                    available_commodities,
                    key="commodity_price"
                )
                
                # Get available mandis for selected commodity
                available_mandis = get_available_mandis(features_data, commodity)
                if not available_mandis:
                    available_mandis = ["Ahmedabad", "Surat", "Amreli", "Other"]
                
                mandi_location = st.selectbox(
                    "Select Mandi Location:",
                    available_mandis,
                    key="mandi_location"
                )
            
            with col2:
                st.markdown("#### Date & History Settings")
                prediction_date = st.date_input(
                    "Prediction Date:",
                    value=datetime.now().date(),
                    key="prediction_date"
                )
                
                date_range_days = st.slider(
                    "Historical data (days back):",
                    min_value=7,
                    max_value=180,
                    value=30,
                    step=7,
                    key="date_range_days"
                )
            
            # Advanced Features
            with st.expander("📊 Advanced Weather & Market Features"):
                adv_col1, adv_col2, adv_col3 = st.columns(3)
                
                with adv_col1:
                    temperature = st.slider(
                        "Temperature (°C):",
                        min_value=10.0,
                        max_value=45.0,
                        value=28.0,
                        step=0.5,
                        key="temp_price"
                    )
                    rainfall = st.slider(
                        "Rainfall (mm):",
                        min_value=0.0,
                        max_value=500.0,
                        value=50.0,
                        step=5.0,
                        key="rainfall_price"
                    )
                
                with adv_col2:
                    humidity = st.slider(
                        "Humidity (%):",
                        min_value=0.0,
                        max_value=100.0,
                        value=65.0,
                        step=5.0,
                        key="humidity_price"
                    )
                    wind_speed = st.slider(
                        "Wind Speed (km/h):",
                        min_value=0.0,
                        max_value=50.0,
                        value=2.5,
                        step=0.5,
                        key="wind_speed_price"
                    )
                
                with adv_col3:
                    solar_rad = st.slider(
                        "Solar Radiation (MJ/m²):",
                        min_value=50.0,
                        max_value=350.0,
                        value=200.0,
                        step=10.0,
                        key="solar_rad_price"
                    )
                    days_since_arrival = st.slider(
                        "Days Since Last Record:",
                        min_value=0,
                        max_value=30,
                        value=1,
                        step=1,
                        key="days_since_arrival"
                    )
            
            # Model Selection
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 🎯 Select Models for Prediction")
                st.write("Use all available models for ensemble prediction")
            
            with col2:
                use_all_models = st.checkbox(
                    "Use All Models",
                    value=True,
                    key="use_all_models_price"
                )
            
            # Model Selection
            if use_all_models:
                selected_models = list(models.keys())
            else:
                selected_models = st.multiselect(
                    "Select Models for Ensemble:",
                    list(models.keys()),
                    default=list(models.keys())[:3],
                    key="selected_models_price"
                )
            
            # Prediction Button
            if st.button("🚀 Predict Price", use_container_width=True, key="predict_price_btn"):
                st.info("⏳ Running predictions with selected models...")
                
                try:
                    # Get commodity AND location-specific data
                    if features_data is not None and features_scaled is not None and len(features_scaled) > 0:
                        # Use features_data for filtering (has metadata columns)
                        # Filter data for selected commodity AND mandi location
                        if mandi_location != "Other":
                            filtered_data = features_data[
                                (features_data.get('Commodity') == commodity) &
                                (features_data['Market'].str.contains(mandi_location, case=False, na=False))
                            ]
                        else:
                            # If "Other" selected, just use commodity data
                            filtered_data = features_data[features_data.get('Commodity') == commodity]
                        
                        if len(filtered_data) > 0:
                            # Get the label index of the last matching row
                            last_index = filtered_data.index[-1]
                            # Use label-based indexing to get from scaled features (same index)
                            feature_vector = features_scaled.loc[last_index:last_index].copy()
                            location_info = f"{commodity} at {mandi_location}"
                        else:
                            # No data available for this commodity + location combination
                            st.error(f"❌ No data available for **{commodity}** at **{mandi_location}**")
                            st.info("💡 Please try:")
                            st.write("- Select a different commodity")
                            st.write("- Select a different mandi location")
                            st.write("- Choose 'Other' to see generic commodity predictions")
                            st.stop()
                        
                        # Make predictions with actual feature dimensions
                        predictions = make_predictions(feature_vector, models, selected_models)
                    
                    if predictions:
                        st.success("✅ Predictions Generated!")
                        st.info(f"📍 Prediction for: **{location_info}**")
                        
                        # Display predictions
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📊 Predicted Prices")
                            valid_preds = {k: v for k, v in predictions.items() if v is not None}
                            
                            if valid_preds:
                                # Calculate ensemble prediction (mean)
                                ensemble_price = np.mean(list(valid_preds.values()))
                                
                                # Display key metrics
                                metric_col1, metric_col2, metric_col3 = st.columns(3)
                                
                                with metric_col1:
                                    st.metric(
                                        "Ensemble Prediction (Mean)",
                                        f"₹{ensemble_price:,.0f}"
                                    )
                                
                                with metric_col2:
                                    min_pred = min(valid_preds.values())
                                    st.metric("Min Prediction", f"₹{min_pred:,.0f}")
                                
                                with metric_col3:
                                    max_pred = max(valid_preds.values())
                                    st.metric("Max Prediction", f"₹{max_pred:,.0f}")
                                
                                # Detailed predictions
                                st.markdown("#### Individual Model Predictions:")
                                pred_df = pd.DataFrame({
                                    'Model': list(valid_preds.keys()),
                                    'Predicted Price (₹)': list(valid_preds.values())
                                }).sort_values('Predicted Price (₹)', ascending=False)
                                
                                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                        
                        with col2:
                            # Visualization
                            fig = create_prediction_visualization(valid_preds, "Price Predictions by Model")
                            if fig:
                                st.pyplot(fig)
                        
                        # Prediction Confidence
                        st.markdown("### 📈 Prediction Confidence Analysis")
                        
                        confidence_col1, confidence_col2, confidence_col3 = st.columns(3)
                        
                        with confidence_col1:
                            if valid_preds:
                                std_dev = np.std(list(valid_preds.values()))
                                mean_pred = np.mean(list(valid_preds.values()))
                                cv = (std_dev / mean_pred) * 100
                                
                                if cv < 10:
                                    confidence = "🟢 HIGH"
                                elif cv < 20:
                                    confidence = "🟡 MEDIUM"
                                else:
                                    confidence = "🔴 LOW"
                                
                                st.markdown(f"**Confidence Level:** {confidence}")
                                st.metric("Coefficient of Variation", f"{cv:.2f}%")
                        
                        with confidence_col2:
                            if valid_preds:
                                st.metric("Standard Deviation", f"₹{std_dev:,.0f}")
                        
                        with confidence_col3:
                            if valid_preds:
                                st.metric("Number of Models", len(valid_preds))
                    
                    else:
                        st.error("❌ No predictions generated. Please check your inputs.")
                    
                    # Show historical price data
                    st.markdown("---")
                    st.markdown("### 📅 Historical Price Analysis")
                    
                    hist_col1, hist_col2 = st.columns([1, 2])
                    
                    with hist_col1:
                        show_history = st.checkbox(
                            "📊 Show Historical Prices",
                            value=True,
                            key="show_history"
                        )
                    
                    if show_history and features_data is not None:
                        try:
                            # Get historical data
                            historical_data = get_historical_prices(
                                features_data, 
                                commodity, 
                                mandi_location if mandi_location != "Other" else "Ahmedabad",
                                days_back=date_range_days
                            )
                            
                            if historical_data is not None and len(historical_data) > 0:
                                # Display historical stats
                                st.markdown(f"#### {commodity} at {mandi_location}")
                                
                                hist_metric_col1, hist_metric_col2, hist_metric_col3, hist_metric_col4 = st.columns(4)
                                
                                if 'Modal_Price' in historical_data.columns:
                                    modal_prices = historical_data['Modal_Price'].dropna()
                                    
                                    if len(modal_prices) > 0:
                                        with hist_metric_col1:
                                            st.metric("Historical Avg", f"₹{modal_prices.mean():,.0f}")
                                        
                                        with hist_metric_col2:
                                            st.metric("Min Price", f"₹{modal_prices.min():,.0f}")
                                        
                                        with hist_metric_col3:
                                            st.metric("Max Price", f"₹{modal_prices.max():,.0f}")
                                        
                                        with hist_metric_col4:
                                            std_dev = modal_prices.std()
                                            st.metric("Std Dev", f"₹{std_dev:,.0f}")
                                
                                # Plot historical with prediction
                                ensemble_price = np.mean([p for p in valid_preds.values() if p is not None])
                                hist_fig = plot_historical_with_prediction(
                                    historical_data, 
                                    ensemble_price, 
                                    commodity, 
                                    mandi_location
                                )
                                
                                if hist_fig:
                                    st.pyplot(hist_fig)
                                
                                # Show recent data table
                                with st.expander("📋 Recent Historical Data (Last 10 records)"):
                                    display_cols = ['Arrival_Date', 'Commodity', 'Market', 'Min_Price', 'Modal_Price', 'Max_Price']
                                    available_display_cols = [col for col in display_cols if col in historical_data.columns]
                                    
                                    if available_display_cols:
                                        recent_data = historical_data[available_display_cols].tail(10).copy()
                                        if 'Arrival_Date' in recent_data.columns:
                                            recent_data['Arrival_Date'] = recent_data['Arrival_Date'].dt.strftime('%Y-%m-%d')
                                        st.dataframe(recent_data, use_container_width=True, hide_index=True)
                            else:
                                st.info(f"ℹ️ No historical data available for {commodity} at {mandi_location} in the last {date_range_days} days.")
                        
                        except Exception as e:
                            st.warning(f"⚠️ Could not load historical data: {str(e)}")
                
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    logger.error(f"Prediction error: {str(e)}")

        # ====== TAB 2: YIELD & SUPPLY PREDICTION ======
        with tab2:
            st.subheader("🌾 Predict Yield & Supply")
            
            st.info("""
            This section predicts crop yield and supply based on:
            - Historical production data
            - Weather patterns
            - Market dynamics
            - Seasonal trends
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                yield_commodity = st.selectbox(
                    "Select Commodity:",
                    available_commodities,
                    key="commodity_yield"
                )
                season = st.selectbox(
                    "Select Season:",
                    ["Kharif", "Rabi", "Summer", "Year-Round"],
                    key="season"
                )
            
            with col2:
                area_cultivated = st.number_input(
                    "Area Cultivated (Hectares):",
                    min_value=100,
                    max_value=1000000,
                    value=50000,
                    step=1000,
                    key="area_cultivated"
                )
                previous_yield = st.number_input(
                    "Previous Year Yield (Tonnes/Ha):",
                    min_value=0.5,
                    max_value=20.0,
                    value=2.5,
                    step=0.1,
                    key="previous_yield"
                )
            
            # Weather impact factors
            with st.expander("🌦️ Weather Impact Factors"):
                weather_col1, weather_col2, weather_col3 = st.columns(3)
                
                with weather_col1:
                    rainfall_season = st.slider(
                        "Seasonal Rainfall (mm):",
                        min_value=0.0,
                        max_value=2000.0,
                        value=600.0,
                        step=50.0,
                        key="rainfall_season"
                    )
                    drought_risk = st.slider(
                        "Drought Risk (%):",
                        min_value=0,
                        max_value=100,
                        value=10,
                        step=5,
                        key="drought_risk"
                    )
                
                with weather_col2:
                    avg_temperature = st.slider(
                        "Average Temperature (°C):",
                        min_value=15.0,
                        max_value=40.0,
                        value=28.0,
                        step=0.5,
                        key="avg_temp_season"
                    )
                    heatwave_days = st.slider(
                        "Heatwave Days:",
                        min_value=0,
                        max_value=30,
                        value=2,
                        step=1,
                        key="heatwave_days"
                    )
                
                with weather_col3:
                    sunshine_hours = st.slider(
                        "Average Sunshine Hours:",
                        min_value=4.0,
                        max_value=12.0,
                        value=8.0,
                        step=0.5,
                        key="sunshine_hours"
                    )
                    humidity_season = st.slider(
                        "Average Humidity (%):",
                        min_value=30.0,
                        max_value=95.0,
                        value=65.0,
                        step=5.0,
                        key="humidity_season"
                    )
            
            if st.button("🚀 Predict Yield & Supply", use_container_width=True, key="predict_yield_btn"):
                st.info("⏳ Calculating yield and supply estimates...")
                
                try:
                    # Use actual feature data from loaded CSV
                    if features_scaled is not None and len(features_scaled) > 0:
                        # Get a sample row for yield predictions
                        yield_features = features_scaled.iloc[0:1].copy()
                        
                        # Make predictions
                        yield_predictions = make_predictions(yield_features, models)
                    
                    if yield_predictions:
                        st.success("✅ Yield & Supply Estimates Generated!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📊 Yield Estimates")
                            
                            valid_yields = {k: v for k, v in yield_predictions.items() if v is not None}
                            
                            if valid_yields:
                                ensemble_yield = np.mean(list(valid_yields.values()))
                                total_production = ensemble_yield * area_cultivated / 1000  # Convert to tonnes
                                
                                metric_col1, metric_col2 = st.columns(2)
                                
                                with metric_col1:
                                    st.metric(
                                        "Ensemble Yield",
                                        f"{ensemble_yield:.2f} T/Ha",
                                        delta=f"vs {previous_yield:.2f} T/Ha"
                                    )
                                
                                with metric_col2:
                                    st.metric(
                                        "Total Production (Estimate)",
                                        f"{total_production:,.0f} Tonnes"
                                    )
                                
                                # Detailed yield estimates
                                st.markdown("#### Individual Model Estimates:")
                                yield_df = pd.DataFrame({
                                    'Model': list(valid_yields.keys()),
                                    'Yield (T/Ha)': list(valid_yields.values()),
                                    'Total Production (T)': [v * area_cultivated / 1000 for v in valid_yields.values()]
                                }).sort_values('Yield (T/Ha)', ascending=False)
                                
                                st.dataframe(yield_df, use_container_width=True, hide_index=True)
                        
                        with col2:
                            # Visualization
                            fig = create_prediction_visualization(valid_yields, "Yield Estimates by Model")
                            if fig:
                                st.pyplot(fig)
                    else:
                        st.error("❌ Could not generate predictions. Data not available.")
                
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    logger.error(f"Prediction error: {str(e)}")

        # ====== TAB 3: BATCH PREDICTIONS ======
        with tab3:
            st.subheader("📊 Batch Predictions")
            
            st.info("""
            Upload a CSV file with multiple records to get predictions for all rows.
            The file should contain the required feature columns.
            """)
            
            uploaded_file = st.file_uploader(
                "Upload CSV file for batch predictions:",
                type="csv",
                key="batch_upload"
            )
            
            if uploaded_file:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.write(f"Loaded {len(batch_df)} records")
                    
                    st.dataframe(batch_df.head(), use_container_width=True)
                    
                    if st.button("🚀 Process Batch Predictions", use_container_width=True, key="batch_predict"):
                        st.info("⏳ Processing batch predictions...")
                        
                        progress_bar = st.progress(0)
                        predictions_list = []
                        
                        for idx, row in batch_df.iterrows():
                            predictions = make_predictions(row, models)
                            predictions_list.append(predictions)
                            progress_bar.progress((idx + 1) / len(batch_df))
                        
                        # Compile results
                        results_df = pd.DataFrame(predictions_list)
                        
                        st.success("✅ Batch predictions completed!")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions (CSV)",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_batch"
                        )
                
                except Exception as e:
                    st.error(f"❌ Error processing batch file: {str(e)}")

# ============================================================================
# MODE 2: MODEL INFORMATION
# ============================================================================

elif app_mode == "📊 Model Info":
    st.header("📊 Model Information & Performance")
    
    models, scalers = load_models()
    
    st.markdown("### 🤖 Trained Models")
    
    model_info = {
        'Linear Regression': {
            'type': 'Baseline',
            'complexity': 'Low',
            'speed': 'Very Fast',
            'best_for': 'Linear relationships'
        },
        'Random Forest': {
            'type': 'Ensemble (Bagging)',
            'complexity': 'Medium',
            'speed': 'Fast',
            'best_for': 'Non-linear relationships, Feature importance'
        },
        'XGBoost': {
            'type': 'Ensemble (Boosting)',
            'complexity': 'High',
            'speed': 'Medium',
            'best_for': 'Complex patterns, Best accuracy'
        },
        'Linear Regression (Tuned)': {
            'type': 'Optimized Baseline',
            'complexity': 'Low',
            'speed': 'Very Fast',
            'best_for': 'Stable predictions with tuned parameters'
        },
        'Random Forest (Tuned)': {
            'type': 'Optimized Ensemble',
            'complexity': 'Medium',
            'speed': 'Fast',
            'best_for': 'Improved stability and accuracy'
        },
        'XGBoost (Tuned)': {
            'type': 'Optimized Boosting',
            'complexity': 'High',
            'speed': 'Medium',
            'best_for': 'State-of-the-art predictions'
        },
        'Stacking Ensemble': {
            'type': 'Meta-Ensemble',
            'complexity': 'Very High',
            'speed': 'Slow',
            'best_for': 'Maximum accuracy with model diversity'
        },
        'Voting Ensemble': {
            'type': 'Simple Ensemble',
            'complexity': 'High',
            'speed': 'Slow',
            'best_for': 'Robust predictions via voting'
        }
    }
    
    # Display model cards
    for model_name, info in model_info.items():
        if model_name in models:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model", model_name)
            with col2:
                st.metric("Type", info['type'])
            with col3:
                st.metric("Complexity", info['complexity'])
            with col4:
                st.metric("Best For", info['best_for'])
            
            st.markdown(f"⚡ **Speed:** {info['speed']}")
            st.divider()
    
    # Model Selection Recommendations
    st.markdown("### 🎯 Recommended Model Selection Strategies")
    
    strategies = {
        'Low Price Range (<₹5,000)': ['Linear Regression (Tuned)', 'Random Forest'],
        'Medium Price Range (₹5,000-₹15,000)': ['XGBoost (Tuned)', 'Voting Ensemble'],
        'High Price Range (>₹15,000)': ['Stacking Ensemble', 'XGBoost (Tuned)'],
        'Maximum Accuracy': ['Stacking Ensemble', 'Voting Ensemble'],
        'Fastest Prediction': ['Linear Regression', 'Linear Regression (Tuned)'],
        'Best Balance': ['XGBoost (Tuned)', 'Random Forest (Tuned)']
    }
    
    for strategy, recommended_models in strategies.items():
        with st.expander(f"🎯 {strategy}"):
            st.write("**Recommended Models:**")
            for model in recommended_models:
                if model in models:
                    st.write(f"✅ {model}")

# ============================================================================
# MODE 3: DATA OVERVIEW
# ============================================================================

elif app_mode == "📈 Data Overview":
    st.header("📈 Dataset Overview")
    
    features_data, features_scaled = load_processed_data()
    
    if features_data is not None:
        st.markdown("### 📊 Feature Statistics")
        
        tab1, tab2, tab3 = st.tabs(["Overview", "Statistics", "Visualizations"])
        
        with tab1:
            st.write(f"**Dataset Shape:** {features_data.shape[0]} rows × {features_data.shape[1]} columns")
            st.write(f"**Date Range:** {features_data['Date'].min()} to {features_data['Date'].max()}")
            
            st.markdown("#### Key Columns:")
            st.dataframe(features_data.head(), use_container_width=True)
        
        with tab2:
            st.markdown("#### Statistical Summary")
            st.dataframe(features_data.describe(), use_container_width=True)
        
        with tab3:
            st.markdown("#### Price Distribution")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(features_data['Modal_Price'].dropna(), bins=50, color='steelblue', edgecolor='black')
            ax.set_xlabel('Modal Price (₹)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Modal Prices')
            st.pyplot(fig)
            
            st.markdown("#### Feature Correlations (Top Features)")
            # Calculate correlations with price
            numeric_cols = features_data.select_dtypes(include=[np.number]).columns
            correlations = features_data[numeric_cols].corr()['Modal_Price'].sort_values(ascending=False)[1:11]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            correlations.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
            ax.set_xlabel('Correlation with Modal Price')
            ax.set_title('Top 10 Features Correlated with Price')
            st.pyplot(fig)

# ============================================================================
# MODE 4: HELP & DOCUMENTATION
# ============================================================================

elif app_mode == "❓ Help & Documentation":
    st.header("❓ Help & Documentation")
    
    st.markdown("""
    ## Overview
    
    **AgriCast360** is an intelligent crop price and yield prediction system that leverages:
    - 9 pre-trained machine learning models
    - Historical crop market data
    - Real-time weather data
    - Advanced ensemble techniques
    - Intelligent model selection based on price ranges
    
    ## Key Features
    
    ### 1️⃣ **Crop Price Prediction**
    - Predicts market prices by Mandi location
    - Considers historical trends and weather patterns
    - Uses intelligent model selection based on price range
    - Provides ensemble predictions for robustness
    
    ### 2️⃣ **Yield & Supply Estimation**
    - Estimates crop yield based on weather conditions
    - Accounts for area cultivated and seasonal factors
    - Predicts total production capacity
    - Considers climate risks (drought, heatwaves)
    
    ### 3️⃣ **Intelligent Model Routing**
    Different models are recommended for different price ranges:
    - **Low Price (<₹5,000):** Linear models, Random Forest
    - **Medium Price (₹5,000-₹15,000):** XGBoost, Voting Ensemble
    - **High Price (>₹15,000):** Stacking Ensemble, XGBoost
    
    ### 4️⃣ **Batch Processing**
    - Process multiple predictions at once
    - Export results to CSV
    - Perfect for market analysis
    
    ## Input Features
    
    ### Required Weather Data:
    - **Temperature (°C):** Average daily/seasonal temperature
    - **Rainfall (mm):** Precipitation amount
    - **Humidity (%):** Relative humidity levels
    - **Wind Speed (km/h):** Wind velocity
    - **Solar Radiation (MJ/m²):** Solar energy
    
    ### Required Market Data:
    - **Commodity:** Type of crop
    - **Mandi Location:** Market location
    - **Price Range:** Expected price bracket
    - **Area Cultivated:** In hectares
    - **Previous Yield:** Historical baseline
    
    ## Model Performance
    
    All models have been trained and validated on historical data with:
    - **MAE:** Mean Absolute Error (₹)
    - **RMSE:** Root Mean Squared Error (₹)
    - **MAPE:** Mean Absolute Percentage Error (%)
    - **R² Score:** Coefficient of Determination (0-1)
    
    ## How to Use
    
    1. **Go to Predictions Tab**
    2. **Select Prediction Type:**
       - 💰 Crop Price Prediction
       - 🌾 Yield & Supply Prediction
       - 📊 Batch Predictions
    3. **Enter Required Input:**
       - Commodity and location details
       - Weather parameters
       - Historical data (if available)
    4. **Configure Model Selection:**
       - Use automatic recommendations
       - Or select specific models
    5. **Get Results:**
       - Individual model predictions
       - Ensemble prediction (mean/weighted)
       - Confidence analysis
       - Visualizations
    6. **Download Results** (for batch predictions)
    
    ## Best Practices
    
    - **Always use ensemble predictions** for critical decisions
    - **Check confidence metrics** (low CV = high confidence)
    - **Consider multiple models** to understand uncertainty
    - **Validate with historical data** when possible
    - **Update predictions** regularly with new data
    - **Use recommended models** for optimal results
    
    ## Troubleshooting
    
    ### "No models loaded"
    - Check that Models folder exists and contains .pkl files
    - Ensure all model files are in: `Models/` directory
    
    ### "Missing features"
    - Ensure all required input fields are filled
    - Check data types match requirements
    
    ### "Prediction values seem off"
    - Verify input ranges are realistic
    - Check that weather data is seasonally appropriate
    - Compare with historical averages
    
    ## Contact & Support
    
    For issues or questions:
    - Review the model information section
    - Check data overview visualizations
    - Refer to feature descriptions above
    
    ## Technical Details
    
    - **Framework:** Streamlit (UI)
    - **ML Library:** Scikit-learn, XGBoost, TensorFlow
    - **Data Processing:** Pandas, NumPy
    - **Visualization:** Matplotlib, Seaborn
    
    """)
    
    st.markdown("---")
    st.markdown("**Made with ❤️ for Agricultural Analytics**")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("### 🌾 AgriCast360")
    st.write("Intelligent Crop Prediction System")

with footer_col2:
    st.markdown("### 📊 Features")
    st.write("""
    - 9 ML Models
    - Weather Data
    - Market Analysis
    - Batch Processing
    """)

with footer_col3:
    st.markdown("### ⚙️ Tech Stack")
    st.write("""
    - Streamlit
    - Scikit-learn
    - XGBoost
    - TensorFlow
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #558B2F;'>" +
            "<p><strong>AgriCast360 © 2024</strong> | Powered by ML & Weather Analytics</p>" +
            "</div>", unsafe_allow_html=True)