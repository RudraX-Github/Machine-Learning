import streamlit as st
import numpy as np
import joblib
import pickle
import os
import sys
import warnings

# Suppress numpy deprecation warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------
# ARCHITECTURAL FIX: Dynamic Path Handling
# ------------------------------------------------------------------------
# This calculates the path relative to this script file.
# 1. Get path of current script: .../Geen_Texi/webapp/app.py
# 2. Go up two levels to project root: .../Geen_Texi/
# 3. Join with 'Models' folder
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
MODEL_DIR = os.path.join(project_root, 'Models')

# Debugging: Print the path being used (optional, can be removed in production)
print(f"Loading models from: {MODEL_DIR}")

# Verify the path exists immediately to aid debugging
if not os.path.exists(MODEL_DIR):
    st.error(f"Configuration Error: The 'Models' directory was not found at: {MODEL_DIR}")
    st.stop()
# ------------------------------------------------------------------------

# Custom unpickler to handle numpy random state compatibility issues
class NumpyRNGPickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Handle numpy random state compatibility
        if module == 'numpy.random' and 'MT19937' in name:
            # Use the current numpy random state instead
            module = 'numpy.random._mt19937'
            name = 'MT19937'
        elif module == 'numpy.random._pickle':
            # Handle the pickle module reference
            return super().find_class('numpy.random._pickle', name)
        return super().find_class(module, name)

# Load models with compatibility handling
@st.cache_resource
def load_models():
    models = {}
    scaler = None
    
    # Load Random Forest model
    try:
        models['Random Forest'] = joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl'), mmap_mode=None)
    except (ValueError, ModuleNotFoundError, FileNotFoundError) as e:
        st.warning(f"Could not load Random Forest model: {e}")
    
    # Load Gradient Boosting model
    try:
        models['Gradient Boosting'] = joblib.load(os.path.join(MODEL_DIR, 'gb_model.pkl'), mmap_mode=None)
    except (ValueError, ModuleNotFoundError, FileNotFoundError) as e:
        st.warning(f"Could not load Gradient Boosting model: {e}")
    
    # Load Extra Trees model if it exists
    et_path = os.path.join(MODEL_DIR, 'et_model.pkl')
    if os.path.exists(et_path):
        try:
            models['Extra Trees'] = joblib.load(et_path, mmap_mode=None)
        except (ValueError, ModuleNotFoundError) as e:
            st.warning(f"Could not load Extra Trees model: {e}")
    
    # Load Stacking Ensemble model
    try:
        # Note: Using standard pickle here as per original code
        meta_path = os.path.join(MODEL_DIR, 'meta_model.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                models['Stacking Ensemble'] = pickle.load(f)
    except (ValueError, ModuleNotFoundError, FileNotFoundError) as e:
        st.warning(f"Could not load Stacking Ensemble model: {e}")
    
    # Load scaler
    try:
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'), mmap_mode=None)
    except FileNotFoundError:
        st.error("Critical Error: Scaler file (scaler.pkl) not found. Cannot proceed.")
        st.stop()
        
    return models, scaler

# Load resources
models, scaler = load_models()

st.title('Trip Fare Prediction Web App')
st.write('Select a model and enter the features to predict the trip fare.')

# Feature input fields for the selected features except rate_code
inputs = {}
col1, col2 = st.columns(2)

with col1:
    inputs['Passenger_count'] = st.number_input('Passenger Count', min_value=1.0, max_value=10.0, value=1.0, step=1.0)
    inputs['Trip_distance'] = st.number_input('Trip Distance (miles)', min_value=0.0, value=1.0)

with col2:
    inputs['Tolls_amount'] = st.number_input('Tolls Amount ($)', min_value=0.0, value=0.0)
    inputs['Fare_amount'] = st.number_input('Base Fare Amount ($)', min_value=0.0, value=0.0)

# Rate code dropdown and one-hot encoding
rate_code_options = {
    'Standard rate': 'rate_code_2',
    'JFK Airport flat rate': 'rate_code_3',
    'Newark Airport': 'rate_code_4',
    'Nassau or Westchester Counties': 'rate_code_5',
    'Negotiated fare': 'rate_code_6'
}
rate_code_selection = st.selectbox('Rate Code', list(rate_code_options.keys()))

# Build one-hot encoded rate_code features
rate_code_features = ['rate_code_2', 'rate_code_3', 'rate_code_4', 'rate_code_5', 'rate_code_6']
rate_code_values = [1 if rate_code_options[rate_code_selection] == col else 0 for col in rate_code_features]

# Prepare input for prediction (order must match scaler/model training)
# Structure: [Passenger_count, Trip_distance, Tolls_amount, Fare_amount, rate_code_2, rate_code_3, rate_code_4, rate_code_5, rate_code_6]
X_input = np.array([
    inputs['Passenger_count'],
    inputs['Trip_distance'],
    inputs['Tolls_amount'],
    inputs['Fare_amount'],
    *rate_code_values
]).reshape(1, -1)

if models:
    # Model selection
    st.markdown("### Prediction")
    model_choice = st.selectbox('Select Model for Prediction', list(models.keys()))

    if st.button('Predict Fare'):
        # Check if scaler is loaded before transforming
        if scaler:
            try:
                # Scale the input
                X_scaled = scaler.transform(X_input)
                
                # Predict
                if model_choice == 'Stacking Ensemble':
                    # For stacking, use meta-model directly on the 9 input features
                    pred = models['Stacking Ensemble'].predict(X_scaled)
                else:
                    pred = models[model_choice].predict(X_scaled)
                
                st.success(f'Predicted Total Amount: ${pred[0]:.2f}')
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("Scaler is missing, cannot make predictions.")
else:
    st.error("No models were loaded successfully. Please check the Models directory.")