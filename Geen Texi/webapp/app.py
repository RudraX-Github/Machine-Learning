import streamlit as st
import numpy as np
import joblib
import pickle
import os
import sys
import warnings

# Suppress numpy deprecation warnings
warnings.filterwarnings('ignore')

# Set model directory
MODEL_DIR = r"D:\CUDA_Experiments\Git_HUB\Machine-Learning\Geen Texi\Models"

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
    
    # Load Random Forest model
    try:
        models['Random Forest'] = joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl'), mmap_mode=None)
    except (ValueError, ModuleNotFoundError) as e:
        st.warning(f"Could not load Random Forest model: {e}")
    
    # Load Gradient Boosting model
    try:
        models['Gradient Boosting'] = joblib.load(os.path.join(MODEL_DIR, 'gb_model.pkl'), mmap_mode=None)
    except (ValueError, ModuleNotFoundError) as e:
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
        with open(os.path.join(MODEL_DIR, 'meta_model.pkl'), 'rb') as f:
            models['Stacking Ensemble'] = pickle.load(f)
    except (ValueError, ModuleNotFoundError) as e:
        st.warning(f"Could not load Stacking Ensemble model: {e}")
    
    # Load scaler
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'), mmap_mode=None)
    return models, scaler

models, scaler = load_models()

st.title('Trip Fare Prediction Web App')
st.write('Select a model and enter the features to predict the trip fare.')



# Feature input fields for the selected features except rate_code
inputs = {}
inputs['Passenger_count'] = st.number_input('Passenger_count', value=0.0)
inputs['Trip_distance'] = st.number_input('Trip_distance', value=0.0)
inputs['Tolls_amount'] = st.number_input('Tolls_amount', value=0.0)
inputs['Fare_amount'] = st.number_input('Fare_amount', value=0.0)

# Rate code dropdown and one-hot encoding
rate_code_options = {
    'Standard rate': 'rate_code_2',
    'JFK Airport flat rate': 'rate_code_3',
    'Newark Airport': 'rate_code_4',
    'Nassau or Westchester Counties': 'rate_code_5',
    'Negotiated fare': 'rate_code_6'
}
rate_code_selection = st.selectbox('rate_code', list(rate_code_options.keys()))

# Build one-hot encoded rate_code features
rate_code_features = ['rate_code_2', 'rate_code_3', 'rate_code_4', 'rate_code_5', 'rate_code_6']
rate_code_values = [1 if rate_code_options[rate_code_selection] == col else 0 for col in rate_code_features]

# Prepare input for prediction (order must match scaler/model)
X_input = np.array([
    inputs['Passenger_count'],
    inputs['Trip_distance'],
    inputs['Tolls_amount'],
    inputs['Fare_amount'],
    *rate_code_values
]).reshape(1, -1)

# Scale input
X_scaled = scaler.transform(X_input)

# Model selection
model_choice = st.selectbox('Select Model', list(models.keys()))

if st.button('Predict'):
    if model_choice == 'Stacking Ensemble':
        # For stacking, use meta-model directly on the 9 input features
        pred = models['Stacking Ensemble'].predict(X_scaled)
    else:
        pred = models[model_choice].predict(X_scaled)
    st.success(f'Predicted Total Amount: {pred[0]:.2f}')
