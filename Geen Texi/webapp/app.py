import streamlit as st
import numpy as np
import joblib
import pickle
import os

# Set model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../Models')

# Load models
@st.cache_resource
def load_models():
    models = {}
    models['Random Forest'] = joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl'))
    models['Gradient Boosting'] = joblib.load(os.path.join(MODEL_DIR, 'gb_model.pkl'))
    models['Extra Trees'] = joblib.load(os.path.join(MODEL_DIR, 'et_model.pkl'))
    with open(os.path.join(MODEL_DIR, 'meta_model.pkl'), 'rb') as f:
        models['Stacking Ensemble'] = pickle.load(f)
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
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
