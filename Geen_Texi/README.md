🚖 NYC Smart Taxi Predictor

AI-Powered Fare Estimation & Route Analytics for Green Taxis

📖 Project Overview

The NYC Smart Taxi Predictor is a machine learning application designed to estimate taxi fares for NYC's "Green Taxi" service. Built with a focus on precision and user experience, the application utilizes an XGBoost Regressor trained on over 100,000 trip records to predict fares with high accuracy (R² ≈ 0.97).

The project features a full lifecycle implementation: from Exploratory Data Analysis (EDA) and model training to a production-ready web interface built with Streamlit.

🚀 Key Features

🤖 Precision AI Prediction: Real-time fare estimation using a tuned XGBoost model.

🗺️ Interactive Mapping: Visualizes pickup and drop-off coordinates using PyDeck 3D maps.

📍 Smart Landmark Integration: One-click selection for 40+ major NYC landmarks (JFK, LaGuardia, MoMA, Central Park, etc.).

📊 Embedded Analytics: A dedicated "Data Analytics" tab featuring an integrated Pandas Profiling report.

🎨 Premium UI: Custom CSS implementation for a "Dark Mode" aesthetic with glass-morphism effects.

🛠️ Tech Stack

Frontend & Deployment

Streamlit: Main application framework.

HTML/CSS: Custom styling for the dashboard and documentation.

PyDeck: Geospatial data visualization.

Backend & Machine Learning

Python: Core logic.

Pandas & NumPy: Data manipulation and feature engineering.

Scikit-Learn: Pipeline construction and metric evaluation.

XGBoost: Gradient boosting framework for the regression model.

Joblib: Model serialization.

🧠 Model Performance

Multiple regression algorithms were tested during the development phase. The XGBoost Regressor outperformed traditional linear models and other tree-based ensembles.

Model

R² Score

MSE

MAE

XGBoost (Final)

0.9718

2.2278

0.4790

Linear Regression

~0.78

Higher

Higher

Note: Performance metrics based on a sample size of 100,000 records from the 2013 Green Taxi Dataset.

📂 Repository Structure

├── Green_texi.py               # Main Streamlit Application entry point
├── Green_chunk_3_eda.ipynb     # Jupyter Notebook: EDA, Training, and Evaluation
├── documentation_green_taxi_app.html # Project Showcase/Documentation Webpage
├── green_model.pkl             # Trained XGBoost Model (Generated via Notebook)
├── Advanced_pandas_profiling...html # Generated Profiling Report (Required for Analytics tab)
└── README.md                   # Project Documentation


⚙️ Installation & Usage

1. Clone the Repository

git clone [https://github.com/yourusername/nyc-smart-taxi-predictor.git](https://github.com/yourusername/nyc-smart-taxi-predictor.git)
cd nyc-smart-taxi-predictor


2. Install Dependencies

Ensure you have Python installed, then install the required libraries:

pip install pandas numpy scikit-learn xgboost streamlit pydeck joblib seaborn matplotlib


3. Generate the Model (If not present)

If green_model.pkl is not in the directory, run the notebook to train and save it:

Open Green_chunk_3_eda.ipynb.

Run all cells.

This will output green_model.pkl.

4. Run the Application

streamlit run Green_texi.py


📊 Data Insights

The application analyzes the 2013 Green Taxi dataset, focusing on:

Trip Distance: Calculated using the Haversine formula.

Time Features: Breakdown of pickup hours and days of the week.

Geospatial Clustering: Analysis of high-density pickup/drop-off zones.

📸 Usage Guide

Dashboard: Enter Pickup/Drop-off coordinates manually or select popular landmarks from the dropdown.

Prediction: Input passenger count and click "Estimate Fare" to see the predicted cost and distance.

Analytics: Switch to the "Data Analytics" tab to view the deep-dive statistical report of the training data.

System Specs: View model architecture details and performance metrics.

📜 License

This project is open-source and available for educational and portfolio purposes.

Developed with 💚 using Python