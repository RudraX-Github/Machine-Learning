🌾 AgriCast360_V2

Intelligent Crop Price, Yield & Supply Prediction System

AgriCast360 is a comprehensive machine learning solution designed to analyze agricultural market trends and weather patterns to predict crop prices and yields. By integrating historical Mandi (market) data with granular weather metrics, the system provides actionable insights for agricultural decision-making through an interactive Streamlit dashboard.

🚀 Project Overview

The primary objective of AgriCast360 is to build accurate prediction models for:

Crop Prices: Based on historical Mandi trends and weather conditions.

Yield & Supply: analyzing supply chain fluctuations.

The system evaluates 9 different machine learning models to determine the best performance, routing predictions through the most accurate model based on specific price ranges and commodity types.

✨ Key Features

Multi-Source Data Integration: Combines specific Market (Mandi) geospatial data with historical weather data (via Weatherbit API).

Advanced Feature Engineering: Includes temporal features, lag correlations, seasonal decomposition, and weather-price interaction terms.

Comprehensive EDA: Detailed Exploratory Data Analysis and Time Series Analysis to identify seasonality and trends.

Robust Modeling Pipeline:

Models: Linear Regression, Random Forest, XGBoost, Neural Networks (TensorFlow), and Ensemble methods (Stacking/Voting).

Metrics: MAE, RMSE, and MAPE.

Interactive Dashboard: A Streamlit-based UI for real-time predictions, market analysis, and model comparison.

🛠️ Tech Stack

Language: Python 3.11+

Interface: Streamlit

Data Processing: Pandas, NumPy, SciPy

Machine Learning: Scikit-learn, XGBoost, TensorFlow/Keras

Visualization: Matplotlib, Seaborn

APIs: Weatherbit API (for historical weather data)

📂 Project Structure & Workflow

The project follows a structured data science pipeline implemented across several Jupyter Notebooks:

1. Data Analysis & Exploration

EDA.ipynb: The entry point. Handles data loading, cleaning, and initial exploration of Mandi and Weather data. Generates the initial HTML reports.

Time_Series_Analysis.ipynb: Deep dive into temporal patterns. Analyzes seasonality, trends, residuals, and autocorrelation to understand cyclical market behaviors.

2. Feature Engineering

Features_building.ipynb: Constructs the feature set, including price lags, rolling averages, and weather interactions.

Feature_Preprocessing.ipynb: Prepares data for modeling. Handles skewness correction, kurtosis control, and scaling (Standard, Robust, MinMax, PowerTransformer).

3. Model Development

Model_building.ipynb: Trains and evaluates 9 different models.

Current Best Model: Random Forest (MAE: 313.18)

Generates AgriCast_Full_Report.html comparing model performance.

4. Deployment

streamlit_app.py: The production-ready application.

Loads trained models and processing pipelines.

Provides an interface for users to select commodities and markets (e.g., Bardoli, Surat, Mahuva).

Visualizes predictions vs. historical averages.

📊 Performance Highlights

Based on the AgriCast_Full_Report.html, the system evaluated 9 models.

Best Performing Model: Random Forest Regressor

Mean Absolute Error (MAE): ~313.18

Key Markets Analyzed: Bardoli, Kosamba, Mahuva, Mandvi, Nizar, Songadh, Surat (Gujarat Region).

⚙️ Installation & Usage

Clone the repository:

git clone [https://github.com/YourUsername/AgriCast360_V2.git](https://github.com/YourUsername/AgriCast360_V2.git)
cd AgriCast360_V2


Install Dependencies:
(Ensure you have Python installed. It is recommended to use a virtual environment).

pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn streamlit scipy


Run the Application:

streamlit run streamlit_app.py


📝 Configuration

API Keys: The project utilizes the Weatherbit API. Ensure your API key is configured if re-running data collection scripts (rough.ipynb).

Directory Structure: The app expects Models/ and Processed_Data/ directories to be present in the root path.

🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Made with ❤️ for Agricultural Analytics