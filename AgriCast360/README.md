🌾 AgriCast360

Intelligent Crop Price, Yield & Supply Prediction System

AgriCast360 is a comprehensive agricultural intelligence solution designed to analyze market trends and weather patterns to predict crop prices and yields. The project has evolved through two major versions, transitioning from a robust Business Intelligence dashboard to an advanced Machine Learning prediction application.

🚀 Version Evolution

AgriCast360 V2: Machine Learning & Web App

Focus: Real-time Prediction & Advanced Modeling
The latest iteration is a Python-based machine learning solution that integrates historical Mandi data with granular weather metrics. It features a Streamlit dashboard for real-time interaction.

Core Capability: Evaluates 9 ML models to route predictions through the most accurate model based on commodity type.

Best Model: Random Forest Regressor (MAE: ~313.18).

Key Features:

Advanced Feature Engineering: Temporal features, lag correlations, seasonal decomposition, and weather-price interaction terms.

Interactive Dashboard: User interface to select commodities and markets (e.g., Bardoli, Surat, Mahuva).

Robust Pipeline: Includes Skewness correction, Kurtosis control, and various scaling techniques.

AgriCast360 V1: Intelligence Dashboard

Focus: Data Visualization & Market Intelligence
The foundational version is a Power BI dashboard providing a "360-degree view" of the agricultural commodity market in Gujarat.

Core Capability: Executive-level KPIs and correlation analysis between weather and prices.

Key Insights: Identified a strong positive correlation (+0.81) between Minimum Temperature and Modal Prices.

Dashboard Structure: 5 core pages covering Executive Overview, Commodity Analysis, Market Intelligence, Weather Impact, and Predictive Analytics.

🛠️ Tech Stack & Architecture

The ecosystem utilizes different technologies across its versions:

Category

Technologies Used

Interfaces

Streamlit (V2), Microsoft Power BI (V1)

Languages

Python 3.11+ (Pandas, NumPy, SciPy), SQL

Machine Learning

Scikit-learn, XGBoost, TensorFlow/Keras

Data Processing

Jupyter Notebooks (EDA, Feature Engineering, Preprocessing)

APIs (Weather)

Weatherbit API (V2), Open-Meteo API (V1)

Assistance

Google Gemini (Code assistance & modeling logic)

📊 Data Sources & Insights

Data Sources:

Markets (Mandi): Agmarknet Price Reports (Modal Price, Arrival Date, Variety, Grade).

Weather: Historical data including Temperature, Precipitation, Wind Speed, and Cloud Cover.

Key Findings:

Seasonality: Prices generally peak in June (approx. ₹5,102) and hit lows in January (approx. ₹3,840), aligning with monsoon onsets.

Geography: Detailed analysis performed on Gujarat region markets including Bardoli, Kosamba, Mahuva, Mandvi, Nizar, Songadh, and Surat.

Commodity Trends: Analyzed 68 commodities; specific trends found in Sesamum, Kartali, and Turmeric.

📂 Project Structure

The repository supports both the analysis and deployment workflows:

├── AgriCast.pbix                        # V1: Final Power BI Dashboard
├── streamlit_app.py                     # V2: Production-ready ML Application
├── EDA.ipynb                            # Data Analysis & Cleaning
├── Time_Series_Analysis.ipynb           # V2: Seasonality & Trend Analysis
├── Features_building.ipynb              # V2: Feature Construction
├── Feature_Preprocessing.ipynb          # V2: Scaling & Transformation
├── Model_building.ipynb                 # V2: Training 9 Models
├── AgriCast_Full_Report.html            # V2: Model Performance Report
└── Agmarknet_Price_Report_2024.csv      # Source Data


⚙️ Installation & Usage

Running AgriCast360 V2 (Python App)

Clone the repository:

git clone [https://github.com/YourUsername/AgriCast360.git](https://github.com/YourUsername/AgriCast360.git)
cd AgriCast360


Install Dependencies:

pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn streamlit scipy


Run the Streamlit App:

streamlit run streamlit_app.py
