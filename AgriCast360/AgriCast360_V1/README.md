🌾 AgriCast360_V1

A 360-Degree Agricultural Commodity Market Intelligence & Price Prediction Dashboard.

📖 Overview

AgriCast360 is a comprehensive analytical dashboard designed to provide deep insights into the agricultural commodity market in Gujarat. By integrating executive-level KPIs, detailed price analysis, and meteorological data, this project aims to decipher the correlation between weather patterns and commodity prices.

This repository contains the data analysis (EDA), data dictionary, and the final Power BI dashboard (.pbix).

Primary Goal: Build a Commodity Price Predictor with Weather Integration.

📊 Dashboard Features

The Power BI dashboard consists of 5 core analytical pages providing a "360-degree view":

Executive Overview: High-level KPIs including Average Price (4.51K), Total Records (14.97K), and Market count (19).

Commodity Analysis: detailed trends on top commodities like Sesamum, Kartali, and Turmeric.

Market Intelligence: Visualizes market share trends and dominance of key markets (Mahuva, Mandvi, Nizar, Surat).

Weather Impact: Analyzes correlations between rainfall/temperature and price spikes.

Predictive Analytics: Forecasting models with a 97% accuracy rate.

🔍 Key Insights & Findings

Based on the analysis of 14,965 records across 68 commodities:

Weather Correlation: There is a strong positive correlation (+0.81) between Minimum Temperature and Modal Prices.

Seasonal Peaks: Prices generally peak in June (approx. ₹5,102) and fall to their lowest in January (approx. ₹3,840). This aligns with peak summer temperatures and the onset of the monsoon.

Rainfall Impact: The monsoon season (June–September) aligns with peak prices and cloud cover, suggesting weather plays a critical role in supply chain volatility.

Seasonality: Specific commodities show distinct seasonality (e.g., Ajwan prices peak in May and September).

📂 Repository Structure

├── AgriCast.pbix                        # The final Power BI Dashboard file
├── EDA.ipynb                            # Python Notebook for Exploratory Data Analysis
├── PowerBI.ipynb                        # Guide for Dashboard Design & Strategy
├── Dashboard_Summary.pdf                # Executive Summary and Q&A of the project
├── Summary_Agmarknet_Price_Report_2024.txt  # Data Dictionary & Column Metadata
└── README.md                            # Project Documentation


🛠️ Data Architecture

The analysis is built upon two primary data sources:

Price Data: Agmarknet_Price_Report_2024.csv

Metrics: Modal Price, Min/Max Price, Arrival Date.

Dimensions: State, District, Market, Commodity, Variety, Grade.

Weather Data: market_historical_weather_open-meteo.csv

Metrics: Temperature (Max/Min/Mean), Precipitation, Wind Speed, Cloud Cover.

Source: Open-Meteo API.

Data Quality: The dataset is 100% complete with 0 missing values for the analyzed period.

🚀 Technologies Used

Power BI: For data visualization, DAX measures, and interactive reporting.

Python (Pandas): For Exploratory Data Analysis (EDA) and cleaning.

SQL: Used for querying weather history data.

Google Gemini: Utilized for code assistance and modeling logic.

💻 How to Use

Clone the repository:

git clone [https://github.com/yourusername/AgriCast360_V1.git](https://github.com/yourusername/AgriCast360_V1.git)


View the Analysis: Open EDA.ipynb in Jupyter Notebook or VS Code to see the Python data preprocessing steps.

Run the Dashboard:

Ensure you have Microsoft Power BI Desktop installed.

Open AgriCast.pbix.

🔮 Future Scope

Integration of real-time API data for live price forecasting.

Expansion of the dataset to include markets outside of Gujarat.

Advanced ML modeling deployed directly within the dashboard.

Dashboard created by [Your Name]