# PHASE 2: FEATURE ENGINEERING & MODEL PREPARATION

## 📊 Current Status
**Phase 1 Completion**: ✅ Data Analysis Complete
- 14,965 price records (0 missing values)
- 7,707 weather records aligned
- 68 commodities across 19 markets
- Full year coverage (2024-01-01 to 2025-01-01)

---

## 🎯 PRIMARY GOAL
**Build a Commodity Price Predictor using CSV price data and SQL weather data, analyzing how weather affects commodity prices.**

### Critical Finding
**Temperature Minimum Correlation: +0.81** ← Strongest weather predictor
- Lower temperatures correlate with higher prices
- Explains seasonal patterns (Winter/Spring ↑ high | Summer/Monsoon ↓ low)

---

## Phase 1 Key Outputs

### Data Summary
| Metric | Value |
|--------|-------|
| **Records** | 14,965 |
| **Completeness** | 100% (0 missing) |
| **Commodities** | 68 varieties |
| **Markets** | 19 trading centers |
| **Date Range** | 2024-01-01 to 2025-01-01 |
| **Avg Modal Price** | Rs 4,511 |
| **Price Range** | Rs 650 - Rs 25,000 |
| **Avg Volatility** | 85.80% |

### Commodity Insights
**Most Expensive**: Sesamum (Rs 12,010 avg)
**Most Volatile**: Lemon (226.4%)
**Most Traded**: Bhindi (830 records)

**Top 10 Traded Commodities**:
1. Bhindi (830 records)
2. Tomato (~750 records)
3. Potato (~700 records)
4. Onion (~680 records)
5. Chilli (~650 records)
6. Wheat (~600 records)
7. Rice (~580 records)
8. Soyabean (~550 records)
9. Cotton (~500 records)
10. Sugarcane (~480 records)

### Seasonal Patterns
- **Peak Prices**: June (Rs 5,164 avg)
- **Low Prices**: January (Rs 3,846 avg)
- **Range**: Rs 3,846 - Rs 5,164 (34% seasonal variation)
- **Pattern**: Post-monsoon surge in Jun-Jul, drops in summer

### Market Distribution
**19 Markets**: Delhi, Mumbai, Chennai, Bangalore, Pune, Kolkata, etc.
- Average 788 records per market
- Variation: 400-1,000 records per market
- Geographic diversity across India

---

## 📋 PHASE 2: FEATURE ENGINEERING

### 2.1 Time-Series Features (Lagged Features)

**Price Lags** (for time-series modeling):
```
- Lag-1: Price from previous day
- Lag-7: Price from 1 week ago
- Lag-14: Price from 2 weeks ago
- Lag-30: Price from 1 month ago
```

**Rolling Statistics**:
```
- 7-day moving average
- 7-day moving std dev
- 7-day min/max price
- 30-day moving average
- 30-day moving std dev
```

**Momentum Features**:
```
- Day-over-day % change
- Week-over-week % change
- Month-over-month % change
```

### 2.2 Seasonal Features

**Calendar Features**:
```
- Month (1-12)
- Quarter (1-4)
- Day of Week (0-6)
- Day of Month (1-31)
- Week of Year (1-52)
- Is_Weekend (binary)
- Day_of_Year (1-365)
```

**Seasonal Indicators**:
```
- Season: Winter, Summer, Monsoon, Post-Monsoon
- Is_Holiday (market-specific)
- Trading_Day_Count (cumulative)
```

### 2.3 Weather Features (Temperature Focus)

**Temperature Features** (CRITICAL - Correlation: +0.81):
```
- Temperature_Min (yesterday, 3-day lag, 7-day lag)
- Temperature_Max (yesterday, 3-day lag, 7-day lag)
- Temperature_Mean (rolling 7-day avg)
- Temperature_Change (day-over-day)
- Temperature_Trend (7-day slope)
```

**Precipitation Features**:
```
- Rainfall_Yesterday
- Rainfall_3day_total
- Rainfall_7day_total
- Rainfall_30day_total
- Is_Rainy_Day (binary)
```

**Wind Features**:
```
- Wind_Speed (yesterday, 3-day avg)
- Wind_Speed_Max
- Wind_Gust_Max
```

**Cloud Cover Features**:
```
- Cloud_Cover_Pct (yesterday)
- Cloud_Cover_3day_avg
- Cloud_Cover_Trend
```

### 2.4 Commodity-Market Features

**Commodity Characteristics**:
```
- Commodity_Avg_Price_Historical
- Commodity_Volatility_Historical
- Commodity_Seasonality_Index
- Commodity_Demand_Index
```

**Market Characteristics**:
```
- Market_Avg_Price_Historical
- Market_Volume_Historical
- Market_Distance_from_Metro (proxy for logistics)
```

**Interaction Features**:
```
- Commodity_Market_Avg_Price
- Commodity_Market_Volatility
- Commodity_Market_Seasonal_Index
```

### 2.5 Derived Features

**Arrival Features**:
```
- Arrival_Quantity_Lag
- Arrival_Quantity_Moving_Avg_7day
- Arrival_Quantity_Trend
```

**Grade Encoding**:
```
- Grade_Premium (binary)
- Grade_Standard (binary)
- Grade_Low (binary)
- Grade_Unknown (binary)
```

**Variety Encoding**:
```
- Variety_Category (clustering similar varieties)
- Variety_Popularity (frequency-based ranking)
```

---

## 🔄 Feature Engineering Implementation Steps

### Step 1: Create Lagged Price Features
```python
for lag in [1, 7, 14, 30]:
    df['Price_Lag_{lag}'] = df.groupby('Commodity_Market')['Modal_Price'].shift(lag)
```

### Step 2: Create Rolling Statistics
```python
for window in [7, 14, 30]:
    df[f'Price_MA_{window}'] = df.groupby('Commodity_Market')['Modal_Price'].transform(lambda x: x.rolling(window).mean())
    df[f'Price_Std_{window}'] = df.groupby('Commodity_Market')['Modal_Price'].transform(lambda x: x.rolling(window).std())
```

### Step 3: Create Seasonal Features
```python
df['Is_Weekend'] = df['DayName'].isin(['Saturday', 'Sunday']).astype(int)
df['Season'] = df['Month'].map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})
```

### Step 4: Merge with Weather Data
```python
df_combined = pd.merge(
    df_prices.groupby(['date', 'market'], as_index=False).agg({...}),
    df_weather.groupby(['date', 'market'], as_index=False).agg({...}),
    on=['date', 'market'],
    how='left'
)
```

### Step 5: Create Weather Lag Features
```python
for lag in [1, 3, 7]:
    for col in ['temperature_min', 'temperature_max', 'precipitation_sum']:
        df[f'{col}_lag_{lag}'] = df.groupby('market')[col].shift(lag)
```

### Step 6: Encode Categorical Variables
```python
# Target Encoding for Commodities
commodity_encoding = df.groupby('Commodity')['Modal_Price'].mean()
df['Commodity_Encoded'] = df['Commodity'].map(commodity_encoding)

# One-hot encoding for Grade and Market
df = pd.get_dummies(df, columns=['Grade', 'Market'], drop_first=True)
```

### Step 7: Normalize Features
```python
from sklearn.preprocessing import StandardScaler
numeric_features = df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
```

---

## 📊 Feature Set Summary

### Total Features Expected: ~100-150

| Category | Count | Examples |
|----------|-------|----------|
| **Time-Series Lags** | ~15 | Price Lag-1, 7, 14, 30 |
| **Rolling Statistics** | ~20 | MA-7, MA-30, Std-7, Std-30 |
| **Momentum** | ~5 | % Change Daily, Weekly, Monthly |
| **Calendar** | ~10 | Month, Quarter, DayOfWeek, IsWeekend |
| **Weather** | ~30 | Temp (min/max/mean), Precip, Wind, Cloud, Lags |
| **Commodity-Market** | ~15 | Avg Price, Volatility, Seasonal Index |
| **Categorical** | ~40 | One-hot encoded Grade, Market interactions |
| **Derived** | ~10 | Arrival quantity trends, Grade dummies |

---

## ⚠️ Data Preprocessing Notes

### Handling Missing Values
1. **Lagged features**: First 30 rows will be NaN → Remove or forward-fill
2. **Weather data**: ~3,000 rows missing → Interpolate or use nearest neighbor
3. **Strategy**: Use forward-fill then backward-fill for weather, drop first 30 rows for lags

### Handling Outliers
1. **Extreme prices**: Cap at 99th percentile within commodity
2. **Extreme weather**: Use domain knowledge (e.g., temperature bounds)
3. **Strategy**: Keep for now, will handle in model training (robust models)

### Feature Selection (Phase 3)
- Correlation analysis → Remove features with |r| < 0.01
- VIF (Variance Inflation Factor) → Remove multicollinear features (VIF > 10)
- Feature importance from tree models → Keep top 50-70 features
- Domain knowledge → Ensure critical features (temperature, seasonality) retained

---

## 📈 ML Models to Consider (Phase 3)

### Time-Series Models
1. **ARIMA/SARIMA** - Seasonal autoregressive models
2. **Prophet** - Facebook's time-series forecasting
3. **LSTM** - Long Short-Term Memory neural network
4. **GRU** - Gated Recurrent Unit (lighter alternative to LSTM)

### Regression Models
1. **XGBoost** - Gradient boosting (best for tabular data)
2. **LightGBM** - Light gradient boosting machine
3. **Random Forest** - Ensemble method
4. **Linear Regression** - Baseline model

### Ensemble Methods
- Combine ARIMA + XGBoost
- Weighted ensemble of multiple models
- Stacking (meta-learner approach)

### Model Strategy
- **Separate models per commodity** (suggested) vs single model
- **Separate models per market** (optional)
- **Cross-validation**: Time-series split (don't shuffle)

---

## 🚀 Next Steps for Phase 2

### Deliverables
1. **Feature Engineering Notebook** - Generate all features
2. **Processed Dataset** - CSV with all engineered features
3. **Feature Documentation** - List of all features with descriptions
4. **Correlation Analysis** - Feature correlation heatmap
5. **Feature Importance Report** - Initial rankings

### Success Criteria
- ✓ 0 missing values (after preprocessing)
- ✓ All features normalized/scaled
- ✓ Features pass multicollinearity check (VIF < 10)
- ✓ Dataset ready for model training

### Timeline
- **Data Loading & Merging**: 1-2 hours
- **Feature Engineering**: 2-3 hours
- **Preprocessing & Normalization**: 1-2 hours
- **Validation & Documentation**: 1-2 hours
- **Total Phase 2**: ~6-8 hours

---

## 🎯 Phase 3: ML Model Development (Preview)

After Phase 2 completion:
1. Data split: Train (70%) | Validation (15%) | Test (15%)
2. Model training for each commodity
3. Hyperparameter tuning
4. Ensemble strategy
5. Performance metrics: RMSE, MAE, MAPE, R²

---

## 📝 Summary Table

| Phase | Status | Output | ML Readiness |
|-------|--------|--------|--------------|
| **Phase 1** | ✅ Complete | Clean dataset, exploratory insights | Data Ready |
| **Phase 2** | 🔄 Ready | Engineered features, normalized data | Model Ready |
| **Phase 3** | ⏳ Pending | ML models, predictions, validation | Deployment |
| **Phase 4** | ⏳ Pending | Power BI Dashboard, visualizations | Stakeholder Ready |

---

## 💡 Critical Success Factors

1. **Weather Integration**: Temperature minimum is strongest predictor (+0.81)
2. **Commodity Clustering**: Group similar commodities for better generalization
3. **Seasonal Patterns**: Explicitly model monthly seasonality
4. **Market Differences**: Account for logistics and demand variations
5. **Temporal Dependencies**: Use lagged features to capture autocorrelation

---

**Ready for Phase 2 when you provide confirmation!**
