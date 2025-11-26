# 🚀 PHASE 2 ACTION PLAN: FEATURE ENGINEERING

## 📋 QUICK START GUIDE

### Current Status
✅ **Phase 1**: Data Analysis Complete  
📊 **Input**: Price_Data_Processed.csv (14,965 records)  
🎯 **Goal**: Engineer 100-150 features for ML model training  
⏱️ **Timeline**: 7-12 hours (estimated)

---

## 🔧 PHASE 2 IMPLEMENTATION ROADMAP

### STEP 1: Data Loading & Merging (1-2 hours)
**Objective**: Load processed data and weather data, merge by date + market

**Tasks**:
```python
1. Load Price_Data_Processed.csv
2. Load df_weather from MySQL (or pre-processed weather CSV)
3. Create Commodity_Market composite key
4. Time-series groupby: Group by (Commodity, Market, Date)
5. Weather groupby: Group by (Market, Date) or (Date)
6. Merge price with weather on date + market
7. Validate merge success rate (~95%+)
8. Handle missing weather data (forward-fill or interpolate)
```

**Output**:
- `df_combined`: Price data merged with weather (~14,000 rows after merge)
- `merge_report.txt`: Merge statistics

---

### STEP 2: Lagged Price Features (2 hours)
**Objective**: Create time-series lag features (previous days' prices)

**Code Template**:
```python
# Lag features for each (Commodity, Market) pair
for lag in [1, 7, 14, 30]:
    df_combined[f'Price_Lag_{lag}'] = df_combined.groupby(['Commodity', 'Market'])['Modal_Price'].shift(lag)
    
# Similar for Min and Max prices
for lag in [1, 7]:
    df_combined[f'Min_Price_Lag_{lag}'] = df_combined.groupby(['Commodity', 'Market'])['Min_Price'].shift(lag)
    df_combined[f'Max_Price_Lag_{lag}'] = df_combined.groupby(['Commodity', 'Market'])['Max_Price'].shift(lag)
```

**Features Generated**:
- Price_Lag_1, Price_Lag_7, Price_Lag_14, Price_Lag_30 (4 features)
- Min_Price_Lag_1, Min_Price_Lag_7 (2 features)
- Max_Price_Lag_1, Max_Price_Lag_7 (2 features)
- **Total**: 8 features

**Notes**:
- First 30 rows will have NaN values (no previous data)
- Option 1: Drop these rows (lose 30 records)
- Option 2: Forward-fill from historical data
- **Recommendation**: Drop first 30 rows (cleaner data)

---

### STEP 3: Rolling Statistics (2 hours)
**Objective**: Create moving averages and volatility features

**Code Template**:
```python
# Rolling averages for each (Commodity, Market) pair
for window in [7, 14, 30]:
    df_combined[f'Price_MA_{window}'] = df_combined.groupby(['Commodity', 'Market'])['Modal_Price'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df_combined[f'Price_Std_{window}'] = df_combined.groupby(['Commodity', 'Market'])['Modal_Price'].transform(lambda x: x.rolling(window, min_periods=1).std())
    df_combined[f'Price_Max_{window}'] = df_combined.groupby(['Commodity', 'Market'])['Modal_Price'].transform(lambda x: x.rolling(window, min_periods=1).max())
    df_combined[f'Price_Min_{window}'] = df_combined.groupby(['Commodity', 'Market'])['Modal_Price'].transform(lambda x: x.rolling(window, min_periods=1).min())
```

**Features Generated**:
- Price_MA_7, Price_MA_14, Price_MA_30 (3 features)
- Price_Std_7, Price_Std_14, Price_Std_30 (3 features)
- Price_Max_7, Price_Max_14, Price_Max_30 (3 features)
- Price_Min_7, Price_Min_14, Price_Min_30 (3 features)
- **Total**: 12 features

---

### STEP 4: Momentum & Trend Features (1 hour)
**Objective**: Create percentage change and trend indicators

**Code Template**:
```python
# Percentage changes
df_combined['Price_Change_1d_Pct'] = df_combined.groupby(['Commodity', 'Market'])['Modal_Price'].pct_change()
df_combined['Price_Change_7d_Pct'] = df_combined.groupby(['Commodity', 'Market'])['Modal_Price'].pct_change(7)
df_combined['Price_Change_30d_Pct'] = df_combined.groupby(['Commodity', 'Market'])['Modal_Price'].pct_change(30)

# Trend (slope calculation)
from scipy import stats
for commodity in df_combined['Commodity'].unique():
    for market in df_combined['Market'].unique():
        mask = (df_combined['Commodity'] == commodity) & (df_combined['Market'] == market)
        x = np.arange(len(df_combined[mask]))
        y = df_combined[mask]['Modal_Price'].values
        if len(y) > 7:
            slope, _ = stats.linregress(x[-7:], y[-7:])[:2]
            df_combined.loc[mask, 'Price_Trend_7d'] = slope
```

**Features Generated**:
- Price_Change_1d_Pct, Price_Change_7d_Pct, Price_Change_30d_Pct (3 features)
- Price_Trend_7d (1 feature)
- **Total**: 4 features

---

### STEP 5: Calendar & Seasonal Features (1 hour)
**Objective**: Extract temporal and seasonal indicators

**Code Template**:
```python
# Already extracted in Phase 1, verify and add more
df_combined['Month'] = df_combined['Arrival_Date'].dt.month
df_combined['Quarter'] = df_combined['Arrival_Date'].dt.quarter
df_combined['DayOfWeek'] = df_combined['Arrival_Date'].dt.dayofweek  # 0=Monday, 6=Sunday
df_combined['DayOfYear'] = df_combined['Arrival_Date'].dt.dayofyear
df_combined['WeekOfYear'] = df_combined['Arrival_Date'].dt.isocalendar().week
df_combined['IsWeekend'] = (df_combined['DayOfWeek'] >= 5).astype(int)

# Seasonal indicators (India-specific)
def get_season(month):
    if month in [12, 1, 2]: return 1  # Winter
    elif month in [3, 4, 5]: return 2  # Summer
    elif month in [6, 7, 8, 9]: return 3  # Monsoon
    else: return 4  # Post-Monsoon

df_combined['Season'] = df_combined['Month'].apply(get_season)

# Cyclical encoding (sine/cosine for circular features)
df_combined['Month_Sin'] = np.sin(2 * np.pi * df_combined['Month'] / 12)
df_combined['Month_Cos'] = np.cos(2 * np.pi * df_combined['Month'] / 12)
df_combined['DayOfWeek_Sin'] = np.sin(2 * np.pi * df_combined['DayOfWeek'] / 7)
df_combined['DayOfWeek_Cos'] = np.cos(2 * np.pi * df_combined['DayOfWeek'] / 7)
```

**Features Generated**:
- Month, Quarter, DayOfWeek, DayOfYear, WeekOfYear, IsWeekend (6 features)
- Season (1 feature)
- Month_Sin, Month_Cos, DayOfWeek_Sin, DayOfWeek_Cos (4 features)
- **Total**: 11 features

---

### STEP 6: Weather & Weather Lag Features (2-3 hours)
**Objective**: Engineer weather features with lags (CRITICAL: +0.81 correlation)

**Code Template**:
```python
# Temperature features (MOST IMPORTANT)
for lag in [0, 1, 3, 7]:
    if lag == 0:
        df_combined[f'Temp_Min'] = df_combined['temperature_min']
        df_combined[f'Temp_Max'] = df_combined['temperature_max']
        df_combined[f'Temp_Mean'] = df_combined['temperature_mean']
    else:
        df_combined[f'Temp_Min_Lag_{lag}'] = df_combined.groupby('Market')['temperature_min'].shift(lag)
        df_combined[f'Temp_Max_Lag_{lag}'] = df_combined.groupby('Market')['temperature_max'].shift(lag)
        df_combined[f'Temp_Mean_Lag_{lag}'] = df_combined.groupby('Market')['temperature_mean'].shift(lag)

# Temperature rolling statistics
for window in [7, 14, 30]:
    df_combined[f'Temp_Min_MA_{window}'] = df_combined.groupby('Market')['temperature_min'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df_combined[f'Temp_Range_{window}'] = (
        df_combined.groupby('Market')['temperature_max'].transform(lambda x: x.rolling(window, min_periods=1).max()) -
        df_combined.groupby('Market')['temperature_min'].transform(lambda x: x.rolling(window, min_periods=1).min())
    )

# Precipitation features
for lag in [0, 1, 3, 7]:
    if lag == 0:
        df_combined['Rainfall'] = df_combined['precipitation_sum']
        df_combined['Is_Rainy'] = (df_combined['precipitation_sum'] > 1).astype(int)
    else:
        df_combined[f'Rainfall_Lag_{lag}'] = df_combined.groupby('Market')['precipitation_sum'].shift(lag)

# Rolling precipitation
for window in [7, 30]:
    df_combined[f'Rainfall_Sum_{window}'] = df_combined.groupby('Market')['precipitation_sum'].transform(lambda x: x.rolling(window, min_periods=1).sum())

# Wind features
df_combined['Wind_Speed'] = df_combined['wind_speed_max']
df_combined['Wind_Speed_Lag_1'] = df_combined.groupby('Market')['wind_speed_max'].shift(1)
df_combined[f'Wind_Speed_MA_7'] = df_combined.groupby('Market')['wind_speed_max'].transform(lambda x: x.rolling(7, min_periods=1).mean())

# Cloud cover
df_combined['Cloud_Cover'] = df_combined['cloud_cover']
df_combined['Cloud_Cover_Lag_1'] = df_combined.groupby('Market')['cloud_cover'].shift(1)
```

**Features Generated**:
- Temperature (base + 3 lags): 12 features
- Temperature rolling (3 windows × 2 metrics): 6 features
- Precipitation (base + 3 lags + rolling): 8 features
- Wind features: 3 features
- Cloud cover: 2 features
- **Total**: ~31 weather features

---

### STEP 7: Commodity-Market Features (1 hour)
**Objective**: Create aggregated commodity and market characteristics

**Code Template**:
```python
# Historical commodity statistics (computed on training data only!)
commodity_stats = df_combined.groupby('Commodity').agg({
    'Modal_Price': ['mean', 'std', 'min', 'max'],
    'Price_Volatility_%': 'mean',
    'Arrival': 'mean'
}).reset_index()
commodity_stats.columns = ['Commodity', 'Commodity_Avg_Price', 'Commodity_Std_Price', 
                           'Commodity_Min_Price', 'Commodity_Max_Price', 'Commodity_Volatility', 'Commodity_Avg_Arrival']

df_combined = df_combined.merge(commodity_stats, on='Commodity', how='left')

# Historical market statistics
market_stats = df_combined.groupby('Market').agg({
    'Modal_Price': ['mean', 'std'],
    'Commodity': 'nunique'
}).reset_index()
market_stats.columns = ['Market', 'Market_Avg_Price', 'Market_Std_Price', 'Market_Commodity_Count']

df_combined = df_combined.merge(market_stats, on='Market', how='left')

# Commodity-Market interaction
commodity_market_stats = df_combined.groupby(['Commodity', 'Market']).agg({
    'Modal_Price': ['mean', 'std', 'count']
}).reset_index()
commodity_market_stats.columns = ['Commodity', 'Market', 'Comm_Market_Avg_Price', 'Comm_Market_Std_Price', 'Comm_Market_Records']

df_combined = df_combined.merge(commodity_market_stats, on=['Commodity', 'Market'], how='left')

# Seasonal commodity premium (month-commodity average)
seasonal_premium = df_combined.groupby(['Month', 'Commodity'])['Modal_Price'].mean().reset_index()
seasonal_premium.columns = ['Month', 'Commodity', 'Month_Commodity_Avg_Price']

df_combined = df_combined.merge(seasonal_premium, on=['Month', 'Commodity'], how='left')
```

**Features Generated**:
- Commodity stats: 5 features (avg, std, min, max, volatility)
- Market stats: 3 features (avg, std, commodity count)
- Commodity-Market interaction: 3 features
- Seasonal premium: 1 feature
- **Total**: 12 features

---

### STEP 8: Categorical Encoding (1.5 hours)
**Objective**: Encode categorical variables for ML models

**Code Template**:
```python
# Option 1: Target Encoding (Recommended for trees)
from category_encoders import TargetEncoder

target_encoder = TargetEncoder(cols=['Commodity', 'Market', 'Grade', 'Variety'])
df_combined_encoded = target_encoder.fit_transform(df_combined[['Commodity', 'Market', 'Grade', 'Variety']], df_combined['Modal_Price'])
df_combined[['Commodity_Encoded', 'Market_Encoded', 'Grade_Encoded', 'Variety_Encoded']] = df_combined_encoded

# Option 2: One-Hot Encoding (Alternative)
df_combined = pd.get_dummies(df_combined, columns=['Grade', 'Variety'], drop_first=True)

# Keep DayName as category but encode
df_combined['DayName_Encoded'] = pd.Categorical(df_combined['DayName'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).codes
```

**Features Generated**:
- Grade (one-hot): 3 features (assuming 4 grades)
- Variety (one-hot): ~15-20 features (reduced by dropping first)
- DayName: 1 encoded feature
- Commodity_Encoded, Market_Encoded (target encoding): 2 features
- **Total**: ~20-25 features

---

### STEP 9: Feature Validation & Normalization (1 hour)
**Objective**: Clean features and normalize for ML

**Code Template**:
```python
# Drop rows with NaN from lagged features (first 30 rows)
df_combined = df_combined.iloc[30:].reset_index(drop=True)

# Check for remaining NaNs
nan_counts = df_combined.isnull().sum()
print(f"Remaining NaN values:\n{nan_counts[nan_counts > 0]}")

# Fill any remaining NaNs (weather data gaps)
df_combined = df_combined.fillna(method='ffill').fillna(method='bfill')

# Remove infinite values
df_combined = df_combined.replace([np.inf, -np.inf], np.nan)
df_combined = df_combined.fillna(df_combined.mean())

# Separate features and target
feature_cols = [col for col in df_combined.columns if col != 'Modal_Price' and col != 'Arrival_Date']
X = df_combined[feature_cols]
y = df_combined['Modal_Price']

# Normalize numeric features
from sklearn.preprocessing import StandardScaler
numeric_features = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Save scaler for inference
import pickle
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print(f"\nFinal Dataset Shape: {X.shape}")
print(f"Total Features: {X.shape[1]}")
print(f"Sample of Features: {list(X.columns[:20])}")
```

**Output**:
- `df_combined_final.csv`: Dataset with all engineered features (~13,900 rows, 100-150 columns)
- `scaler.pkl`: Fitted StandardScaler for inference
- `features.txt`: List of all features with descriptions

---

### STEP 10: Documentation & Reporting (1 hour)
**Objective**: Create feature documentation for team

**Deliverables**:
```
1. Feature_Engineering_Report.md
   - Summary of all engineered features
   - Feature statistics (mean, std, min, max)
   - Missing value rates
   - Correlation with target (Modal_Price)

2. Feature_List.csv
   - All feature names
   - Data types
   - Source (price/weather/temporal/encoded)
   - Descriptions

3. Correlation_Heatmap.png
   - Heatmap of all features vs Modal_Price
   - Heatmap of feature-to-feature correlations
   - Identify multicollinearity

4. Missing_Value_Analysis.csv
   - Which features have missing values
   - Percentage missing
   - Strategy used to handle
```

---

## 📊 FEATURE SUMMARY TABLE

| Category | Count | Examples | Status |
|----------|-------|----------|--------|
| **Price Lags** | 8 | Price_Lag_1, 7, 14, 30 | ✓ Implement |
| **Rolling Stats** | 12 | Price_MA_7, Std_14, Max_30 | ✓ Implement |
| **Momentum** | 4 | Price_Change_1d_Pct | ✓ Implement |
| **Calendar** | 11 | Month, Quarter, Season, Sin/Cos | ✓ Implement |
| **Weather Base** | 6 | Temp_Min, Max, Mean, Rainfall | ✓ Implement |
| **Weather Lags** | 12 | Temp_Min_Lag_1, 3, 7 | ✓ Implement |
| **Weather Rolling** | 8 | Temp_Min_MA_7, Rainfall_Sum_30 | ✓ Implement |
| **Commodity Stats** | 5 | Commodity_Avg_Price, Volatility | ✓ Implement |
| **Market Stats** | 3 | Market_Avg_Price, Count | ✓ Implement |
| **Interactions** | 4 | Comm_Market_Avg_Price, Premium | ✓ Implement |
| **Categorical** | 20-25 | Grade_A, Grade_B, Variety_*, DayName | ✓ Implement |
| **TOTAL** | **93-98** | Various | ✓ Ready |

---

## ⏱️ PHASE 2 TIMELINE

| Step | Duration | Deliverable |
|------|----------|------------|
| 1. Data Merge | 1-2 hrs | df_combined.csv |
| 2. Lagged Features | 2 hrs | +8 features |
| 3. Rolling Stats | 2 hrs | +12 features |
| 4. Momentum | 1 hr | +4 features |
| 5. Calendar & Seasonal | 1 hr | +11 features |
| 6. Weather Features | 2-3 hrs | +31 features |
| 7. Commodity-Market | 1 hr | +12 features |
| 8. Categorical Encoding | 1.5 hrs | +20-25 features |
| 9. Validation & Normalization | 1 hr | Final dataset |
| 10. Documentation | 1 hr | Reports & lists |
| **TOTAL** | **7-12 hrs** | **~100-150 features** |

---

## ✅ PHASE 2 SUCCESS CRITERIA

- ✓ All 93-98 planned features engineered
- ✓ 0 missing values in final dataset
- ✓ No infinite values
- ✓ All numeric features normalized (mean=0, std=1)
- ✓ Categorical features properly encoded
- ✓ Temporal structure preserved (no shuffling)
- ✓ Feature documentation complete
- ✓ Dataset saved as CSV
- ✓ Scaler pickled for inference

---

## 🚀 READY TO START PHASE 2?

When ready, create a new notebook `PHASE_2_Feature_Engineering.ipynb` and follow the steps above.

**Key Requirements**:
1. Load `Price_Data_Processed.csv` from Phase 1
2. Load weather data from MySQL
3. Implement 10 engineering steps sequentially
4. Validate at each step
5. Export final engineered dataset

---

**Estimated Completion**: 7-12 hours  
**Next Phase**: Phase 3 - ML Model Development (after this is complete)

**All instructions and code templates provided above. Ready to execute! 🚀**
