# AgriCast360 Model Results

## Model Performance Summary

                           RMSE       MAE     R2
Random Forest         1876.7543 1227.1182 0.8133
Stacking Ensemble     1971.1938 1299.0605 0.7940
XGBoost               2199.3741 1473.3042 0.7436
Voting Ensemble       2358.4589 1577.9014 0.7051
Gradient Boosting     2358.7951 1568.5473 0.7051
Ridge Regression (L2) 3332.0410 2216.7474 0.4115
Lasso Regression (L1) 3332.4199 2217.0273 0.4113

## Best Model

**Random Forest**
- R²: 0.8133
- RMSE: ₹1876.75
- MAE: ₹1227.12

## Feature Importance (Top 15)

                Feature  Importance
   price_volatility_30d      0.4163
        price_range_pct      0.1888
       season_temp_mean      0.0612
   season_humidity_mean      0.0379
       wind_rolling_30d      0.0370
    price_volatility_7d      0.0340
    season_precip_total      0.0312
     season_precip_mean      0.0265
     precip_rolling_90d      0.0218
       temp_rolling_90d      0.0212
   humidity_rolling_90d      0.0206
    temp_variability_7d      0.0158
              DayOfYear      0.0099
       temp_rolling_30d      0.0087
season_precip_deviation      0.0079

## Weather Impact Analysis

- **Current Weather Impact**: 0.3472
- **Seasonal Weather Impact**: 0.2321

### Key Findings
1. Weather features (current and seasonal) significantly influence commodity prices
2. Last season's weather patterns (90-day lags) provide predictive value
3. Price lag features are strong predictors (market momentum)
4. Ensemble methods provide robust predictions
