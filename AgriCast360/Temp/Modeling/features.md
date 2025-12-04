
# Engineered Features Documentation

## Weather Features (Current Impact)

### Temperature Features
- **temp_range**: Daily temperature range (max_temp - min_temp) in °C
- **temp_avg**: Average daily temperature ((max_temp + min_temp) / 2) in °C
- **temp_rolling_7d**: 7-day rolling average of temperature in °C
- **temp_rolling_30d**: 30-day rolling average of temperature in °C
- **temp_lag_1d**: Temperature from 1 day ago in °C
- **temp_lag_7d**: Temperature from 7 days ago in °C
- **temp_lag_14d**: Temperature from 14 days ago in °C

### Precipitation Features
- **precip_rolling_7d**: 7-day cumulative rainfall in mm
- **precip_rolling_30d**: 30-day cumulative rainfall in mm
- **precip_lag_1d**: Rainfall from 1 day ago in mm
- **precip_lag_7d**: Rainfall from 7 days ago in mm
- **precip_lag_14d**: Rainfall from 14 days ago in mm
- **precip_cumulative**: Cumulative rainfall since start of year in mm

### Humidity Features
- **humidity_rolling_7d**: 7-day rolling average of relative humidity (%)
- **humidity_rolling_30d**: 30-day rolling average of relative humidity (%)

### Wind Features
- **wind_rolling_7d**: 7-day rolling average of wind speed in m/s
- **wind_rolling_30d**: 30-day rolling average of wind speed in m/s

## Seasonal Features (Last Season Impact)

### Seasonal Aggregates
- **Season**: Indian season (Winter: Dec-Feb, Summer: Mar-May, Monsoon: Jun-Sep, Post-Monsoon: Oct-Nov)
- **season_temp_mean**: Mean temperature for the current season in °C
- **season_temp_std**: Standard deviation of temperature for current season in °C
- **season_precip_total**: Total rainfall for current season in mm
- **season_precip_mean**: Mean daily rainfall for current season in mm
- **season_humidity_mean**: Mean humidity for current season (%)
- **season_wind_mean**: Mean wind speed for current season in m/s

### Seasonal Lag Features (90-day ≈ 1 season)
- **temp_lag_90d**: Temperature from 90 days ago (last season) in °C
- **precip_lag_90d**: Rainfall from 90 days ago (last season) in mm
- **humidity_lag_90d**: Humidity from 90 days ago (last season) (%)
- **temp_rolling_90d**: 90-day rolling average of temperature in °C
- **precip_rolling_90d**: 90-day cumulative rainfall in mm
- **humidity_rolling_90d**: 90-day rolling average of humidity (%)

## Mandi Price Features

### Price Volatility
- **price_volatility_7d**: 7-day rolling standard deviation of modal price (₹)
- **price_volatility_30d**: 30-day rolling standard deviation of modal price (₹)

### Price Moving Averages
- **price_ma_7d**: 7-day moving average of modal price (₹)
- **price_ma_30d**: 30-day moving average of modal price (₹)

### Price Change Rates
- **price_change_1d**: 1-day percentage change in modal price (%)
- **price_change_7d**: 7-day percentage change in modal price (%)

### Price Lag Features
- **price_lag_1d**: Modal price from 1 day ago (₹)
- **price_lag_7d**: Modal price from 7 days ago (₹)
- **price_lag_14d**: Modal price from 14 days ago (₹)

### Price Range Features
- **price_range**: Daily price range (Max_Price - Min_Price) (₹)
- **price_range_pct**: Price range as percentage of modal price (%)

## Time-based Features
- **Month**: Month of year (1-12)
- **Quarter**: Quarter of year (1-4)
- **DayOfYear**: Day of year (1-366)

---
**Total Features Created:**
- Weather (Current): 16 features
- Weather (Seasonal): 13 features  
- Mandi (Price): 11 features
- **Grand Total: 40+ engineered features**
