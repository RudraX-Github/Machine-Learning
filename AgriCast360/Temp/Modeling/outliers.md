# Outlier Treatment Log

## Detection Method
IQR (Interquartile Range) method with 1.5×IQR threshold

## Treatment Strategy

### Kept (Domain-Meaningful Extremes)
- **Modal_Price**: Price spikes/drops are real market events
- **precip (mm)**: Heavy rainfall/dry periods are meaningful weather events

### Winsorized (Capped at 5th/95th percentiles)
- **temp (°C)**: Sensor errors or extreme anomalies capped
- **rh (%)**: Humidity extremes capped
- **wind_spd (m/s)**: Wind speed extremes capped

## Detected Outliers

        Column  Outlier_Count  Outlier_Pct  Lower_Bound  Upper_Bound                   Action
   Modal_Price           2080         5.64     -4047.50     13292.50 Keep (domain-meaningful)
     temp (°C)             13         0.04        16.75        38.75                      Cap
   precip (mm)           7417        20.10        -1.50         2.50 Keep (domain-meaningful)
        rh (%)              0         0.00         0.04         1.21                      Cap
wind_spd (m/s)            667         1.81        -0.85         5.95                      Cap