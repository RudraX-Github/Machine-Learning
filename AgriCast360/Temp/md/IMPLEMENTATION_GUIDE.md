# AgriCast360 - Improvement Implementation Guide

## Quick Reference: What to Do Next

### 🎯 Current Best Model
- **Model:** Hybrid Ensemble (DNN + LSTM + XGBoost + RandomForest + GradientBoosting)
- **Performance:** R² = 0.7696, RMSE = ₹2093.33, MAE = ₹1425.42
- **Strength:** 82.88% directional accuracy
- **Weakness:** High-price segment errors 2.8x higher

---

## Priority 1: Segment-Specific Models 🔴 HIGH IMPACT

### Why This Works
- High-price commodities (₹6,752-₹31,125) have RMSE ₹3433.62
- Low-price commodities (₹400-₹2,450) have RMSE ₹1304.49
- 2.6x difference means separate models will specialize better

### Implementation Steps

```python
# Step 1: Create training subsets by price quartile
q1, q2, q3, q4 = np.percentile(y_train, [25, 50, 75, 100])
mask_q1 = (y_train >= y_train.min()) & (y_train < q1)
mask_q2 = (y_train >= q1) & (y_train < q2)
mask_q3 = (y_train >= q2) & (y_train < q3)
mask_q4 = (y_train >= q3) & (y_train <= y_train.max())

# Step 2: Train separate XGBoost for each segment
models_segment = {}
for segment_name, mask in [('Q1', mask_q1), ('Q2', mask_q2), 
                           ('Q3', mask_q3), ('Q4', mask_q4)]:
    model = xgb.XGBRegressor(n_estimators=200, max_depth=7, 
                              learning_rate=0.05, random_state=42)
    model.fit(X_train_enhanced[mask], y_train[mask])
    models_segment[segment_name] = model

# Step 3: Predict using segment-specific models
def predict_segment_specific(X_test, y_test_values):
    predictions = np.zeros_like(y_test_values, dtype=float)
    for segment_name, model in models_segment.items():
        # Identify which test samples belong to this segment
        segment_mask = get_segment_mask(y_test_values, segment_name)
        predictions[segment_mask] = model.predict(X_test[segment_mask])
    return predictions

# Step 4: Evaluate
rmse_segment = np.sqrt(mean_squared_error(y_test, predictions))
mae_segment = mean_absolute_error(y_test, predictions)
```

### Expected Improvement
- **Current RMSE:** ₹2093.33
- **After Segment Models:** ₹1465-₹1675 (-30%)
- **Effort:** 2-3 hours coding + testing

### Success Metric
Each segment should have RMSE < overall average

---

## Priority 2: Cyclical & Commodity Features 🔴 HIGH IMPACT

### Why This Works
- Weather is seasonal (patterns repeat yearly)
- Commodities respond differently to seasons
- Currently missing temporal structure

### Implementation Steps

```python
# Step 1: Add cyclical temporal features
from math import pi

# Day of year (0-365)
data['day_of_year'] = data.index.dayofyear
data['day_sin'] = np.sin(2 * pi * data['day_of_year'] / 365)
data['day_cos'] = np.cos(2 * pi * data['day_of_year'] / 365)

# Week of year (0-52)
data['week_of_year'] = data.index.isocalendar().week
data['week_sin'] = np.sin(2 * pi * data['week_of_year'] / 52)
data['week_cos'] = np.cos(2 * pi * data['week_of_year'] / 52)

# Month (useful for seasonal commodities)
data['month'] = data.index.month
data['month_sin'] = np.sin(2 * pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * pi * data['month'] / 12)

# Step 2: Add commodity-specific seasonal dummies
commodities = data['commodity'].unique()
for commodity in commodities:
    data[f'is_{commodity.lower()}'] = (data['commodity'] == commodity).astype(int)

# Step 3: Create commodity-weather interactions
# For each commodity, multiply by weather features
for commodity in commodities:
    com_mask = data['commodity'] == commodity
    data[f'{commodity.lower()}_temp_interaction'] = \
        (com_mask.astype(int)) * data['temperature']
    data[f'{commodity.lower()}_humidity_interaction'] = \
        (com_mask.astype(int)) * data['humidity']
    data[f'{commodity.lower()}_rainfall_interaction'] = \
        (com_mask.astype(int)) * data['rainfall']

# Step 4: Update feature list
new_features = [col for col in data.columns if col not in original_features]
enhanced_features.extend(new_features)

print(f"Added {len(new_features)} new features")
```

### Expected Improvement
- **Current RMSE:** ₹2093.33
- **After Cyclical Features:** ₹1575-₹1778 (-20%)
- **Cumulative with Priority 1:** ₹1172 (-44%)
- **Effort:** 1-2 hours coding + validation

### Success Metric
- Cyclical features should have non-zero importance in XGBoost
- Commodity interactions should show up in top 20 features

---

## Priority 3: Hyperparameter Optimization 🟡 MEDIUM IMPACT

### Why This Works
- Current hyperparameters are manually set/defaults
- Different segments might need different parameters
- Ensemble weights are fixed (20%, 25%, etc.)

### Installation & Setup

```bash
# Install Optuna for Bayesian optimization
pip install optuna

# Optional: Install for better progress tracking
pip install optuna-integration-skopt
```

### Implementation Steps

```python
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_val_score

# Step 1: Define objective function
def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    
    # Create model with suggested parameters
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=42,
        n_jobs=-1
    )
    
    # Evaluate with cross-validation
    scores = cross_val_score(
        model, X_train_enhanced, y_train,
        cv=5, scoring='neg_mean_squared_error'
    )
    rmse = np.sqrt(-scores.mean())
    
    return rmse

# Step 2: Run optimization
study = optuna.create_study(
    direction='minimize',
    pruner=MedianPruner()
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

# Step 3: Get best parameters
best_params = study.best_params
print(f"Best RMSE: ₹{study.best_value:.2f}")
print(f"Best parameters: {best_params}")

# Step 4: Train final model with best parameters
best_model = xgb.XGBRegressor(**best_params, random_state=42)
best_model.fit(X_train_enhanced, y_train)

# Step 5: Optimize ensemble weights
def ensemble_objective(trial):
    w_dnn = trial.suggest_float('w_dnn', 0, 1)
    w_lstm = trial.suggest_float('w_lstm', 0, 1)
    w_xgb = trial.suggest_float('w_xgb', 0, 1)
    w_rf = trial.suggest_float('w_rf', 0, 1)
    w_gb = trial.suggest_float('w_gb', 0, 1)
    
    # Normalize weights to sum to 1
    total = w_dnn + w_lstm + w_xgb + w_rf + w_gb
    weights = {
        'dnn': w_dnn/total,
        'lstm': w_lstm/total,
        'xgb': w_xgb/total,
        'rf': w_rf/total,
        'gb': w_gb/total
    }
    
    # Create ensemble with these weights
    ensemble_pred = (weights['dnn'] * y_pred_dnn +
                    weights['lstm'] * y_pred_lstm +
                    weights['xgb'] * y_pred_xgb +
                    weights['rf'] * y_pred_rf +
                    weights['gb'] * y_pred_gb)
    
    rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    return rmse

study_weights = optuna.create_study(direction='minimize')
study_weights.optimize(ensemble_objective, n_trials=30)

optimal_weights = study_weights.best_params
print(f"Optimal ensemble weights: {optimal_weights}")
```

### Expected Improvement
- **Current RMSE:** ₹2093.33
- **After Hyperparameter Optimization:** ₹1781-1952 (-10-15%)
- **Cumulative with Priorities 1-2:** ₹870-1050 (-50-60%)
- **Effort:** 3-4 hours (mostly waiting for optimization)
- **Can run in background:** Yes! Set `show_progress_bar=True`

### Success Metrics
- Best trial RMSE should be lower than random search
- Ensemble weights should be noticeably different from equal weighting
- Cross-validation score should be better than single model

---

## Alternative: Priority 4: AutoML Approaches

### Why This Works
- LightGBM/CatBoost often beat XGBoost on tabular data
- AutoML finds optimal pipeline automatically
- Less manual tuning needed

### Quick Implementation

```python
# Option A: LightGBM
import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=7,
    random_state=42,
    n_jobs=-1
)
lgb_model.fit(X_train_enhanced, y_train)
y_pred_lgb = lgb_model.predict(X_test_enhanced)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))

# Option B: CatBoost
from catboost import CatBoostRegressor

cat_model = CatBoostRegressor(
    iterations=200,
    learning_rate=0.05,
    depth=7,
    random_state=42,
    verbose=0  # Suppress output
)
cat_model.fit(X_train_enhanced, y_train)
y_pred_cat = cat_model.predict(X_test_enhanced)
rmse_cat = np.sqrt(mean_squared_error(y_test, y_pred_cat))

# Option C: TPOT AutoML (finds best pipeline)
from tpot import TPOTRegressor

tpot = TPOTRegressor(
    generations=5,
    population_size=20,
    verbosity=2,
    random_state=42,
    n_jobs=-1
)
tpot.fit(X_train_enhanced, y_train)
y_pred_tpot = tpot.predict(X_test_enhanced)
rmse_tpot = np.sqrt(mean_squared_error(y_test, y_pred_tpot))
```

### Expected Improvement
- **Current RMSE:** ₹2093.33
- **After AutoML:** ₹1465-1675 (-20-30%)
- **Effort:** 1-2 hours (mostly waiting for search)
- **Advantage:** No manual hyperparameter tuning needed

---

## Priority 5: Consider Alternative Target

### Current Approach: Absolute RMSE
- Target: ₹100 RMSE
- Problem: Extremely difficult for high-price items

### Alternative Approach: Percentage Error
- Target: 5% MAPE (currently 34.71%)
- Benefit: Treats ₹1000 and ₹10000 items equally
- Much more achievable

```python
# Convert to percentage error target
def percentage_rmse(y_true, y_pred):
    pct_errors = np.abs((y_true - y_pred) / y_true) * 100
    return np.sqrt(np.mean(pct_errors ** 2))

# Or use MAPE directly
def mape_loss(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# This would make 5-10% MAPE target much more reasonable
pct_rmse = percentage_rmse(y_test, y_pred_hybrid)
print(f"Percentage RMSE: {pct_rmse:.2f}%")
```

---

## Recommended Timeline

### Week 1: Segment-Specific Models
- Implement segment analysis
- Train 4 separate models
- **Target RMSE:** ₹1500-1700

### Week 2: Feature Engineering
- Add cyclical features
- Create commodity interactions
- Re-train all models
- **Target RMSE:** ₹1200-1400

### Week 3: Hyperparameter Optimization
- Set up Optuna
- Optimize all models (background process)
- Find optimal ensemble weights
- **Target RMSE:** ₹1000-1200

### Week 4: Try Alternatives
- Test LightGBM/CatBoost
- Evaluate TPOT AutoML
- Compare results
- **Target RMSE:** ₹900-1100

### Ongoing: Monitor & Iterate
- Track metrics across improvements
- Document what works/doesn't work
- Build on successful approaches

---

## Quick Wins (Can Do Today)

### 1. Switch to Percentage Error Metric
- Change target from ₹100 to 5% MAPE
- Much more achievable and realistic
- Takes 5 minutes to implement

### 2. Create Segment Analysis Visualization
- Already done! See `segment_analysis.png`
- Use to identify which segments need work

### 3. Test LightGBM
- Install: `pip install lightgbm`
- Train: 10 lines of code
- Compare RMSE
- Takes 1 hour

---

## Success Criteria

| Priority | Current | Target | Metric |
|---|---|---|---|
| 1. Segment Models | ₹2093 | ₹1500-1700 | RMSE |
| 2. Cyclical Features | ₹1600 | ₹1200-1400 | RMSE |
| 3. Hyperparameter Optimization | ₹1300 | ₹1000-1200 | RMSE |
| 4. AutoML | ₹1100 | ₹900-1100 | RMSE |
| Alternative | 34.71% | <10% | MAPE |

---

## Common Pitfalls to Avoid

❌ **Don't:** Optimize ensemble weights without retraining models  
✅ **Do:** Retrain each model independently before ensemble

❌ **Don't:** Use same features for all commodities  
✅ **Do:** Create commodity-specific features

❌ **Don't:** Skip cross-validation for hyperparameter optimization  
✅ **Do:** Always use CV to prevent overfitting

❌ **Don't:** Treat high-price and low-price items equally  
✅ **Do:** Consider segment-specific or percentage-based targets

---

## Resources & References

- **Optuna Documentation:** https://optuna.readthedocs.io/
- **LightGBM:** https://lightgbm.readthedocs.io/
- **CatBoost:** https://catboost.ai/
- **TPOT:** http://epistasislab.github.io/tpot/
- **Time Series Features:** https://en.wikipedia.org/wiki/Cyclical_coordinates

---

**Last Updated:** 2024  
**Next Phase:** Phase 3 - Implementation of top priorities  
**Contact:** Refer to main PHASE2_COMPLETION_REPORT.md for detailed analysis
