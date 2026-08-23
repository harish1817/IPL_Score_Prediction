# =============================================================
# IPL LIVE SCORE PREDICTOR
# =============================================================

# PROBLEM
# - Predict final first innings score from any over checkpoint
# - Features: live match situation (runs, wickets, run rate)
# - Type: Regression

# DATA ENGINEERING
# - Transformed ball by ball data → over level snapshots
# - One row per over per match → 21674 rows
# - Cumulative runs, wickets, run rate calculated at each over
# - Powerplay features (overs 1-6) added carefully to avoid leakage
# - Leakage fix: powerplay stats only used after over 6

# HYPOTHESIS TESTING
# - T-Test: bat vs field first scores → p=0.0000 → significant
#   Field first scores higher (168 vs 160) — good pitch = teams prefer chasing
# - ANOVA: venue impact on scores → p=0.0000 → significant
#   Chinnaswamy, Dharamsala = high scoring; South African venues = low scoring

# MODELS & RESULTS
# - Baseline RMSE: 31.78
# - Ridge Regression → RMSE: 20.35, R²: 0.590
# - Decision Tree   → RMSE: 19.33, R²: 0.630
# - Random Forest   → RMSE: 13.72, R²: 0.814
# - Tuned RF        → RMSE: 12.60, R²: 0.843 ← WINNER

# WHY RANDOM FOREST WINS
# - 200 trees, each trained on random rows and features
# - Individual overfitting cancels out when averaged
# - Deep trees okay in RF — variance cancels across trees
# - Single Decision Tree: high variance, overfits easily

# HYPERPARAMETER TUNING
# - RandomizedSearchCV: tries 20 random combinations (not all 72)
# - cv=5: each combination evaluated 5 times
# - n_jobs=-1: uses all CPU cores simultaneously
# - Best params: n_estimators=200, max_depth=None, 
#   min_samples_split=2, max_features='sqrt'
# - Trust the search — manual tuning rarely beats systematic search

# FEATURE IMPORTANCE
# - current_run_rate (0.24) — strongest predictor
# - powerplay_runs (0.11) — sets innings tempo
# - cumulative_runs (0.085) — total so far
# - season (0.073) — IPL scoring has increased over years
# - Teams matter but less than match situation

# BIAS VARIANCE TRADEOFF OBSERVED
# - max_depth=None: Train RMSE 4.77, Test RMSE 12.60 → overfitting
# - max_depth=15:  Train RMSE 16.55, Test RMSE 17.97 → underfitting
# - Sweet spot: tuned model with systematic search

# LIVE PREDICTION
# - RCB vs CSK, Chinnaswamy, over 10, 89/2 → predicted 187
# - Chinnaswamy high scoring ground — prediction makes cricket sense
# =============================================================
