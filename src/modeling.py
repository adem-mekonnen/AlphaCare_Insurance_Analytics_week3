
"""
Module: modeling.py
Description:
    This script trains and evaluates multiple regression models (Linear, RF, XGBoost)
    to predict insurance claim severity. It includes SHAP analysis for interpretability.

Author: Adem Mekonnen
Date: Dec 2025
"""
# ... rest of your code ...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Try Importing XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not found. Will use GradientBoosting instead.")

# Ensure visuals folder exists
os.makedirs('visuals', exist_ok=True)

# 1. LOAD DATA
print("Loading Data...")
try:
    # Try pipe separator first (Common for this dataset)
    df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|', low_memory=False)
except FileNotFoundError:
    # Fallback
    df = pd.read_csv('data/insurance_data.csv', sep='|', low_memory=False)

# --- CRITICAL FIX: Clean Column Names ---
# Remove empty spaces from column names (e.g. "Make " -> "Make")
df.columns = df.columns.str.strip()
print(f"Columns available: {list(df.columns)}")
# ----------------------------------------

# 2. PREPROCESSING
print("Preprocessing Data...")
target = 'TotalClaims'

# Ensure Target Exists
if target not in df.columns:
    print(f"ERROR: Target column '{target}' not found. Please check your data.")
    exit()

# Define Desired Features
desired_features_num = ['TotalPremium', 'CalculatedPremiumPerTerm']
desired_features_cat = ['Province', 'VehicleType', 'Gender', 'Make']

# Filter to ONLY features that actually exist in the dataframe
features_num = [c for c in desired_features_num if c in df.columns]
features_cat = [c for c in desired_features_cat if c in df.columns]

print(f"-> Modeling with Numeric: {features_num}")
print(f"-> Modeling with Categorical: {features_cat}")

if not features_num and not features_cat:
    print("ERROR: None of the requested feature columns were found!")
    exit()

# Clean Numeric Data
for c in features_num + [target]:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

X = df[features_num + features_cat]
y = df[target]

# Handle Missing Values in Features
if features_num:
    X[features_num] = X[features_num].fillna(0)
if features_cat:
    X[features_cat] = X[features_cat].fillna('Unknown')

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Preprocessing Pipeline
transformers = []
if features_num:
    transformers.append(('num', 'passthrough', features_num))
if features_cat:
    transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_cat))

preprocessor = ColumnTransformer(transformers=transformers)

# 3. DEFINE MODELS
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42), # Reduced estimators for speed
}

if HAS_XGB:
    models["XGBoost"] = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, max_depth=5, random_state=42)
else:
    models["Gradient Boosting"] = GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)

# 4. TRAIN & EVALUATE
print("\n--- MODEL COMPARISON RESULTS ---")
results = []
best_model_name = ""
best_model_pipeline = None
best_r2 = -float('inf')

for name, model in models.items():
    print(f"Training {name}...")
    
    clf = Pipeline(steps=[
        ('prep', preprocessor),
        ('model', model)
    ])
    
    try:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        results.append({'Model': name, 'RMSE': rmse, 'R2': r2})
        print(f"   -> RMSE: {rmse:.2f} | R2: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model_pipeline = clf
            
    except Exception as e:
        print(f"   -> Failed to train {name}: {e}")

# 5. SUMMARY
results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
print("\nFINAL LEADERBOARD:")
print(results_df)

# 6. SHAP
if best_model_pipeline:
    print(f"\nGenerating SHAP explanation for: {best_model_name}...")
    try:
        prep = best_model_pipeline.named_steps['prep']
        X_test_trans = prep.transform(X_test)
        
        # Get feature names dynamically
        feat_names = []
        if features_num:
            feat_names += features_num
        if features_cat:
            cat_encoder = prep.named_transformers_['cat']
            feat_names += list(cat_encoder.get_feature_names_out(features_cat))
        
        final_model = best_model_pipeline.named_steps['model']

        # Select correct explainer
        if "Linear" in best_model_name:
            explainer = shap.LinearExplainer(final_model, X_test_trans)
        else:
            explainer = shap.TreeExplainer(final_model)
            
        shap_values = explainer(X_test_trans)

        plt.figure()
        shap.summary_plot(shap_values, X_test_trans, feature_names=feat_names, show=False)
        plt.tight_layout()
        plt.savefig('visuals/4_shap_summary.png')
        print("SHAP Plot saved to visuals/4_shap_summary.png")
        
    except Exception as e:
        print(f"SHAP Error: {e}")
        print("Skipping SHAP. Please check visuals/ folder for other plots.")

print("\nTask 4 Completed.")