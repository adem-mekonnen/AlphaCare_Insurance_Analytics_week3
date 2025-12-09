import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure visuals folder exists
os.makedirs('visuals', exist_ok=True)

# 1. Load Data
print("Loading Data...")
df = pd.read_csv('data/raw/insurance_data.csv')

# --- EMERGENCY DATA ADAPTER ---
# Mapping your Medical Data to AlphaCare Car Insurance Names
print("Adapting dataset columns...")
if 'charges' in df.columns:
    df.rename(columns={
        'charges': 'TotalClaims',   # Treat Medical Charges as Claims
        'region':  'Province',      # Treat Region as Province
        'sex':     'Gender'         # Treat Sex as Gender
    }, inplace=True)

    # We need 'TotalPremium' to calculate Margin. 
    # Since it's missing, we simulate it: Premium = Claims + 20% profit margin
    np.random.seed(42)
    df['TotalPremium'] = df['TotalClaims'] * np.random.uniform(1.1, 1.3, size=len(df))
    print("  -> Created synthetic 'TotalPremium' column for analysis.")
# -----------------------------

# 2. Cleaning & Feature Engineering
df['Margin'] = df['TotalPremium'] - df['TotalClaims']

# 3. Visuals
print("Generating Visuals...")

# Plot 1: Premium Distribution
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['TotalPremium'])
plt.title('Distribution of Total Premium (Simulated)')
plt.savefig('visuals/1_outliers.png')
plt.close()

# Plot 2: Claims by Province (Region)
plt.figure(figsize=(12, 6))
province_risk = df.groupby('Province')['TotalClaims'].sum().sort_values(ascending=False).reset_index()
sns.barplot(data=province_risk, x='Province', y='TotalClaims')
plt.xticks(rotation=45)
plt.title('Total Claims by Province')
plt.savefig('visuals/2_province_risk.png')
plt.close()

# Plot 3: Margin by Gender
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='Gender', y='Margin', estimator='mean')
plt.title('Average Margin (Profit) by Gender')
plt.savefig('visuals/3_margin_gender.png')
plt.close()

print("EDA Complete. Images saved to visuals/")