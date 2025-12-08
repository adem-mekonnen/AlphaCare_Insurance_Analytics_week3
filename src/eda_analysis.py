import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set up the path for saving figures
FIGURE_DIR = 'reports/figures'
# Ensures the directory exists before saving plots
os.makedirs(FIGURE_DIR, exist_ok=True) 

# 1. Load Data
FILE_PATH = 'data/raw/insurance_data.csv'

try:
    # Try loading the file. If it's a TSV/semi-colon delimited file, add 'sep=';'' or 'sep='\t''
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"CRITICAL ERROR: File not found at {FILE_PATH}.")
    exit()

# Crucial: Clean headers to remove any hidden spaces or characters
df.columns = df.columns.str.strip() 

# The columns required for the challenge (MUST be present in the correct file):
REQUIRED_COLS_NUMERIC = ['TotalPremium', 'TotalClaims']
REQUIRED_COLS_CATEGORICAL = ['Gender', 'Province']

# 2. Data Cleaning
for col in REQUIRED_COLS_NUMERIC:
    if col in df.columns:
        # Convert to numeric, coercing errors to NaN
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
    else:
        # This will be printed if the column is still missing (meaning the data is wrong)
        print(f"CRITICAL WARNING: Column '{col}' not found. Please verify the dataset is correct.")

# Handle Missing Values 
if 'Gender' in df.columns:
    df.loc[:, 'Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

if 'TotalClaims' in df.columns:
    # Impute 'TotalClaims' with 0 (as NaN usually means 'No Claim')
    df.loc[:, 'TotalClaims'] = df['TotalClaims'].fillna(0)

# 3. Feature Engineering for Analysis
if all(col in df.columns for col in REQUIRED_COLS_NUMERIC):
    df.loc[:, 'LossRatio'] = df['TotalClaims'] / df['TotalPremium']
else:
    print("Cannot proceed with EDA/ML: Required numeric columns are missing. Exiting.")
    exit() 

# --- Visualizations: Saving to 'reports/figures/' ---

# 4. VISUALIZATION 1: Distribution of Total Premium (Outlier detection)
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['TotalPremium'])
plt.title('Distribution of Total Premium (Outlier Detection)')
plt.savefig(os.path.join(FIGURE_DIR, 'premium_outliers.png')) 
plt.close()

# 5. VISUALIZATION 2: Total Claims (Sum) by Province (Risk Analysis)
plt.figure(figsize=(12, 6))
if 'Province' in df.columns:
    sns.barplot(data=df, x='Province', y='TotalClaims', estimator='sum')
    plt.xticks(rotation=45, ha='right')
    plt.title('Total Claims Sum by Province (Risk Exposure)')
    plt.savefig(os.path.join(FIGURE_DIR, 'claims_by_province.png'))
    plt.close()
else:
    print("WARNING: Cannot create 'Claims by Province' plot: 'Province' column missing.")

# 6. VISUALIZATION 3: Premium vs Claims Correlation
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='TotalPremium', y='TotalClaims', alpha=0.5)
plt.title('Correlation: Premium vs Claims')
plt.savefig(os.path.join(FIGURE_DIR, 'correlation.png'))
plt.close()

print("EDA Complete. Visuals saved to reports/figures/.")