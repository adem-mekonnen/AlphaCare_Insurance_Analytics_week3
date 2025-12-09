import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency, f_oneway
import numpy as np

# 1. Load Real Data (Using pipe separator)
print("Loading Real Data...")
try:
    df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|', low_memory=False)
except FileNotFoundError:
    # Fallback if you renamed it
    df = pd.read_csv('data/insurance_data.csv', sep='|', low_memory=False)

# 2. Data Cleaning
# Fix Numeric Columns (replace commas if any)
cols = ['TotalPremium', 'TotalClaims']
for c in cols:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

df['Margin'] = df['TotalPremium'] - df['TotalClaims']
# Define Risk: Claim Frequency (IsClaimed) and Severity (TotalClaims > 0)
df['IsClaimed'] = df['TotalClaims'].apply(lambda x: 1 if x > 0 else 0)

print("\n--- TASK 3: HYPOTHESIS TESTING (Real Data) ---")

def report_result(name, p):
    print(f"\nHypothesis: {name}")
    print(f"P-Value: {p:.5e}")
    if p < 0.05:
        print(">> RESULT: REJECT Null Hypothesis (Significant Difference)")
    else:
        print(">> RESULT: FAIL TO REJECT Null Hypothesis")

# H1: Risk differences across provinces (Chi-Squared on Frequency)
contingency = pd.crosstab(df['Province'], df['IsClaimed'])
chi2, p_prov, dof, ex = chi2_contingency(contingency)
report_result("Risk Differences across Provinces", p_prov)

# H2: Risk differences between ZipCodes (ANOVA on Claims)
# We select top 20 zipcodes to ensure statistical validity
top_zips = df['PostalCode'].value_counts().head(20).index
zip_groups = [df[df['PostalCode'] == z]['TotalClaims'] for z in top_zips]
f_stat, p_zip = f_oneway(*zip_groups)
report_result("Risk Differences between ZipCodes", p_zip)

# H3: Margin difference between ZipCodes (ANOVA on Margin)
margin_groups = [df[df['PostalCode'] == z]['Margin'] for z in top_zips]
f_stat_m, p_zip_m = f_oneway(*margin_groups)
report_result("Margin Differences between ZipCodes", p_zip_m)

# H4: Risk difference between Women and Men (T-Test on Margin)
# Clean Gender column first
df['Gender'] = df['Gender'].astype(str).str.title()
male = df[df['Gender'] == 'Male']['Margin']
female = df[df['Gender'] == 'Female']['Margin']
t_stat, p_gen = ttest_ind(male, female, nan_policy='omit')
report_result("Risk (Margin) Difference: Women vs Men", p_gen)