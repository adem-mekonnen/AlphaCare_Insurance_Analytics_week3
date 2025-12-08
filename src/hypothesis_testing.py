import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

df = pd.read_csv('data/insurance_data.csv')
# Ensure TotalClaims is numeric
df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce').fillna(0)
df['IsClaimed'] = df['TotalClaims'].apply(lambda x: 1 if x > 0 else 0)

def perform_ttest(group_col, value_col, group1, group2):
    g1 = df[df[group_col] == group1][value_col]
    g2 = df[df[group_col] == group2][value_col]
    t_stat, p_val = ttest_ind(g1, g2, equal_var=False)
    print(f"T-Test {group_col}: {group1} vs {group2} | P-value: {p_val}")
    if p_val < 0.05:
        print("Result: Reject Null Hypothesis (Significant Difference)")
    else:
        print("Result: Fail to Reject Null Hypothesis")

def perform_chi2(col1, col2):
    contingency_table = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    print(f"Chi-Square {col1} vs {col2} | P-value: {p}")
    if p < 0.05:
        print("Result: Reject Null Hypothesis (Dependent)")
    else:
        print("Result: Fail to Reject Null Hypothesis")

print("--- HYPOTHESIS TESTING ---")

# Hypothesis 1: Risk differences between Gender (Women vs Men)
# KPI: Margin (TotalPremium - TotalClaims)
df['Margin'] = df['TotalPremium'] - df['TotalClaims']
perform_ttest('Gender', 'Margin', 'Male', 'Female') # Adjust strings based on your data labels

# Hypothesis 2: Risk differences across Provinces (Chi-Square for frequency)
perform_chi2('Province', 'IsClaimed')

# Hypothesis 3: Risk difference between ZipCodes (PostalCode)
# Since ZipCodes are many, we might group them or pick top 2 for T-test
top_zips = df['PostalCode'].value_counts().head(2).index.tolist()
perform_ttest('PostalCode', 'TotalClaims', top_zips[0], top_zips[1])