markdown
# AlphaCare Insurance Risk Analytics & Predictive Modeling

![CI/CD Status](https://github.com/adem-mekonnen/AlphaCare_Insurance_Analytics_week3/actions/workflows/unittests.yml/badge.svg)

## 🎯 Business Objective

This project, undertaken for AlphaCare Insurance Solutions (ACIS), utilizes historical claim data (2014-2015) to optimize car insurance planning and marketing in South Africa. The core objectives are to:

1.  **Identify Low-Risk Segments:** Discover geographic and demographic clusters where premiums can be reduced to attract new clients.
2.  **Optimize Pricing:** Develop a predictive modeling framework to estimate claim severity and refine premium setting strategies.

---

## 🛠 Project Structure

This repository follows a standard MLOps-ready structure ensuring modularity, reproducibility, and auditability.

text
AlphaCare_Insurance_Analytics/
│
├── .github/workflows/       # CI/CD Pipelines (Tests & Linting)
├── data/                    # Data Version Controlled via DVC
│   └── MachineLearningRating_v3.txt  # Raw dataset (tracked by DVC)
├── notebooks/               # Jupyter notebooks for visual reporting
│   └── final_report_figures.ipynb
├── reports/                 # Generated reports and figures
│   └── figures/             # Final EDA plots and SHAP analysis graphs
├── src/                     # Source code
│   ├── eda.py               # Exploratory Data Analysis script
│   ├── hypothesis_testing.py# Statistical A/B Testing script
│   └── modeling.py          # Machine Learning training & evaluation
│
├── .dvcignore               # DVC configuration
├── .gitignore               # Git configuration
└── requirements.txt         # Python dependencies


---

## 🚀 Quick Start Guide

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/adem-mekonnen/AlphaCare_Insurance_Analytics_week3.git
    cd AlphaCare_Insurance_Analytics
    ```
2.  **Set up Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/Scripts/activate # Windows Git Bash
    # source venv/bin/activate   # Linux/macOS
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Data Setup (DVC):**
    *   This project uses DVC to track large files. To pull the data from the remote storage:
    ```bash
    dvc pull
    ```

---

## 📊 Completed Tasks

### ✅ Task 1: Dev Environment & CI/CD
*   **Git Workflow:** Feature-branch workflow used for all tasks.
*   **CI/CD Pipeline:** A GitHub Actions workflow is configured to run sanity checks and verify environment dependencies on every push to `main`.
*   **Exploratory Data Analysis (EDA):** Initial analysis cleaned the `MachineLearningRating_v3` dataset, handled pipe (`|`) delimiters, and identified key outliers in Premium distributions.

### ✅ Task 2: Data Version Control (DVC)
To ensure auditability required by financial regulations, raw data is decoupled from the codebase.
*   **Status:** The primary dataset (`MachineLearningRating_v3.txt`) is tracked via DVC.
*   **Storage:** Data is pushed to a configured local remote storage, ensuring the Git repository remains lightweight.

### ✅ Task 3: A/B Hypothesis Testing
We statistically validated business assumptions using Chi-Squared and ANOVA tests ($p < 0.05$).

| Hypothesis Tested | Test Used | P-Value | Result | Insight |
| :--- | :--- | :--- | :--- | :--- |
| **Risk vs. Province** | Chi-Squared | `< 0.001` | **Significant** | Location is a primary driver of claim frequency. |
| **Risk vs. ZipCode** | ANOVA | `< 0.001` | **Significant** | Risk varies significantly even within provinces. |
| **Margin vs. ZipCode** | ANOVA | `0.011` | **Significant** | Some neighborhoods are inherently more profitable. |
| **Margin vs. Gender** | T-Test | `0.833` | **Not Significant** | **Gender does not impact profitability.** |

### ✅ Task 4: Predictive Modeling
We built and compared three models to predict Claim Severity (`TotalClaims`).

**Model Leaderboard:**
1.  **Linear Regression:** RMSE: 2202.05 | $R^2$: 0.0076 (Best Baseline)
2.  **XGBoost:** RMSE: 2213.54 | $R^2$: -0.0027
3.  **Random Forest:** RMSE: 2271.64 | $R^2$: -0.0561

**Feature Importance (SHAP):**
*   **TotalPremium:** The strongest predictor of claims, confirming that current underwriting rules are directionally correct.
*   **Province & VehicleType:** Secondary drivers that offer opportunities for granular pricing adjustments.

---

## 📈 Recommendations
Based on the analysis, ACIS should:
1.  **Implement Hyper-Local Pricing:** Move from provincial base rates to Zip Code-level risk loading.
2.  **Focus Marketing:** Target high-margin postal codes identified in Task 3.
3.  **Review Vehicle Classes:** Apply surcharges to specific high-risk vehicle types identified by the SHAP analysis.
```

