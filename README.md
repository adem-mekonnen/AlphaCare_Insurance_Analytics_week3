# AlphaCare Insurance Risk Analytics & Predictive Modeling

![CI/CD Status](https://github.com/adem-mekonnen/AlphaCare_Insurance_Analytics_week3/actions/workflows/unittests.yml/badge.svg)

## 🎯 Business Objective

This project, undertaken for AlphaCare Insurance Solutions (ACIS), utilizes historical claim data (2014-2015) to optimize car insurance planning and marketing in South Africa. The core objectives are to:

1.  **Identify Low-Risk Segments:** Discover geographic and demographic clusters where premiums can be reduced to attract new clients.
2.  **Optimize Pricing:** Develop a predictive modeling framework to estimate claim severity and refine premium setting strategies.

---

## 🛠 Project Structure

This repository follows a standard MLOps-ready structure ensuring modularity, reproducibility, and auditability.

```text
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
