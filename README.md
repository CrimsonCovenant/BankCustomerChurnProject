# Customer Complaint Analysis for Churn Risk Prediction

## Project Goal

The primary objective of this project was to develop a machine learning system capable of analyzing customer complaints to predict potential customer churn risk. By integrating complaint data with demographic, geographic, and economic information, the project aimed to:
1.  Identify customers or segments at high risk of leaving the bank.
2.  Understand the key factors (products, issues, locations, demographics) driving dissatisfaction and potential churn.
3.  Provide insights to inform targeted customer retention strategies and operational improvements.

## Data Sources & Preparation

The project utilized several datasets:
* **Customer Complaints:** Detailed records including product type, issue description, company response, state, and ZIP code (e.g., `complaints-2025-05-01_22_48.csv`).
* **National Income & Demographics:** ZIP code level data including population and various income metrics (`national_income_clean.csv`).
* **Economic Indicators:** Additional datasets covering median wages, consumer credit, business loans, and bank deposits (`median_wages_clean.csv`, `cclac_clean.csv`, `busloans_clean.csv`, `sod_clean.csv`).

**Data Cleaning:**
Significant effort was dedicated to data wrangling and cleaning:
* Standardizing formats, particularly ZIP codes (ensuring 5-digit string representation).
* Handling missing values through imputation (e.g., imputing missing ZIPs based on state capitals, imputing missing states based on ZIPs using `pgeocode`) or strategic removal.
* Dropping irrelevant columns.
* Aggregating data where necessary (e.g., calculating complaint counts per ZIP).
* Defining a **Churn Risk Proxy:** Since direct churn data was unavailable, the `Company response to consumer` field was used to create a binary target variable (`is_churn_risk`). Responses like 'Closed without relief' were mapped to indicate higher churn risk (1), while others indicated lower risk (0).

The final cleaned dataset used for the primary churn prediction model is saved as `part_4_data_cleaning.csv`.

## Methodology & Models

A multi-model approach was adopted to explore different facets of the data and address the project goals:

1.  **Linear Regression:** Attempted to predict complaint *rates* per capita based on aggregated annual economic/demographic data.
2.  **Classification Complaint Rate Category:** Random Forest and Neural Network models were used to classify ZIP codes into 'Low'/'Medium'/'High' complaint rate categories based on a median split, incorporating income/demographics and aggregated complaint features.
3.  **Random Forest Churn Risk Prediction:** Focused on predicting the binary `is_churn_risk` proxy using only categorical features derived from individual complaints: `Product`, `Issue`, `Company`, and `State`. This isolated the impact of complaint context.

Preprocessing for classification models involved imputation (median/most frequent) and scaling (StandardScaler) for numeric features (where used), and imputation (most frequent) and One-Hot Encoding for categorical features, managed within scikit-learn Pipelines.

## Results & Key Findings

* **Linear Regression Failure:** The initial attempt to predict complaint rates linearly based on aggregate economic/demographic data yielded very poor results (R² ≈ 0.03), indicating these factors alone are weak predictors of complaint volume in a linear fashion.
* **Initial Classification Success (Caution Advised):** Models classifying ZIP codes based on a median split of complaint rates achieved near-perfect accuracy. However, this was likely due to the strong signal from aggregated complaint features easily separating the two groups, rather than nuanced prediction.
* **Final Churn Risk Model:**
    * **Performance:** The Random Forest using only complaint characteristics (Product, Issue, Company, State) achieved ~74% test accuracy. It demonstrated excellent **Recall (98%)** for the 'Churn Risk' class, successfully identifying most potential high-risk complaints based on the defined proxy. However, **Precision (7%)** for this class was very low, indicating a high rate of false positives.
    * **Key Drivers:** Feature importance analysis revealed that specific `Product` categories (e.g., Mortgage, Checking/Savings) and `Issue` types (e.g., Loan modification/collection, Managing an account) were the most influential factors differentiating potential churn risk.
    * **Hot Spots:** The model enabled the identification of ZIP codes with the highest average predicted churn risk based on complaints originating from those areas in the test set.

## Successes & Limitations (Win/Loss)

**Successes:**
* Demonstrated that complaint data (product, issue, company, state) contains significant predictive signals for potential churn risk (proxied by company response).
* Successfully identified the majority of high-risk complaints (high recall).
* Pinpointed specific products and issues most strongly associated with negative outcomes, providing actionable insights.
* Identified geographic hot spots for potential churn risk.
* Successfully implemented multiple machine learning models (Linear Regression, Random Forest, Neural Network).

**Limitations/Failures:**
* Inability of linear models to predict complaint rates from aggregate economic data.
* Very low precision in predicting the 'Churn Risk' class in the final model, limiting its direct use for targeted interventions without refinement.
* Reliance on a *proxy* for churn (`Company response to consumer`) instead of actual customer churn data.
* Limitations inherent in using aggregated geographic data (income/demographics by ZIP) versus individual customer data.

## Conclusion & Future Work

This project successfully leveraged machine learning to extract valuable insights from customer complaint data regarding potential churn risk. While predicting exact complaint rates proved difficult, classification models effectively identified high-risk scenarios and key driving factors related to specific products and issues.

The most critical next step is to **obtain and integrate individual customer-level data**, linking demographics, account details, transaction history, complaints, and actual churn events. This will enable the development of far more accurate and directly actionable churn prediction models. If only current data types remain available, focus should be on **improving the precision** of the final Random Forest model (e.g., through threshold tuning, advanced sampling techniques like SMOTE) and potentially **refining the definition of the churn risk proxy**. Analyzing the identified product/issue drivers and geographic hot spots can guide immediate operational improvements and targeted customer service strategies.
