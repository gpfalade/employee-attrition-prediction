# Predicting Employee Attrition in Nigerian Financial Institutions
### Using Advanced Machine Learning Techniques

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Overview

This project predicts employee attrition in a Nigerian financial institution
using advanced machine learning techniques. It is a reconstruction of my
MSc Data Science dissertation research, completed at the University of East
London.

Employee attrition is one of the most operationally disruptive and
financially costly challenges facing Nigerian banks today. Replacing a
single employee, when recruitment, onboarding, knowledge transfer, and
lost productivity are factored in, can cost between 50% and 200% of their
annual salary. For large financial institutions managing thousands of staff
across multiple departments and locations, unmanaged attrition compounds
into a strategic risk.

This research investigates whether machine learning models can identify
employees at high risk of leaving with sufficient accuracy to enable
proactive intervention by HR and management teams before the resignation
letter arrives.

---

## Dissertation Reference

**Title:** Predicting Employee Attrition in Financial Institutions Using
Advanced Machine Learning Techniques

**Institution:** University of East London - MSc Data Science

**Grade:** Distinction

**Year:** 2025

> The original dissertation used simulated institutional HR data from a Nigerian
> financial services environment. This repository reconstructs the full
> methodology using a synthetic dataset programmatically generated to
> mirror realistic Nigerian banking HR records, preserving data
> confidentiality obligations while demonstrating the complete
> analytical approach.

---

## Dataset

The dataset used in this project is a synthetic HR record set for a
Nigerian financial institution, covering the period January 2015 to
December 2024.

| Property | Detail |
|---|---|
| Total records | 5,000 |
| Features | 36 |
| Target variable | Attrition (Yes / No) |
| Attrition rate | ~26% |
| Date range | 2015 – 2024 |
| Employment types | Permanent and Contract staff |
| Departments | 37 Nigerian banking departments |

**The dataset was programmatically generated** using
`generate_hr_data.py` in this repository. It reflects realistic
Nigerian banking sector characteristics including:

- Nigerian state distribution for employee origins
- Naira-denominated salary ranges by job level
- Nigerian banking department structures
- Grade levels aligned to Nigerian banking industry conventions
- Post-2022 remote work patterns
- Realistic attrition drivers for the Nigerian labour market

The dataset is available in the `/data` folder.

---

## Research Objectives

- Identify the key drivers of employee attrition in a Nigerian
  financial institution
- Build and compare multiple machine learning classification models
- Handle class imbalance using appropriate resampling techniques
- Evaluate models using industry-standard classification metrics
- Derive actionable HR strategy insights from model outputs

---

## Methodology

### 1. Exploratory Data Analysis
- Attrition distribution analysis - count and proportion charts
- Attrition rate breakdown by job level, department group, and salary band
- Monthly income distribution by attrition status
- Satisfaction score comparison (job satisfaction, work-life balance,
  manager relationship, environment satisfaction) across attrition groups
- Engagement score distribution by attrition outcome
- Correlation heatmap of all numeric features against the attrition target

---

### 2. Feature Engineering
- **Geopolitical Zone mapping:** 37 Nigerian states consolidated into
  6 geopolitical zones (South West, South East, South South, North West,
  North East, North Central) to reduce dimensionality and improve
  model interpretability
- **Department Group mapping:** 37 banking departments consolidated into
  6 functional groups (Business Banking, Treasury & Finance, Operations,
  Digital & Technology, Risk & Compliance, Support Functions)

---

### 3. Data Preprocessing
- Removed leakage and non-predictive columns: EmployeeID, HireDate,
  ExitDate, EmploymentStatus, Reason_for_Leaving, SalaryBand
- Binary encoding: Gender, EmploymentType, BonusReceived, RemoteWorkOption
- Ordinal encoding: OvertimeFrequency (Never=0 to Always=4),
  BusinessTravel (No Travel=0 to Frequent=2)
- Label encoding: MaritalStatus, EducationField,
  GeopoliticalZone, DepartmentGroup
- Target encoding: Attrition (Yes=1, No=0)
- Feature scaling: StandardScaler applied to all features

---

### 4. Train-Test Split
- 80/20 stratified split to preserve attrition class proportions
  across training and test sets
- Test set: 1,000 samples - held out completely for final evaluation
- Training set: 4,000 samples - used for all model fitting and tuning

---

### 5. Class Imbalance Handling
Two complementary strategies were applied and compared:

**Strategy A - SMOTE (Synthetic Minority Oversampling Technique)**
Applied to the training set only to generate synthetic samples
for the minority attrition class, producing a balanced 50/50
training distribution while preserving the test set integrity.

**Strategy B - Class-Weighted Models**
Applied class weights inversely proportional to class frequencies,
penalising the model more heavily for misclassifying attrition cases.
This approach avoids generating synthetic data while still addressing
the imbalance.

---

### 6. Baseline Model Training
Three classification models trained on the SMOTE-resampled data
as a performance baseline:

| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable linear baseline |
| Random Forest | Ensemble tree-based model |
| Gradient Boosting | Sequential ensemble comparison |

---

### 7. Threshold Optimisation
Default classification threshold (0.50) was evaluated against multiple
alternative thresholds (0.20 to 0.60) to identify the optimal
precision-recall trade-off for the attrition class.

For HR attrition applications, recall is prioritised - correctly
identifying employees likely to leave is more valuable than
minimising false positives. Threshold selection was guided by
maximising F1-Score for the attrition class across the threshold range.

---

### 8. Stratified K-Fold Cross-Validation
5-fold stratified cross-validation applied to all candidate models
to assess generalisation performance and detect overfitting.
PR-AUC (Precision-Recall Area Under Curve) used as the primary
cross-validation metric given the class imbalance context.

---

### 9. Feature Signal Validation
A shuffled-label test was performed to confirm the model was
learning genuine patterns rather than noise:
- Real labels CV PR-AUC: **0.4865 ± 0.033**
- Shuffled labels CV PR-AUC: **0.2456 ± 0.007**

The significant gap confirms the model is responding to real
signal in the data, not random variation.

---

### 10. Mutual Information Feature Selection
Mutual information scores computed for all features to rank
predictive relevance. Top features by information gain:

| Rank | Feature | MI Score |
|---|---|---|
| 1 | EngagementScore | 0.0687 |
| 2 | WorkLifeBalance | 0.0348 |
| 3 | JobSatisfaction | 0.0340 |
| 4 | ManagerRelationship | 0.0267 |
| 5 | LastSalaryIncreasePct | 0.0199 |

---

### 11. Collinearity Pruning
Pearson correlation analysis performed across all features.
One feature dropped for high collinearity (correlation > 0.95),
producing a final pruned feature set of **37 features**.

---

### 12. Advanced Boosting Models
Three advanced gradient boosting frameworks evaluated on the
pruned feature set with hyperparameter tuning via RandomizedSearchCV:

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|
| XGBoost | 0.4824 | 0.4743 | - |
| LightGBM | 0.4645 | 0.4754 | - |
| **CatBoost** | **0.4879** | **0.4777** | **0.7504** |

CatBoost selected as the final model based on highest CV PR-AUC
and best generalisation to the test set.

---

### 13. Final Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | CatBoost Classifier |
| Classification threshold | 0.54 (optimised) |
| Test ROC-AUC | 0.7504 |
| Test PR-AUC | 0.4777 |
| Recall (Attrition class) | 0.69 |
| F1-Score (Attrition class) | 0.54 |

The threshold was set to 0.54 to maximise recall for the attrition
class, reflecting the real-world HR objective of identifying
at-risk employees before they resign, accepting some false positives
as an operational trade-off.

---

## Key Results

| Model | Accuracy | ROC-AUC | F1 Score |
|---|---|---|---|
| Logistic Regression | 0.68 | 0.737 | 0.50 |
| Random Forest | 0.75 | 0.740 | 0.40 |
| Gradient Boosting | 0.74 | 0.742 | 0.43 |
| **CatBoost (Final)** | **0.70** | **0.750** | **0.54** |

*Best model: CatBoost with optimised classification threshold (0.54),
selected for highest ROC-AUC and best recall on the minority attrition class.*



---

## Repository Structure

```
employee-attrition-prediction/
│
├── README.md
├── requirements.txt
├── generate_hr_data.py
├── .gitignore
│
├── data/
│   ├── README.md
│   └── nigerian_bank_hr_dataset_final.csv
│
└── notebooks/
└── attrition_prediction.ipynb

```

---

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Notebook
1. Clone this repository
2. Navigate to the `/notebooks` folder
3. Open `attrition_prediction.ipynb`
4. Run all cells sequentially

### Regenerate the Dataset
```bash
python generate_hr_data.py
```

---

## Tech Stack

- **Language:** Python 3.9+
- **ML Libraries:** Scikit-learn, XGBoost, LightGBM, CatBoost
- **Data Processing:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Imbalanced Learning:** Imbalanced-learn (SMOTE)
- **Environment:** Jupyter Notebook

---


## Author

**Gbemileke Falade**

Senior Data Analyst | Data Scientist | AI/ML Practitioner | Data Consultant

https://www.linkedin.com/in/gbemileke-falade

