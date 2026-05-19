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

**Institution:** University of East London — MSc Data Science

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
| Attrition rate | ~24% |
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
- Distribution analysis of key features
- Correlation analysis
- Attrition patterns by department, job level, salary band,
  satisfaction scores, and tenure

### 2. Data Preprocessing
- Encoding of categorical variables (Label Encoding and
  One-Hot Encoding)
- Feature scaling using StandardScaler
- Removal of non-predictive identifier columns
- Train-test split (80/20)

### 3. Handling Class Imbalance
- Applied SMOTE (Synthetic Minority Oversampling Technique)
  to address the imbalanced target variable (~76% No, ~24% Yes)

### 4. Models Built and Compared
| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable baseline |
| Random Forest | Primary model, ensemble method |
| Gradient Boosting | Performance comparison |

### 5. Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix
- Feature Importance Analysis

---

## Key Results

| Model | Accuracy | ROC-AUC | F1 Score |
|---|---|---|---|
| Logistic Regression | TBC | TBC | TBC |
| Random Forest | TBC | TBC | TBC |
| Gradient Boosting | TBC | TBC | TBC |



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
- **ML:** Scikit-learn, XGBoost
- **Data Processing:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Imbalanced Learning:** Imbalanced-learn (SMOTE)
- **Environment:** Jupyter Notebook

---


## Author

**Gbemileke Falade**

Senior Data Analyst | Data Scientist | AI/ML Practitioner | Data Consultant

https://www.linkedin.com/in/gbemileke-falade

