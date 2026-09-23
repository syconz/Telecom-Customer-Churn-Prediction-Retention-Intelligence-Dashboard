# 📡 Telecom Customer Churn Prediction & Retention Intelligence Dashboard

> **IBM SkillsBuild Data Analytics with AI Academic Internship**
> **Author:** Shivam Singh &nbsp;|&nbsp; **Program:** IBM SkillsBuild Academic Internship &nbsp;|&nbsp; **Language:** Python 3.9+

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Problem Statement

Telecom companies face high customer churn rates — losing customers to competitors is expensive
because acquiring a new customer costs **5–10× more** than retaining an existing one.
The challenge is to identify *which* customers are likely to churn *before* they do,
understand the key drivers of churn, and enable the business to take targeted retention actions.

---

## 🎯 Project Objective

Build an end-to-end **Business Intelligence and Machine Learning** solution that:

1. Cleans and analyses telecom customer data to surface churn patterns.
2. Trains ML models to predict which customers will churn.
3. Assigns churn risk categories (Low / Medium / High) to every customer.
4. Identifies the top business drivers of churn via feature importance.
5. Presents all findings through an **executive-grade interactive Streamlit dashboard**.
6. Generates AI-style business insights and recommended retention actions — all derived from the data.

---

## 🗂 Dataset Description

| Property      | Value |
|---------------|-------|
| Source        | IBM Sample Dataset (hosted on Kaggle) |
| URL           | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| File name     | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Rows          | ~7,043 customers |
| Columns       | 21 |
| Target column | `Churn` (Yes / No) |

### Key Columns

| Column | Description |
|--------|-------------|
| `customerID` | Unique identifier — dropped during model training |
| `gender` | Male / Female |
| `SeniorCitizen` | 1 = Senior, 0 = Non-senior |
| `Partner` | Has a partner (Yes/No) |
| `Dependents` | Has dependents (Yes/No) |
| `tenure` | Months with the company |
| `InternetService` | DSL / Fiber optic / No |
| `Contract` | Month-to-month / One year / Two year |
| `PaperlessBilling` | Yes / No |
| `PaymentMethod` | Electronic check / Mailed check / Bank transfer / Credit card |
| `MonthlyCharges` | Monthly bill amount ($) |
| `TotalCharges` | Total billed amount ($) |
| `Churn` | **Target** — Yes / No |

> Full column list: `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `PhoneService`, `MultipleLines`

---

## 🛠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.9+ | Core language |
| pandas | Data manipulation & analysis |
| numpy | Numerical operations |
| matplotlib | Static EDA visualisations (notebook) |
| seaborn | Statistical visualisations (notebook) |
| scikit-learn | ML models, preprocessing, evaluation |
| streamlit | Interactive executive dashboard |
| plotly | Interactive charts in dashboard |
| joblib | Model serialisation & persistence |

---

## 🔄 Project Workflow

```
Raw CSV Data
    │
    ▼
Data Loading (pandas)
    │
    ▼
Data Cleaning
  • Fix TotalCharges (blank strings → numeric)
  • Drop 11 rows with missing TotalCharges
  • Encode Churn column (Yes→1, No→0)
  • Drop customerID
  • Remove duplicate rows
    │
    ▼
Exploratory Data Analysis (12 dimensions)
  • Churn distribution
  • Churn by contract, tenure, payment method, internet service
  • Churn by gender, senior citizen, tech support, online security
  • Monthly charges & total charges analysis
    │
    ▼
Business KPI Calculation
  • Total customers, churn rate, retention rate
  • Monthly revenue, avg charges, avg tenure
  • High-risk customer count, revenue at risk
    │
    ▼
Machine Learning Pipeline
  • ColumnTransformer: OneHotEncoder (categorical) + StandardScaler (numerical)
  • Logistic Regression  (class_weight=balanced)
  • Random Forest        (class_weight=balanced, 200 trees)
  • Compare: Accuracy, Precision, Recall, F1, ROC-AUC
  • Select best model by ROC-AUC
    │
    ▼
Predictions & Risk Labelling
  • ChurnProbability for every customer
  • Risk: Low (<30%) | Medium (30–60%) | High (>60%)
    │
    ▼
Feature Importance (Top Churn Drivers)
    │
    ▼
AI-style Business Insights (auto-derived from data)
    │
    ▼
Executive Streamlit Dashboard (5 pages)
    │
    ▼
Business Decisions & Recommended Retention Actions
```

---

## 🤖 Machine Learning Models

### Logistic Regression
- Linear probabilistic classifier — interpretable baseline.
- `class_weight='balanced'` handles the ~26% churn class imbalance.
- `max_iter=1000` ensures convergence.

### Random Forest
- Ensemble of 200 decision trees — captures non-linear relationships.
- `class_weight='balanced'`, `max_depth=10` — prevents overfitting.
- Provides reliable feature importance scores.

### Model Selection Strategy
The **best model is selected by ROC-AUC**, not accuracy.
- Churn prediction is a **business-risk problem** — a missed churner (False Negative) costs more than a false alarm (False Positive).
- ROC-AUC measures discrimination ability across all thresholds, making it the appropriate primary metric.

---

## 📊 Evaluation Metrics

| Metric | Why it matters for churn |
|--------|--------------------------|
| Accuracy | Overall correctness — can mislead with class imbalance |
| Precision | Of predicted churners, how many actually churn |
| Recall | Of actual churners, how many the model catches *(most critical)* |
| F1 Score | Harmonic mean of Precision and Recall |
| **ROC-AUC** | **Primary metric** — discrimination ability across all thresholds |

---

## 🖥 Dashboard Features

### Page 1 — 🏠 Executive Overview
- KPI cards: Total Customers, Churn Rate, Retention Rate, Monthly Revenue, High-Risk count, Avg Monthly Charges
- Churn distribution pie chart
- Churn by contract type
- Churn risk distribution bar chart
- Monthly charges vs churn box plot
- **Executive Insights tabs:** Key Findings | Risks | Opportunities | Recommended Actions

### Page 2 — 📊 Churn & Customer Analysis
- **6 sidebar filters:** Contract, Internet Service, Payment Method, Senior Citizen, Gender, Risk Level
- Churn by tenure (grouped), payment method, internet service, online security
- Monthly charges distribution by churn status
- Tech support & churn, gender & senior citizen churn

### Page 3 — 🎯 Risk, Opportunity & Action
- Risk KPI cards: High / Medium / Low risk counts + Revenue at Risk
- Risk distribution by contract type (stacked bar)
- Top-10 churn drivers (feature importance)
- Opportunity charts: paperless billing, tenure groups
- Data-driven recommended actions (auto-generated from dataset)
- **High-risk customer snapshot table** (top 20 by churn probability)

### Page 4 — 🔮 Customer Prediction
- 20-field individual customer input form
- **Churn probability gauge chart**
- Risk category (Low / Medium / High) with colour coding
- Key factors influencing the prediction
- Personalised recommended retention action

### Page 5 — 🤖 Model Performance
- Side-by-side model comparison table (highlighted best scores)
- Dual ROC curves with AUC labels
- Confusion matrix for best model
- Full sklearn classification report

---

## ⚙️ Installation Instructions

### Prerequisites
- Python 3.9 or higher
- pip

### Step 1 — Clone the Repository

```bash
git clone https://github.com/syconz/Telecom-Customer-Churn-Prediction-Retention-Intelligence-Dashboard.git
cd Telecom-Customer-Churn-Prediction-Retention-Intelligence-Dashboard
```

### Step 2 — Create a Virtual Environment *(recommended)*

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download the Dataset

1. Go to → https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Place the file in the **same folder** as `ShivamSingh_TelecomCustomerChurn.py`

---

## ▶️ How to Run

### Streamlit Dashboard

```bash
streamlit run ShivamSingh_TelecomCustomerChurn.py
```

Opens at **http://localhost:8501**

> **First run:** model trains automatically (~30 seconds) and saves to `churn_model_pipeline.joblib`.  
> **Subsequent runs:** trained model loads instantly — no retraining.

### Jupyter Notebook

```bash
jupyter notebook ShivamSingh_TelecomCustomerChurn.ipynb
```

Or open directly in **VS Code** with the Jupyter extension (`Ctrl+Shift+P` → *Open With... → Jupyter Notebook*).

---

## 📤 Expected Outputs

### Streamlit Dashboard
- Professional 5-page dashboard at `http://localhost:8501`
- All charts are interactive (hover, zoom, filter)
- Sidebar filters update all charts in real time
- Individual customer churn prediction with probability gauge

### Jupyter Notebook
- 48 cells (34 code + 14 markdown) — step-by-step analysis
- 15+ visualisations: bar charts, histograms, correlation heatmap, ROC curves, confusion matrix
- Model evaluation tables and feature importance plots
- Auto-generated executive insights printed to output

### Generated Files
| File | When created |
|------|-------------|
| `churn_model_pipeline.joblib` | Auto-saved on first Streamlit run |

---

## 📁 Project Structure

```
Telecom-Customer-Churn-Prediction-Retention-Intelligence-Dashboard/
├── ShivamSingh_TelecomCustomerChurn.py         ← Main Streamlit application
├── ShivamSingh_TelecomCustomerChurn.ipynb      ← Jupyter analysis notebook
├── ShivamSingh_TelecomCustomerChurnReport.docx ← Formal project report (18 sections)
├── requirements.txt                            ← Python dependencies
├── README.md                                   ← This file
├── WA_Fn-UseC_-Telco-Customer-Churn.csv        ← Dataset (download from Kaggle — not in repo)
└── churn_model_pipeline.joblib                 ← Auto-generated on first run (not in repo)
```

---

## ❓ Key Business Questions Answered

| # | Question | Where answered |
|---|----------|----------------|
| 1 | What is the overall churn rate? | KPI card — Executive Overview page |
| 2 | Which contract type has the highest churn? | Contract analysis — all pages |
| 3 | Do newer customers churn more? | Tenure analysis — Page 2 |
| 4 | Does internet service type affect churn? | Internet service chart — Page 2 |
| 5 | How much monthly revenue is at risk? | Revenue at Risk KPI — Page 3 |
| 6 | Which features are the strongest churn predictors? | Feature importance chart — Page 3 |
| 7 | Which specific customers should we contact first? | High-risk customer table — Page 3 |
| 8 | What retention actions should we take? | Recommended Actions — Page 3 |

---

## ⚠️ Disclaimer

- All analysis results are computed directly from the dataset at runtime — **no values are hard-coded**.
- Feature importance indicates **association/correlation**, not causation.
- Model predictions are probabilistic — real retention decisions should be validated with A/B testing.
- This project is intended for educational purposes as part of the IBM SkillsBuild Academic Internship.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

*Made with ❤️ for IBM SkillsBuild Data Analytics with AI Academic Internship &nbsp;|&nbsp; Shivam Singh*
