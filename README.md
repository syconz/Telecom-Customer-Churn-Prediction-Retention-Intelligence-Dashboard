# 📡 Telecom Customer Churn Prediction & Retention Intelligence Dashboard

**IBM SkillsBuild Data Analytics with AI Academic Internship**
**Author:** Shivam Singh

---

## Problem Statement

Telecom companies face high customer churn rates — losing customers to competitors is expensive
because acquiring new customers costs 5–10× more than retaining existing ones.
The challenge is to identify *which* customers are likely to churn *before* they do,
understand the key drivers of churn, and enable the business to take targeted retention actions.

---

## Project Objective

Build an end-to-end Business Intelligence and Machine Learning solution that:

1. Cleans and analyses telecom customer data to surface churn patterns.
2. Trains machine learning models to predict which customers will churn.
3. Assigns churn risk categories to every customer.
4. Identifies the top business drivers of churn.
5. Presents all findings through an executive-grade interactive Streamlit dashboard.
6. Generates AI-style business insights and recommended retention actions.

---

## Dataset Description

| Property       | Value                                      |
|----------------|--------------------------------------------|
| Source         | IBM Sample Dataset (hosted on Kaggle)      |
| URL            | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| File name      | `WA_Fn-UseC_-Telco-Customer-Churn.csv`     |
| Rows           | ~7,043                                     |
| Columns        | 21                                         |
| Target column  | `Churn` (Yes / No)                         |

### Key Columns

| Column            | Description                                    |
|-------------------|------------------------------------------------|
| customerID        | Unique customer identifier (dropped in training)|
| gender            | Customer gender (Male / Female)               |
| SeniorCitizen     | Whether the customer is a senior citizen (0/1)|
| Partner           | Whether the customer has a partner            |
| Dependents        | Whether the customer has dependents           |
| tenure            | Number of months the customer has been with the company |
| PhoneService      | Whether the customer has phone service        |
| MultipleLines     | Whether the customer has multiple lines       |
| InternetService   | DSL / Fiber optic / No                        |
| OnlineSecurity    | Whether the customer has online security      |
| OnlineBackup      | Whether the customer has online backup        |
| DeviceProtection  | Whether the customer has device protection    |
| TechSupport       | Whether the customer has tech support         |
| StreamingTV       | Whether the customer streams TV               |
| StreamingMovies   | Whether the customer streams movies           |
| Contract          | Month-to-month / One year / Two year          |
| PaperlessBilling  | Whether the customer uses paperless billing   |
| PaymentMethod     | Electronic check / Mailed check / Bank transfer / Credit card |
| MonthlyCharges    | The amount charged monthly ($)                |
| TotalCharges      | The total amount charged ($)                  |
| Churn             | Whether the customer churned (Yes / No)       |

---

## Technologies Used

| Technology    | Purpose                                    |
|---------------|--------------------------------------------|
| Python 3.9+   | Core programming language                  |
| pandas        | Data manipulation and analysis             |
| numpy         | Numerical operations                       |
| matplotlib    | Static visualisations (EDA notebook)       |
| seaborn       | Statistical visualisations (EDA notebook)  |
| scikit-learn  | ML models, preprocessing, evaluation       |
| streamlit     | Interactive executive dashboard            |
| plotly        | Interactive charts in the dashboard        |
| joblib        | Model serialisation/persistence            |

---

## Project Workflow

```
Raw CSV Data
    │
    ▼
Data Loading (pandas)
    │
    ▼
Data Cleaning
  • Fix TotalCharges (string → numeric)
  • Drop 11 rows with missing TotalCharges
  • Encode Churn column (Yes→1, No→0)
  • Drop customerID
  • Remove duplicate rows
    │
    ▼
Exploratory Data Analysis
  • Churn distribution
  • Churn by contract, tenure, payment method, internet service
  • Churn by gender, senior citizen, tech support, online security
  • Monthly/total charges analysis
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
  • Random Forest        (class_weight=balanced)
  • Compare by Accuracy, Precision, Recall, F1, ROC-AUC
  • Select best model (ROC-AUC)
    │
    ▼
Predictions & Risk Labelling
  • ChurnProbability for every customer
  • Risk categories: Low / Medium / High
    │
    ▼
Feature Importance (Churn Drivers)
    │
    ▼
AI-style Business Insights
    │
    ▼
Executive Streamlit Dashboard (5 pages)
    │
    ▼
Business Decisions & Recommended Actions
```

---

## Machine Learning Models

### Logistic Regression
- A linear probabilistic classifier.
- Provides interpretable coefficients showing direction and magnitude of each feature's effect.
- `class_weight='balanced'` handles the class imbalance in churn data.
- `max_iter=1000` ensures convergence.

### Random Forest
- An ensemble of decision trees.
- Naturally captures non-linear relationships and feature interactions.
- Provides reliable feature importances.
- `class_weight='balanced'` and `max_depth=10` prevent overfitting.

### Model Selection
The **best model is selected by ROC-AUC** rather than accuracy.
- Churn prediction is a **business-risk problem** — failing to predict a churner (False Negative) is more costly than a false alarm (False Positive).
- ROC-AUC measures the model's ability to discriminate across all probability thresholds, making it the most appropriate business metric here.

---

## Evaluation Metrics

| Metric    | Why it matters for churn                                            |
|-----------|---------------------------------------------------------------------|
| Accuracy  | Overall correctness — can be misleading with class imbalance       |
| Precision | Of predicted churners, how many actually churn                     |
| Recall    | Of actual churners, how many did we catch (most critical for churn)|
| F1 Score  | Harmonic mean of Precision and Recall                              |
| ROC-AUC   | Discrimination ability across all thresholds — primary metric      |

---

## Dashboard Features

### Page 1 — Executive Overview
- KPI cards: Total Customers, Churn Rate, Retention Rate, Revenue, High-Risk count
- Churn distribution pie chart
- Churn by contract type
- Churn risk distribution
- Monthly charges vs churn box plot
- Executive insights tabs: Key Findings, Risks, Opportunities, Recommended Actions

### Page 2 — Churn & Customer Analysis
- Sidebar filters: Contract, Internet Service, Payment Method, Senior Citizen, Gender, Risk Level
- Churn by tenure, payment method, internet service, online security
- Monthly charges distribution by churn
- Tech support & churn analysis
- Gender and senior citizen churn analysis

### Page 3 — Risk, Opportunity & Action
- Risk section: High/Medium/Low risk customer counts, revenue at risk
- Risk distribution by contract, top churn drivers chart
- Opportunity section: paperless billing churn, tenure group analysis
- Action section: data-driven recommended business actions
- High-risk customer snapshot table (top 20)

### Page 4 — Customer Prediction
- Individual customer form with all relevant attributes
- Predicted churn probability (gauge chart)
- Risk category (Low / Medium / High)
- Key factors driving the prediction
- Personalised recommended retention action

### Page 5 — Model Performance
- Side-by-side model comparison table
- ROC curves for both models
- Confusion matrix for the best model
- Full classification report

---

## Installation Instructions

### Prerequisites
- Python 3.9 or higher
- pip

### Step 1 — Clone or Download the Project

```bash
# If using git
git clone <your-repo-url>
cd Telecom_Churn_Project

# Or just unzip the project folder and navigate into it
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download the Dataset

1. Go to https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Place it in the `Telecom_Churn_Project/` folder (same folder as the `.py` file)

---

## How to Run the Streamlit Application

```bash
# From inside the Telecom_Churn_Project/ folder
streamlit run ShivamSingh_TelecomCustomerChurn.py
```

The app will open in your browser at `http://localhost:8501`

**First run:** The model will train automatically (~30 seconds).
A `churn_model_pipeline.joblib` file will be saved to disk.
All subsequent runs will load the trained model instantly.

---

## How to Run the Jupyter Notebook

```bash
# From inside the Telecom_Churn_Project/ folder
jupyter notebook ShivamSingh_TelecomCustomerChurn.ipynb
```

Or open it directly in VS Code with the Jupyter extension.

The notebook contains the full analysis workflow:
1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis (with matplotlib/seaborn)
4. Feature Engineering & Preprocessing
5. Model Training & Evaluation
6. Feature Importance
7. Business Insights

---

## Expected Outputs

### Streamlit Dashboard
- Professional 5-page dashboard accessible at `http://localhost:8501`
- All charts are interactive (hover, zoom, filter)
- Sidebar filters update charts in real time
- Individual customer prediction with gauge chart

### Jupyter Notebook
- Step-by-step analysis with markdown explanations
- 15+ visualisations (bar charts, histograms, correlation heatmaps, ROC curves)
- Model evaluation tables
- Feature importance plots

### Saved Files
- `churn_model_pipeline.joblib` — trained model pipeline

---

## Project Structure

```
Telecom_Churn_Project/
├── ShivamSingh_TelecomCustomerChurn.py        ← Main Streamlit app
├── ShivamSingh_TelecomCustomerChurn.ipynb     ← Jupyter analysis notebook
├── ShivamSingh_TelecomCustomerChurnReport.docx← Project report
├── requirements.txt                           ← Python dependencies
├── README.md                                  ← This file
├── WA_Fn-UseC_-Telco-Customer-Churn.csv       ← Dataset (download from Kaggle)
└── churn_model_pipeline.joblib                ← Auto-generated after first run
```

---

## Key Business Questions Answered

1. **What is the overall churn rate?** — Shown as a KPI card on the overview page.
2. **Which contract type has the highest churn?** — Month-to-month contracts show significantly higher churn.
3. **Do newer customers churn more?** — Yes, customers with tenure ≤ 12 months have the highest churn rate.
4. **Does internet service type affect churn?** — Fiber optic customers churn at a higher rate than DSL customers.
5. **What is the monthly revenue at risk from potential churners?** — Calculated from high-risk customers' monthly charges.
6. **Which features are the strongest predictors of churn?** — Shown in the feature importance chart on Page 3.
7. **What specific retention actions should the business take?** — Generated from actual data patterns on Page 3.

---

## Disclaimer

- All analysis results are computed directly from the dataset. No values are hard-coded.
- Feature importance indicates association/correlation, not causal relationships.
- Model predictions are probabilistic — actual churn outcomes depend on many business factors.
- The project is intended for educational purposes as part of the IBM SkillsBuild Academic Internship.

---

*Made with ❤️ for IBM SkillsBuild Data Analytics with AI Academic Internship*
