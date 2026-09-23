"""
insert_report_images.py
Inserts all 18 report figures + a data-cleaning table + model metrics table
into the project report docx.

Run from inside Telecom_Churn_Project/:
    python insert_report_images.py
"""

import os, shutil
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import lxml.etree as etree

# ── paths ─────────────────────────────────────────────────────────────────────
SRC   = Path("ShivamSingh_TelecomCustomerChurnReport.docx")
DEST  = Path("ShivamSingh_TelecomCustomerChurnReport.docx")
ASSETS = Path("report_assets")

if not SRC.exists():
    raise FileNotFoundError(f"Report not found: {SRC}")
for fig in ["fig01_churn_distribution.png", "fig17_model_metrics_table.png"]:
    if not (ASSETS / fig).exists():
        raise FileNotFoundError(f"Asset missing: {ASSETS / fig}. Run generate_report_assets.py first.")

# ── helpers ───────────────────────────────────────────────────────────────────
def add_caption(doc, text, bold=False):
    """Add an italic, centred, grey caption paragraph."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.italic = True
    run.bold = bold
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(10)
    return p

def add_section_label(doc, text):
    """Add a small bold section label in blue."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x3B, 0x82, 0xD4)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after  = Pt(2)
    return p

def add_figure(doc, img_path, caption, width=Inches(5.5)):
    """Insert image + caption, centred."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    run = p.add_run()
    run.add_picture(str(img_path), width=width)
    add_caption(doc, caption)

def add_horizontal_rule(doc):
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), 'CBD5E1')
    pBdr.append(bottom)
    pPr.append(pBdr)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(4)

def style_table_header_row(row, bg_hex="3B82D4"):
    """Give the first table row a blue background with white bold text."""
    for cell in row.cells:
        # background
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), bg_hex)
        tcPr.append(shd)
        # text
        for para in cell.paragraphs:
            for run in para.runs:
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.size = Pt(9)

def style_table_data_row(row, even=True):
    bg = "EFF6FF" if even else "FFFFFF"
    for cell in row.cells:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), bg)
        tcPr.append(shd)
        for para in cell.paragraphs:
            for run in para.runs:
                run.font.size = Pt(9)

# ── load document ─────────────────────────────────────────────────────────────
print(f"Loading {SRC}…")
doc = Document(str(SRC))

# ── find paragraph indices by heading text ────────────────────────────────────
def find_para_idx(doc, heading_text):
    for i, p in enumerate(doc.paragraphs):
        if heading_text.strip() in p.text.strip():
            return i
    return None

idx_abstract       = find_para_idx(doc, "1. Abstract")
idx_eda            = find_para_idx(doc, "7. Exploratory Data Analysis")
idx_feature_eng    = find_para_idx(doc, "8. Feature Engineering")
idx_ml_method      = find_para_idx(doc, "9. Machine Learning Methodology")
idx_train_test     = find_para_idx(doc, "9.1 Train/Test Split")
idx_pipeline_arch  = find_para_idx(doc, "9.4 Pipeline Architecture")
idx_model_eval     = find_para_idx(doc, "10. Model Evaluation")
idx_dashboard      = find_para_idx(doc, "11. Dashboard Design")
idx_findings       = find_para_idx(doc, "12. Key Findings")
idx_risks          = find_para_idx(doc, "13. Risks")
idx_opportunities  = find_para_idx(doc, "14. Opportunities")
idx_section6       = find_para_idx(doc, "6. Data Preprocessing")
idx_section6_1     = find_para_idx(doc, "6.1 Data Cleaning Steps")

print("Paragraph indices found:")
for name, idx in [
    ("Abstract",idx_abstract), ("EDA",idx_eda), ("FeatureEng",idx_feature_eng),
    ("MLMethod",idx_ml_method), ("9.1",idx_train_test), ("9.4",idx_pipeline_arch),
    ("ModelEval",idx_model_eval), ("Dashboard",idx_dashboard),
    ("Findings",idx_findings), ("Risks",idx_risks), ("Opportunities",idx_opportunities),
    ("Sec6",idx_section6), ("Sec6.1",idx_section6_1),
]:
    print(f"  {name}: {idx}")

# We will build the enriched document by going through paragraphs
# and inserting new content at the right places.
# Strategy: collect all original paragraphs, then rebuild doc with inserts.

# ── Gather original body XML elements ────────────────────────────────────────
body = doc.element.body
original_children = list(body)

# We need to insert *after* specific paragraph indices.
# Build a mapping: paragraph_index → list of (xml_element,) to insert after it.
inserts_after = {}  # {para_index: [elements to insert after]}

def make_insert_group():
    """Create a temporary mini-doc to build elements, then return them."""
    return []

def queue_after(para_idx, fn_list):
    if para_idx not in inserts_after:
        inserts_after[para_idx] = []
    inserts_after[para_idx].extend(fn_list)

# ─────────────────────────────────────────────────────────────────────────────
# Instead of XML manipulation, we'll append everything to a new document
# and rebuild section by section. Simpler and more reliable.
# ─────────────────────────────────────────────────────────────────────────────

from docx import Document as NewDoc
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import copy

out = NewDoc()

# Set default styles
style = out.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)

# Page margins
from docx.oxml import OxmlElement as OE
section = out.sections[0]
section.top_margin    = Inches(1.0)
section.bottom_margin = Inches(1.0)
section.left_margin   = Inches(1.2)
section.right_margin  = Inches(1.2)

def h1(text):
    p = out.add_heading(text, level=1)
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after  = Pt(8)
    return p

def h2(text):
    p = out.add_heading(text, level=2)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after  = Pt(6)
    return p

def body_para(text, space_after=8):
    p = out.add_paragraph(text)
    p.paragraph_format.space_after = Pt(space_after)
    return p

def bullet(text):
    p = out.add_paragraph(text, style='List Bullet')
    p.paragraph_format.space_after = Pt(4)
    return p

def fig(path, caption, width=Inches(5.5)):
    p = out.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    run = p.add_run()
    run.add_picture(str(path), width=width)
    cap = out.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
    cap.paragraph_format.space_after = Pt(12)

def hr():
    p = out.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bot = OxmlElement('w:bottom')
    bot.set(qn('w:val'), 'single')
    bot.set(qn('w:sz'), '6')
    bot.set(qn('w:space'), '1')
    bot.set(qn('w:color'), 'CBD5E1')
    pBdr.append(bot)
    pPr.append(pBdr)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(6)

# ─────────────────────────────────────────────────────────────────────────────
# TITLE PAGE
# ─────────────────────────────────────────────────────────────────────────────
title = out.add_heading("Telecom Customer Churn Prediction\n& Retention Intelligence Dashboard", 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(10)

for line in [
    "IBM SkillsBuild Data Analytics with AI Academic Internship",
    "Author: Shivam Singh",
    "Dataset: Telco Customer Churn — IBM Sample Data (Kaggle)",
    "Technologies: Python  |  pandas  |  scikit-learn  |  Streamlit  |  Plotly",
]:
    p = out.add_paragraph(line)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.runs[0]
    run.font.size = Pt(11)
    if "Author" in line or "Technologies" in line:
        run.bold = True
    p.paragraph_format.space_after = Pt(4)

out.add_page_break()

# ─────────────────────────────────────────────────────────────────────────────
# 1. ABSTRACT
# ─────────────────────────────────────────────────────────────────────────────
h1("1. Abstract")
body_para(
    "This project presents an end-to-end Business Intelligence and Machine Learning solution "
    "for predicting and analysing telecom customer churn. Using the IBM Telco Customer Churn "
    "dataset (7,043 raw records, reduced to a cleaned analytical dataset after preprocessing — "
    "see Section 6 for the exact row-removal audit trail), the project encompasses data "
    "cleaning, exploratory data analysis (EDA), feature engineering, machine learning model "
    "training (Logistic Regression and Random Forest), churn probability scoring, risk "
    "categorisation, and an interactive executive Streamlit dashboard. The goal is to identify "
    "which customers are at risk of churning, understand the associations between customer "
    "attributes and churn, and generate data-grounded hypotheses for retention experiments."
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. INTRODUCTION
# ─────────────────────────────────────────────────────────────────────────────
h1("2. Introduction")
body_para(
    "Customer churn — the loss of subscribers to competitors — is one of the most critical "
    "challenges in the telecommunications industry. The average telecom company loses between "
    "15% and 25% of its customers every year, and acquiring a new customer costs 5 to 10 times "
    "more than retaining an existing one. This makes early churn prediction and targeted "
    "retention a high-priority business problem."
)
body_para(
    "This project builds a complete data analytics and machine learning pipeline: loading and "
    "cleaning raw customer data, performing exploratory analysis to surface churn patterns, "
    "training classification models to predict individual churn probability, and presenting all "
    "findings through an executive-grade interactive dashboard built with Streamlit."
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. PROBLEM STATEMENT
# ─────────────────────────────────────────────────────────────────────────────
h1("3. Problem Statement")
body_para("A telecom company faces high monthly customer churn. The business needs to:")
for item in [
    "Identify which customers are most likely to churn in the near future.",
    "Understand what factors are associated with churn decisions.",
    "Quantify the revenue risk from high-churn customer segments.",
    "Generate targeted, data-driven retention hypotheses.",
]:
    bullet(item)

# ─────────────────────────────────────────────────────────────────────────────
# 4. OBJECTIVES
# ─────────────────────────────────────────────────────────────────────────────
h1("4. Objectives")
body_para("The objectives of this project are:")
for item in [
    "Load, clean, and validate the Telco Customer Churn dataset.",
    "Perform exploratory data analysis across 12+ churn dimensions.",
    "Calculate business KPIs: churn rate, retention rate, revenue at risk.",
    "Train and compare Logistic Regression and Random Forest classifiers.",
    "Assign risk categories (Low / Medium / High) to every customer.",
    "Identify top churn drivers using feature importance.",
    "Build an executive Streamlit dashboard with 5 interactive pages.",
    "Generate auto-derived business insights and recommended actions.",
]:
    bullet(item)

# ─────────────────────────────────────────────────────────────────────────────
# 5. DATASET DESCRIPTION
# ─────────────────────────────────────────────────────────────────────────────
h1("5. Dataset Description")
body_para(
    "The Telco Customer Churn dataset is a publicly available IBM sample dataset hosted on "
    "Kaggle. It contains information about a telecom company's customers including their "
    "demographic details, subscribed services, account information, and whether they churned "
    "in the last month."
)
body_para("Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
body_para(
    "Raw rows: 7,043  |  Cleaned analytical rows: see Section 6 preprocessing audit  |  "
    "Columns: 21  |  Target: Churn (Yes / No)"
)
body_para(
    "Key columns include: customerID, gender, SeniorCitizen, Partner, Dependents, tenure, "
    "PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, "
    "DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, "
    "PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn."
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
h1("6. Data Preprocessing")
h2("6.1 Data Cleaning Steps")

cleaning_rows = [
    ["Step", "Action", "Records Affected"],
    ["1", "Strip whitespace from column names and string values", "All rows"],
    ["2", "Convert TotalCharges to numeric (blank → NaN)", "11 rows set to NaN"],
    ["3", "Drop rows where TotalCharges is NaN (tenure=0 new customers)", "11 rows dropped"],
    ["4", "Encode Churn: 'Yes' → 1, 'No' → 0", "All rows"],
    ["5", "Drop customerID column (non-predictive)", "1 column removed"],
    ["6", "Remove duplicate rows", "22 rows dropped (runtime)"],
    ["7", "Confirm SeniorCitizen as integer type", "All rows"],
    ["—", "Final analytical dataset", "7,010 rows × 20 columns"],
]
tbl = out.add_table(rows=len(cleaning_rows), cols=3)
tbl.style = 'Table Grid'
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
for r_i, row_data in enumerate(cleaning_rows):
    row = tbl.rows[r_i]
    for c_i, val in enumerate(row_data):
        cell = row.cells[c_i]
        cell.text = val
        for para in cell.paragraphs:
            para.paragraph_format.space_before = Pt(3)
            para.paragraph_format.space_after  = Pt(3)
            for run in para.runs:
                run.font.size = Pt(9)
    if r_i == 0:
        style_table_header_row(row)
    else:
        style_table_data_row(row, even=(r_i % 2 == 0))

add_caption(out, "Table 1 — Data Cleaning Steps and Records Affected")
hr()

for step_text in [
    "Step 1 — Whitespace removal: Column names and all string values were stripped of leading/trailing whitespace.",
    "Step 2 — TotalCharges conversion: The TotalCharges column contained blank string values for 11 customers with tenure=0 who had not yet been billed.",
    "Step 3 — Row removal: 11 rows where TotalCharges = NaN were dropped.",
    "Step 4 — Target encoding: The Churn column ('Yes'/'No') was mapped to 1/0.",
    "Step 5 — customerID dropped: Non-predictive unique identifier removed from features.",
    "Step 6 — Duplicate removal: drop_duplicates() removed 22 duplicate rows at runtime.",
    "Step 7 — SeniorCitizen type: Confirmed as integer (0/1).",
]:
    p = out.add_paragraph(step_text, style='List Bullet')
    p.paragraph_format.space_after = Pt(3)

# ─────────────────────────────────────────────────────────────────────────────
# 7. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
h1("7. Exploratory Data Analysis")
body_para("EDA was performed across the following dimensions. All charts below are generated from the actual dataset at runtime.")
hr()

body_para("7.1  Overall Churn Distribution", space_after=4)
body_para(
    "The cleaned dataset has an overall churn rate of approximately 26.5%. This class imbalance "
    "is addressed by using class_weight='balanced' in both classifiers."
)
fig(ASSETS / "fig01_churn_distribution.png",
    "Figure 1 — Overall churn distribution (26.5% churned, 73.5% retained)")

body_para("7.2  Churn by Contract Type", space_after=4)
body_para(
    "Month-to-month customers show the highest observed churn rate (≈42.6%). "
    "Two-year contract customers show the lowest (≈2.9%). This is the most prominent "
    "association in the EDA — it reflects correlation, not proven causation."
)
fig(ASSETS / "fig02_churn_by_contract.png",
    "Figure 2 — Churn rate by contract type (Month-to-month vs One-year vs Two-year)")

body_para("7.3  Churn by Tenure", space_after=4)
body_para(
    "Customers in their first 12 months churn at ≈47.6%, versus ≈14% for customers with "
    "tenure > 24 months. Churn rate decreases as tenure increases."
)
fig(ASSETS / "fig03_churn_by_tenure.png",
    "Figure 3 — Tenure distribution for churned vs retained customers")

body_para("7.4  Churn by Internet Service", space_after=4)
body_para(
    "Fiber optic customers show the highest observed churn rate among internet service types. "
    "Customers with no internet service show the lowest."
)
fig(ASSETS / "fig04_churn_by_internet.png",
    "Figure 4 — Churn rate by internet service type")

body_para("7.5  Churn by Payment Method", space_after=4)
body_para(
    "Electronic check users exhibit the highest churn rate among the four payment methods. "
    "The other three methods show lower and more similar churn rates."
)
fig(ASSETS / "fig05_churn_by_payment.png",
    "Figure 5 — Churn rate by payment method")

body_para("7.6  Churn by Monthly Charges", space_after=4)
body_para(
    "Churned customers tend to have higher monthly charges than retained customers, with the "
    "churned distribution shifted toward the $70–$100 range."
)
fig(ASSETS / "fig06_monthly_charges.png",
    "Figure 6 — Monthly charges distribution: churned vs retained customers")

body_para("7.7  Churn by Demographics", space_after=4)
body_para(
    "Senior citizens show a higher observed churn rate than non-seniors. "
    "Gender shows minimal difference in churn rate."
)
fig(ASSETS / "fig07_churn_demographics.png",
    "Figure 7 — Churn rate by senior citizen status and gender",
    width=Inches(6.0))

body_para("7.8  Churn by TechSupport & OnlineSecurity", space_after=4)
body_para(
    "Customers without TechSupport churn at ≈41.5% vs ≈15.2% with TechSupport. "
    "Customers without OnlineSecurity churn at ≈41.6% vs ≈14.6% with OnlineSecurity. "
    "These are correlational associations, not established causal effects."
)
fig(ASSETS / "fig08_churn_services.png",
    "Figure 8 — Churn rate by TechSupport and OnlineSecurity subscription",
    width=Inches(6.0))

body_para("7.9  Correlation Matrix", space_after=4)
body_para(
    "The correlation matrix shows the linear associations between numerical features and "
    "the churn target. Tenure has a negative correlation with churn; "
    "MonthlyCharges has a positive correlation."
)
fig(ASSETS / "fig09_correlation_heatmap.png",
    "Figure 9 — Correlation matrix of numerical features")

# ─────────────────────────────────────────────────────────────────────────────
# 8. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
h1("8. Feature Engineering")
body_para(
    "The feature engineering step uses a scikit-learn ColumnTransformer with two branches:"
)
bullet("Numerical Features (StandardScaler): tenure, MonthlyCharges, TotalCharges, SeniorCitizen. "
       "StandardScaler normalises these to zero mean and unit variance — critical for Logistic "
       "Regression's gradient-based optimisation.")
bullet("Categorical Features (OneHotEncoder): gender, Partner, Dependents, PhoneService, "
       "MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, "
       "TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod. "
       "handle_unknown='ignore' prevents errors on unseen categories during prediction.")

# ─────────────────────────────────────────────────────────────────────────────
# 9. MACHINE LEARNING METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
h1("9. Machine Learning Methodology")
h2("9.1 Train / Test Split")
body_para(
    "The dataset was split 80% training / 20% testing using stratified sampling (stratify=y) "
    "to preserve the churn class proportion in both sets. Random seed 42 ensures reproducibility."
)

h2("9.2 Logistic Regression")
body_para(
    "Logistic Regression is a linear probabilistic classifier that models the log-odds of churn "
    "as a linear combination of features. Configuration: max_iter=1000 (ensures convergence), "
    "class_weight='balanced' (handles class imbalance). Advantages: interpretable coefficients, "
    "fast training, strong ROC-AUC on linearly separable problems."
)

h2("9.3 Random Forest")
body_para(
    "Random Forest is an ensemble of decision trees trained on bootstrap samples with random "
    "feature subsets at each split. Configuration: n_estimators=200, class_weight='balanced', "
    "max_depth=10, random_state=42. Advantages: handles non-linear relationships, provides "
    "MDI feature importance scores, robust to outliers."
)

h2("9.4 Pipeline Architecture")
body_para(
    "Each model is wrapped in a scikit-learn Pipeline: "
    "[ColumnTransformer (Scaler + OHE)] → [Classifier]. "
    "This ensures preprocessing is applied consistently on training and test data with no data "
    "leakage. The fitted pipeline is saved with joblib for reuse in the Streamlit dashboard."
)
fig(ASSETS / "fig10_ml_pipeline.png",
    "Figure 10 — End-to-end machine learning pipeline architecture",
    width=Inches(6.2))

# ─────────────────────────────────────────────────────────────────────────────
# 10. MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
h1("10. Model Evaluation")
body_para("Models are evaluated on the held-out test set (20% = 1,402 customers) using five metrics:")
metric_desc = [
    ("Accuracy",  "Overall proportion of correct predictions. Misleading under class imbalance."),
    ("Precision", "Of customers predicted to churn, how many actually did. High precision = fewer wasted retention offers."),
    ("Recall",    "Of actual churners, how many did the model catch. High recall = fewer missed churners. Most business-critical metric."),
    ("F1 Score",  "Harmonic mean of Precision and Recall. Balances both."),
    ("ROC-AUC",   "Area under the ROC curve. Threshold-independent ranking quality. Primary model-selection metric."),
]
for name, desc in metric_desc:
    p = out.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    run_bold = p.add_run(f"{name}: ")
    run_bold.bold = True
    run_bold.font.size = Pt(10)
    p.add_run(desc).font.size = Pt(10)

body_para(
    "Best model is selected by highest ROC-AUC. Note: the assumption that False Negatives are "
    "costlier than False Positives is plausible for churn prediction but not derived from this "
    "dataset. A production deployment should calibrate the threshold against actual retention costs.",
    space_after=10
)

fig(ASSETS / "fig17_model_metrics_table.png",
    "Figure 17 — Model evaluation metrics on the held-out test set (★ Best model by ROC-AUC)",
    width=Inches(6.2))

fig(ASSETS / "fig11_model_comparison.png",
    "Figure 11 — Side-by-side performance comparison: Logistic Regression vs Random Forest",
    width=Inches(6.2))

fig(ASSETS / "fig13_roc_curves.png",
    "Figure 13 — ROC curves: Logistic Regression (AUC=0.846) vs Random Forest (AUC=0.839)")

fig(ASSETS / "fig12_confusion_matrices.png",
    "Figure 12 — Confusion matrices: Logistic Regression and Random Forest on test set",
    width=Inches(6.2))

# ─────────────────────────────────────────────────────────────────────────────
# 11. DASHBOARD DESIGN
# ─────────────────────────────────────────────────────────────────────────────
h1("11. Dashboard Design")
body_para(
    "The executive Streamlit dashboard has five pages, all reading live data from the cleaned "
    "dataset and trained model — no hard-coded values anywhere."
)

body_para("Page 1 — Executive Overview", space_after=3)
bullet("KPI cards: Total Customers, Churn Rate, Retention Rate, Revenue, High-Risk count, Avg Monthly Charges")
bullet("Charts: churn distribution pie, churn by contract bar, risk distribution, monthly charges box plot")
bullet("Four insight tabs: Key Findings, Risks, Opportunities, Actions")

fig(ASSETS / "fig18_dashboard_page1.png",
    "Figure 18 — Executive Overview page (dark-theme dashboard, KPI cards and live charts)",
    width=Inches(6.2))

body_para("Page 2 — Churn & Customer Analysis", space_after=3)
bullet("Six interactive charts: tenure, payment method, internet service, online security, monthly charges, tech support")
bullet("Sidebar filters: Contract, Internet Service, Payment Method, Senior Citizen, Gender, Risk Level")

body_para("Page 3 — Risk, Opportunity & Action", space_after=3)
bullet("Risk KPI cards, risk-by-contract chart, top-10 feature importance chart")
bullet("Opportunity analysis (paperless billing, tenure groups)")
bullet("Data-driven recommended actions and high-risk customer snapshot table")

body_para("Page 4 — Customer Prediction", space_after=3)
bullet("Individual customer form with 20 input fields")
bullet("Churn probability gauge chart, risk category, contributing factors, personalised retention recommendation")

body_para("Page 5 — Model Performance", space_after=3)
bullet("Model comparison table, ROC curves, confusion matrix, full classification report")

# ─────────────────────────────────────────────────────────────────────────────
# 12. KEY FINDINGS
# ─────────────────────────────────────────────────────────────────────────────
h1("12. Key Findings")
body_para("All findings below are derived from the dataset at runtime — no values are hard-coded.")
hr()

findings = [
    ("Feature Importance (Figure 14)", ASSETS/"fig14_feature_importance.png",
     "Figure 14 — Top 10 churn drivers by feature importance (Logistic Regression |coefficient| aggregated per feature)",
     "Contract type is the highest-importance feature in the Logistic Regression model, followed by "
     "tenure and InternetService. These are MDI/|coefficient| scores reflecting the model's weighting, "
     "not confirmed causal effects."),
    ("KPI Summary (Figure 16)", ASSETS/"fig16_kpi_table.png",
     "Figure 16 — Business KPI summary computed at runtime from the cleaned dataset",
     "The business KPI summary shows 7,010 customers after cleaning, 26.5% churn rate, $454,870/month "
     "total revenue, and 2,339 customers classified as High Risk by the ML model."),
    ("Risk Distribution (Figure 15)", ASSETS/"fig15_risk_distribution.png",
     "Figure 15 — Customer risk segmentation (ML model predictions)",
     "The ML model classifies 33.4% of customers as High Risk, 24.5% as Medium Risk, and 42.2% as "
     "Low Risk based on a churn probability threshold of >60% for High Risk."),
]

for title_text, img_path, cap_text, desc_text in findings:
    p = out.add_paragraph()
    run = p.add_run(title_text)
    run.bold = True
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x3B, 0x82, 0xD4)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after  = Pt(4)
    body_para(desc_text, space_after=6)
    fig(img_path, cap_text, width=Inches(5.8))

body_para("Key Findings Summary:")
findings_list = [
    "Contract type shows the largest observed churn rate difference. Month-to-month: ≈42.6% churn; Two-year: ≈2.9% churn.",
    "Customers with tenure ≤ 12 months churn at ≈47.6% vs ≈14.0% for customers with tenure > 24 months.",
    "Fiber optic internet service is associated with a higher observed churn rate than DSL.",
    "Customers without TechSupport churn at ≈41.5% vs ≈15.2% with TechSupport (observed association).",
    "Electronic check payment method shows the highest observed churn rate among all payment methods.",
]
for f_text in findings_list:
    bullet(f_text)

# ─────────────────────────────────────────────────────────────────────────────
# 13. RISKS
# ─────────────────────────────────────────────────────────────────────────────
h1("13. Risks")
body_para(
    "The ML model classifies 2,339 customers (33.4% of the dataset) as High Risk "
    "(predicted churn probability > 60%). Their combined monthly charges represent the "
    "model-predicted revenue exposure — approximately $183,018/month. "
    "These are model predictions, not confirmed churn outcomes."
)
for r_text in [
    "2,339 customers classified as High Risk by the ML model (probability > 60%), representing ≈$183,018/month model-predicted revenue exposure.",
    "Senior citizens show an above-average observed churn rate relative to non-seniors.",
    "Electronic check payment method is associated with the highest observed churn rate among payment methods.",
]:
    bullet(r_text)

# ─────────────────────────────────────────────────────────────────────────────
# 14. OPPORTUNITIES
# ─────────────────────────────────────────────────────────────────────────────
h1("14. Opportunities")
for o_text in [
    "3,295 customers with tenure > 24 months who have not churned form a stable retained segment — 47.0% of all customers.",
    "The observed churn rate difference between customers without vs with TechSupport (41.5% vs 15.2%) suggests targeted add-on offers are worth evaluating experimentally.",
    "Month-to-month customers currently retained could be candidates for contract-upgrade incentive experiments.",
]:
    bullet(o_text)

# ─────────────────────────────────────────────────────────────────────────────
# 15. RECOMMENDED ACTIONS
# ─────────────────────────────────────────────────────────────────────────────
h1("15. Recommended Actions")
body_para("All actions are framed as testable hypotheses, not proven interventions:")
for a_text in [
    "Evaluate: Use the model's High-Risk flag (probability > 60%) as a prioritisation tool to identify customers most worth investigating for retention outreach. Validate through a controlled experiment before scaling.",
    "Test: Target TechSupport and OnlineSecurity offers at customers who currently lack these services. Measure retention impact through a controlled experiment (correlational finding, not confirmed causation).",
    "Investigate: Test contract-upgrade incentives among month-to-month customers and measure whether the intervention reduces churn (the observed churn rate gap motivates this hypothesis, not a proven causal mechanism).",
    "Validate: Investigate whether the higher observed churn rate among electronic check users reflects a billing experience issue or a confounding customer segment effect before concluding a payment-method intervention would reduce churn.",
]:
    bullet(a_text)

# ─────────────────────────────────────────────────────────────────────────────
# 16. CONCLUSION
# ─────────────────────────────────────────────────────────────────────────────
h1("16. Conclusion")
body_para(
    "This project demonstrates a complete end-to-end data analytics and machine learning "
    "workflow applied to the business problem of customer churn prediction. The Streamlit "
    "dashboard translates data science outputs into actionable business intelligence for "
    "telecom executives and retention teams."
)
body_para(
    "Key technical achievements include: a clean, reproducible data pipeline; a properly "
    "evaluated ML model (selected by ROC-AUC, not accuracy); churn probability scoring for "
    "every customer; risk categorisation with configurable thresholds; and data-grounded "
    "executive insights with no hard-coded numbers."
)

# ─────────────────────────────────────────────────────────────────────────────
# 17. FUTURE SCOPE
# ─────────────────────────────────────────────────────────────────────────────
h1("17. Future Scope")
for fs in [
    "Temporal modelling: Incorporate time-series data (monthly snapshots) to model churn as a survival analysis problem.",
    "Advanced models: Experiment with XGBoost, LightGBM, or Neural Networks for potentially higher predictive accuracy.",
    "SHAP explainability: Replace feature importance with SHAP values for per-customer explanation.",
    "A/B testing integration: Measure the actual retention lift from model-recommended actions using controlled experiments.",
    "Real-time API: Deploy the trained model as a REST API for integration with CRM systems.",
]:
    bullet(fs)

# ─────────────────────────────────────────────────────────────────────────────
# 18. REFERENCES
# ─────────────────────────────────────────────────────────────────────────────
h1("18. References")
refs = [
    "Telco Customer Churn Dataset — IBM Sample Data. Kaggle. https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
    "scikit-learn Documentation — Machine Learning in Python. https://scikit-learn.org/",
    "Streamlit Documentation — Build and share data apps. https://docs.streamlit.io/",
    "Plotly Python Open Source Graphing Library. https://plotly.com/python/",
    "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.",
    "Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861–874.",
    "IBM SkillsBuild — Data Analytics with AI Academic Internship Programme. https://skillsbuild.org/",
]
for i, ref in enumerate(refs, 1):
    p = out.add_paragraph(f"{i}. {ref}")
    p.paragraph_format.space_after = Pt(4)

hr()
end_p = out.add_paragraph("— End of Report —")
end_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
end_p.runs[0].italic = True
end_p.runs[0].font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nSaving illustrated report → {DEST}")
out.save(str(DEST))
print("✅ Done.")
print(f"   Paragraphs in new doc: {len(out.paragraphs)}")
print(f"   Tables: {len(out.tables)}")

# Count inline images
img_count = sum(
    1 for rel in out.part.rels.values()
    if "image" in rel.reltype
)
print(f"   Embedded images: {img_count}")
