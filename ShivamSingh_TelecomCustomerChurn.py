"""
=============================================================================
Telecom Customer Churn Prediction & Retention Intelligence Dashboard
=============================================================================
Project    : IBM SkillsBuild Data Analytics with AI Academic Internship
Author     : Shivam Singh
Dataset    : Telco Customer Churn (Kaggle)
             https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Description: End-to-end BI + ML project: data cleaning, EDA, churn prediction,
             feature importance, and an executive Streamlit dashboard.
=============================================================================
"""

import os
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------------------------
DATASET_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH   = "churn_model_pipeline.joblib"

# Churn probability risk thresholds (configurable)
LOW_RISK_THRESHOLD    = 0.30
MEDIUM_RISK_THRESHOLD = 0.60

TARGET_COL  = "Churn"
DROP_COLS   = ["customerID"]

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]
NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the Telco Customer Churn CSV dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            "Place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in the same folder as this script."
        )
    df = pd.read_csv(path)
    return df

# ---------------------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning steps:
      1. Strip whitespace from column names and string values.
      2. Convert TotalCharges to numeric (some rows have ' ' strings).
      3. Drop rows where TotalCharges is still NaN after conversion.
      4. Encode target variable: Churn -> 0/1.
      5. Drop customerID (not a predictor).
      6. Remove exact duplicate rows.
      7. Ensure SeniorCitizen is int.
    """
    df = df.copy()

    # Step 1 – strip whitespace
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Step 2 – TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Step 3 – drop NaN TotalCharges (11 rows with ' ')
    before = len(df)
    df.dropna(subset=["TotalCharges"], inplace=True)
    dropped = before - len(df)

    # Step 4 – encode target
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    # Step 5 – drop non-feature columns
    for col in DROP_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Step 6 – remove duplicates
    df.drop_duplicates(inplace=True)

    # Step 7 – SeniorCitizen as int
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    return df

# ---------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS HELPERS
# ---------------------------------------------------------------------------

def churn_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return a DataFrame with churn count and rate per category of col."""
    grp = df.groupby(col)[TARGET_COL].agg(["sum", "count"]).reset_index()
    grp.columns = [col, "Churned", "Total"]
    grp["ChurnRate"] = (grp["Churned"] / grp["Total"] * 100).round(2)
    grp["Retained"] = grp["Total"] - grp["Churned"]
    return grp

# ---------------------------------------------------------------------------
# 4. KPI CALCULATIONS
# ---------------------------------------------------------------------------

def calculate_kpis(df: pd.DataFrame, high_risk_threshold: float = MEDIUM_RISK_THRESHOLD) -> dict:
    """Calculate all business KPIs from the cleaned dataset."""
    total   = len(df)
    churned = df[TARGET_COL].sum()
    retained = total - churned
    churn_rate     = churned / total * 100
    retention_rate = retained / total * 100
    avg_monthly    = df["MonthlyCharges"].mean()
    avg_tenure     = df["tenure"].mean()
    total_monthly_revenue = df["MonthlyCharges"].sum()

    kpis = {
        "Total Customers"       : total,
        "Churned Customers"     : int(churned),
        "Retained Customers"    : int(retained),
        "Churn Rate (%)"        : round(churn_rate, 2),
        "Retention Rate (%)"    : round(retention_rate, 2),
        "Avg Monthly Charges"   : round(avg_monthly, 2),
        "Avg Tenure (months)"   : round(avg_tenure, 2),
        "Total Monthly Revenue" : round(total_monthly_revenue, 2),
    }
    return kpis

# ---------------------------------------------------------------------------
# 5. MACHINE LEARNING – PREPROCESSING & TRAINING
# ---------------------------------------------------------------------------

def preprocess_data(df: pd.DataFrame):
    """
    Split features/target, build a ColumnTransformer pipeline for
    one-hot encoding of categorical columns and standard scaling of
    numerical columns.

    Returns: X_train, X_test, y_train, y_test, preprocessor
    """
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL]

    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_cols = [c for c in NUMERICAL_COLS  if c in X.columns]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(),                                  num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols


def train_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train Logistic Regression and Random Forest classifiers.
    Returns evaluation metrics dict and the best pipeline.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    results   = {}
    pipelines = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   model)
        ])
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        results[name] = {
            "Accuracy" : round(accuracy_score(y_test, y_pred),  4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall"   : round(recall_score(y_test, y_pred),    4),
            "F1 Score" : round(f1_score(y_test, y_pred),        4),
            "ROC-AUC"  : round(roc_auc_score(y_test, y_proba),  4),
            "y_pred"   : y_pred,
            "y_proba"  : y_proba,
        }
        pipelines[name] = pipe

    # Select best model by ROC-AUC (business-risk metric)
    best_name = max(results, key=lambda k: results[k]["ROC-AUC"])
    return results, pipelines, best_name


def save_pipeline(pipeline, path: str = MODEL_PATH):
    """Persist the trained pipeline to disk using joblib."""
    joblib.dump(pipeline, path)


def load_pipeline(path: str = MODEL_PATH):
    """Load a previously saved pipeline from disk."""
    if not os.path.exists(path):
        return None
    return joblib.load(path)

# ---------------------------------------------------------------------------
# 6. CHURN PROBABILITY & RISK LABELLING
# ---------------------------------------------------------------------------

def generate_predictions(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    """
    Predict churn probability for every customer in df.
    Adds columns: ChurnProbability, PredictedChurn, RiskCategory.
    """
    feature_cols = [c for c in df.columns if c not in [TARGET_COL]]
    X = df[feature_cols]
    proba = pipeline.predict_proba(X)[:, 1]
    pred  = (proba >= 0.50).astype(int)

    out = df.copy()
    out["ChurnProbability"] = proba.round(4)
    out["PredictedChurn"]   = pred
    out["RiskCategory"] = pd.cut(
        proba,
        bins   = [-0.001, LOW_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, 1.001],
        labels = ["Low Risk", "Medium Risk", "High Risk"]
    )
    return out

# ---------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------

def get_feature_importance(pipeline, cat_cols: list, num_cols: list) -> pd.DataFrame:
    """Extract feature importances from the Random Forest inside the pipeline."""
    clf = pipeline.named_steps["classifier"]
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    ohe_cols = list(ohe.get_feature_names_out(cat_cols))
    all_features = num_cols + ohe_cols

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        # Logistic Regression – use absolute coefficients
        importances = np.abs(clf.coef_[0])

    fi_df = pd.DataFrame({
        "Feature"   : all_features,
        "Importance": importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Collapse OHE features back to original name for readability
    fi_df["OriginalFeature"] = fi_df["Feature"].apply(
        lambda x: x.split("_")[0] if any(c in x for c in cat_cols) else x
    )
    fi_agg = (
        fi_df.groupby("OriginalFeature")["Importance"]
        .sum()
        .reset_index()
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    return fi_agg

# ---------------------------------------------------------------------------
# 8. BUSINESS INSIGHTS GENERATOR
# ---------------------------------------------------------------------------

def generate_business_insights(df: pd.DataFrame, kpis: dict, fi_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """
    Auto-generate executive insights from the data.
    All numerical statements come from the dataset — no hard-coding.
    """
    # Contract analysis
    contract_churn = churn_rate_by(df, "Contract")
    highest_churn_contract = contract_churn.loc[contract_churn["ChurnRate"].idxmax(), "Contract"]
    highest_churn_contract_rate = contract_churn["ChurnRate"].max()

    # Tenure analysis – low tenure churn
    df_lt12 = df[df["tenure"] <= 12]
    short_tenure_churn = df_lt12[TARGET_COL].mean() * 100

    # Internet service analysis
    internet_churn = churn_rate_by(df, "InternetService")
    highest_internet_service = internet_churn.loc[internet_churn["ChurnRate"].idxmax(), "InternetService"]
    highest_internet_rate = internet_churn["ChurnRate"].max()

    # Senior citizen churn
    senior_churn = df[df["SeniorCitizen"] == 1][TARGET_COL].mean() * 100
    non_senior_churn = df[df["SeniorCitizen"] == 0][TARGET_COL].mean() * 100

    # Top churn driver
    top_feature = fi_df.iloc[0]["OriginalFeature"] if not fi_df.empty else "tenure"

    # Revenue at risk
    high_risk = pred_df[pred_df["RiskCategory"] == "High Risk"]
    revenue_at_risk = high_risk["MonthlyCharges"].sum()
    high_risk_count = len(high_risk)

    # Payment method churn
    pay_churn = churn_rate_by(df, "PaymentMethod")
    highest_pay_method = pay_churn.loc[pay_churn["ChurnRate"].idxmax(), "PaymentMethod"]
    highest_pay_rate = pay_churn["ChurnRate"].max()

    # Retention opportunity – loyal customers (tenure > 24, no churn)
    loyal = df[(df["tenure"] > 24) & (df[TARGET_COL] == 0)]
    loyal_count = len(loyal)

    insights = {
        "key_findings": [
            f"Customers on {highest_churn_contract} contracts have the highest churn rate at {highest_churn_contract_rate:.1f}%.",
            f"Customers with tenure ≤ 12 months churn at {short_tenure_churn:.1f}% — early-stage retention is critical.",
            f"{highest_internet_service} internet service customers show the highest churn rate at {highest_internet_rate:.1f}%.",
            f"Senior citizens churn at {senior_churn:.1f}% compared to {non_senior_churn:.1f}% for non-seniors.",
            f"'{top_feature}' is the strongest predictor of churn based on model feature importance.",
        ],
        "risks": [
            f"{high_risk_count:,} customers ({high_risk_count/len(df)*100:.1f}%) are classified as High Risk with churn probability > {MEDIUM_RISK_THRESHOLD}.",
            f"Estimated monthly revenue at risk from high-risk customers: ${revenue_at_risk:,.0f}.",
            f"Customers paying via {highest_pay_method} have the highest churn rate at {highest_pay_rate:.1f}% — this payment segment requires immediate attention.",
        ],
        "opportunities": [
            f"{loyal_count:,} long-tenure loyal customers (tenure > 24 months, currently retained) can be targeted for upselling.",
            "Customers without tech support or online security show significantly higher churn — these add-on services represent a retention lever.",
            "Month-to-month customers who upgrade to annual or two-year contracts show substantially lower churn — targeted contract upgrade campaigns can improve retention.",
        ],
        "actions": [
            f"Launch a proactive 90-day retention program for all {high_risk_count:,} high-risk customers — prioritize those on month-to-month contracts.",
            "Offer discounted tech support and online security bundles to customers who currently have neither — this addresses a key churn driver.",
            f"Introduce contract upgrade incentives specifically for {highest_churn_contract} contract customers — even a 3-month discount for switching to annual contracts can reduce churn significantly.",
            "Investigate why customers using electronic check have higher churn — consider offering payment method switch incentives or improving billing experience.",
        ],
    }
    return insights

# ---------------------------------------------------------------------------
# STREAMLIT DASHBOARD
# ---------------------------------------------------------------------------

def setup_page():
    st.set_page_config(
        page_title="Telecom Churn Intelligence",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Custom CSS for professional look
    st.markdown("""
    <style>
    /* Main background */
    .main { background-color: #f8f9fb; }
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #3b6fd4;
        margin-bottom: 12px;
    }
    .metric-card.red  { border-left-color: #e74c3c; }
    .metric-card.green{ border-left-color: #27ae60; }
    .metric-card.orange{border-left-color: #f39c12; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1a1a2e; }
    .metric-label { font-size: 13px; color: #6c757d; margin-top: 2px; }
    /* Section headers */
    .section-header {
        font-size: 20px; font-weight: 700;
        color: #1a1a2e; border-bottom: 2px solid #3b6fd4;
        padding-bottom: 6px; margin: 24px 0 16px 0;
    }
    /* Insight boxes */
    .insight-box {
        background: #eef2ff; border-radius: 8px;
        padding: 14px 18px; margin: 8px 0;
        border-left: 4px solid #3b6fd4;
        font-size: 14px; color: #1a1a2e;
    }
    .risk-box    { background: #fff0f0; border-left-color: #e74c3c; }
    .oppty-box   { background: #f0fff4; border-left-color: #27ae60; }
    .action-box  { background: #fffbf0; border-left-color: #f39c12; }
    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


def kpi_card(label: str, value, color: str = "blue", suffix: str = ""):
    color_map = {"blue": "", "red": "red", "green": "green", "orange": "orange"}
    cls = color_map.get(color, "")
    st.markdown(f"""
    <div class="metric-card {cls}">
        <div class="metric-value">{value}{suffix}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ---- PAGE 1: Executive Overview ----

def page_overview(df, kpis, pred_df, insights):
    st.title("📡 Telecom Customer Churn Intelligence")
    st.markdown("**IBM SkillsBuild Data Analytics with AI Academic Internship**  |  *Executive Overview*")
    st.markdown("---")

    # KPI Row
    section_header("Key Performance Indicators")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi_card("Total Customers",       f"{kpis['Total Customers']:,}", "blue")
    with c2: kpi_card("Churned Customers",      f"{kpis['Churned Customers']:,}", "red")
    with c3: kpi_card("Churn Rate",             f"{kpis['Churn Rate (%)']}", "red", "%")
    with c4: kpi_card("Retention Rate",         f"{kpis['Retention Rate (%)']}", "green", "%")
    with c5: kpi_card("Monthly Revenue",        f"${kpis['Total Monthly Revenue']:,.0f}", "blue")
    with c6:
        high_risk_count = len(pred_df[pred_df["RiskCategory"] == "High Risk"])
        kpi_card("High-Risk Customers", f"{high_risk_count:,}", "orange")

    c7, c8 = st.columns(2)
    with c7: kpi_card("Avg Monthly Charges", f"${kpis['Avg Monthly Charges']}", "blue")
    with c8: kpi_card("Avg Tenure (months)", f"{kpis['Avg Tenure (months)']}", "blue")

    st.markdown("---")

    # Charts Row 1
    col_a, col_b = st.columns(2)

    with col_a:
        section_header("Churn Distribution")
        churn_counts = df[TARGET_COL].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        churn_counts["Label"] = churn_counts["Churn"].map({1: "Churned", 0: "Retained"})
        fig = px.pie(
            churn_counts, values="Count", names="Label",
            color="Label",
            color_discrete_map={"Churned": "#e74c3c", "Retained": "#27ae60"},
            hole=0.45,
        )
        fig.update_layout(margin=dict(t=10, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        section_header("Churn by Contract Type")
        contract_data = churn_rate_by(df, "Contract")
        fig2 = px.bar(
            contract_data, x="Contract", y="ChurnRate",
            color="ChurnRate",
            color_continuous_scale="RdYlGn_r",
            text="ChurnRate",
            labels={"ChurnRate": "Churn Rate (%)"},
        )
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Charts Row 2
    col_c, col_d = st.columns(2)

    with col_c:
        section_header("Churn Risk Distribution")
        risk_counts = pred_df["RiskCategory"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        color_map_risk = {"Low Risk": "#27ae60", "Medium Risk": "#f39c12", "High Risk": "#e74c3c"}
        fig3 = px.bar(
            risk_counts, x="Risk", y="Count",
            color="Risk",
            color_discrete_map=color_map_risk,
            text="Count",
        )
        fig3.update_traces(textposition="outside")
        fig3.update_layout(margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        section_header("Monthly Charges vs Churn")
        sample = df.sample(min(1000, len(df)), random_state=42)
        fig4 = px.box(
            sample, x=TARGET_COL, y="MonthlyCharges",
            color=TARGET_COL,
            color_discrete_map={0: "#27ae60", 1: "#e74c3c"},
            labels={TARGET_COL: "Churn (0=No, 1=Yes)", "MonthlyCharges": "Monthly Charges ($)"},
        )
        fig4.update_layout(margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    # Executive Insights
    st.markdown("---")
    section_header("🔍 Executive Insights")
    tab1, tab2, tab3, tab4 = st.tabs(["Key Findings", "Risks", "Opportunities", "Recommended Actions"])

    with tab1:
        for i, finding in enumerate(insights["key_findings"], 1):
            st.markdown(f'<div class="insight-box">📌 <b>Finding {i}:</b> {finding}</div>', unsafe_allow_html=True)
    with tab2:
        for i, risk in enumerate(insights["risks"], 1):
            st.markdown(f'<div class="insight-box risk-box">⚠️ <b>Risk {i}:</b> {risk}</div>', unsafe_allow_html=True)
    with tab3:
        for i, oppty in enumerate(insights["opportunities"], 1):
            st.markdown(f'<div class="insight-box oppty-box">✅ <b>Opportunity {i}:</b> {oppty}</div>', unsafe_allow_html=True)
    with tab4:
        for i, action in enumerate(insights["actions"], 1):
            st.markdown(f'<div class="insight-box action-box">🎯 <b>Action {i}:</b> {action}</div>', unsafe_allow_html=True)


# ---- PAGE 2: Churn & Customer Analysis ----

def page_churn_analysis(df, pred_df):
    st.title("📊 Churn & Customer Analysis")
    st.markdown("*Use sidebar filters to drill down into specific customer segments.*")
    st.markdown("---")

    # Sidebar Filters
    st.sidebar.markdown("## 🔧 Filters")
    contracts      = ["All"] + sorted(df["Contract"].unique().tolist())
    internet_types = ["All"] + sorted(df["InternetService"].unique().tolist())
    pay_methods    = ["All"] + sorted(df["PaymentMethod"].unique().tolist())
    senior_opts    = ["All", "Senior Citizen", "Non-Senior"]
    gender_opts    = ["All"] + sorted(df["gender"].unique().tolist())
    risk_opts      = ["All", "High Risk", "Medium Risk", "Low Risk"]

    f_contract = st.sidebar.selectbox("Contract",          contracts)
    f_internet = st.sidebar.selectbox("Internet Service",  internet_types)
    f_payment  = st.sidebar.selectbox("Payment Method",    pay_methods)
    f_senior   = st.sidebar.selectbox("Senior Citizen",    senior_opts)
    f_gender   = st.sidebar.selectbox("Gender",            gender_opts)
    f_risk     = st.sidebar.selectbox("Risk Level",        risk_opts)

    # Apply filters
    filtered = pred_df.copy()
    if f_contract != "All": filtered = filtered[filtered["Contract"] == f_contract]
    if f_internet != "All": filtered = filtered[filtered["InternetService"] == f_internet]
    if f_payment  != "All": filtered = filtered[filtered["PaymentMethod"] == f_payment]
    if f_senior == "Senior Citizen":  filtered = filtered[filtered["SeniorCitizen"] == 1]
    elif f_senior == "Non-Senior":    filtered = filtered[filtered["SeniorCitizen"] == 0]
    if f_gender   != "All": filtered = filtered[filtered["gender"] == f_gender]
    if f_risk     != "All": filtered = filtered[filtered["RiskCategory"] == f_risk]

    st.markdown(f"**Filtered Dataset:** {len(filtered):,} customers")
    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        section_header("Churn by Tenure (months)")
        bins = [0, 12, 24, 36, 48, 60, 72]
        labels_bins = ["0–12", "13–24", "25–36", "37–48", "49–60", "61–72"]
        filtered["TenureBin"] = pd.cut(filtered["tenure"], bins=bins, labels=labels_bins, right=True)
        tenure_churn = churn_rate_by(filtered, "TenureBin")
        fig = px.bar(
            tenure_churn, x="TenureBin", y="ChurnRate",
            color="ChurnRate", color_continuous_scale="RdYlGn_r",
            text="ChurnRate",
            labels={"ChurnRate": "Churn Rate (%)", "TenureBin": "Tenure (months)"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("Churn by Payment Method")
        pay_churn = churn_rate_by(filtered, "PaymentMethod")
        fig2 = px.bar(
            pay_churn, x="PaymentMethod", y="ChurnRate",
            color="ChurnRate", color_continuous_scale="RdYlGn_r",
            text="ChurnRate",
            labels={"ChurnRate": "Churn Rate (%)"},
        )
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(
            margin=dict(t=10, b=10), coloraxis_showscale=False,
            xaxis=dict(tickangle=-20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        section_header("Churn by Internet Service")
        inet_churn = churn_rate_by(filtered, "InternetService")
        fig3 = px.bar(
            inet_churn, x="InternetService", y="ChurnRate",
            color="ChurnRate", color_continuous_scale="RdYlGn_r",
            text="ChurnRate",
            labels={"ChurnRate": "Churn Rate (%)"},
        )
        fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig3.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        section_header("Churn by Online Security")
        sec_churn = churn_rate_by(filtered, "OnlineSecurity")
        fig4 = px.bar(
            sec_churn, x="OnlineSecurity", y="ChurnRate",
            color="ChurnRate", color_continuous_scale="RdYlGn_r",
            text="ChurnRate",
        )
        fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig4.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    col5, col6 = st.columns(2)

    with col5:
        section_header("Monthly Charges Distribution by Churn")
        fig5 = px.histogram(
            filtered, x="MonthlyCharges", color=TARGET_COL,
            nbins=40,
            color_discrete_map={0: "#27ae60", 1: "#e74c3c"},
            barmode="overlay", opacity=0.75,
            labels={TARGET_COL: "Churn (0=No, 1=Yes)"},
        )
        fig5.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        section_header("Tech Support & Churn")
        tech_churn = churn_rate_by(filtered, "TechSupport")
        fig6 = px.bar(
            tech_churn, x="TechSupport", y="ChurnRate",
            color="ChurnRate", color_continuous_scale="RdYlGn_r",
            text="ChurnRate",
        )
        fig6.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig6.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig6, use_container_width=True)

    # Additional analysis
    section_header("Churn by Gender & Senior Citizen Status")
    col7, col8 = st.columns(2)

    with col7:
        gender_churn = churn_rate_by(filtered, "gender")
        fig7 = px.bar(
            gender_churn, x="gender", y="ChurnRate",
            color="gender",
            color_discrete_sequence=["#3b6fd4", "#e74c3c"],
            text="ChurnRate",
        )
        fig7.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig7.update_layout(margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig7, use_container_width=True)

    with col8:
        filtered["SeniorLabel"] = filtered["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior Citizen"})
        senior_churn = churn_rate_by(filtered, "SeniorLabel")
        fig8 = px.bar(
            senior_churn, x="SeniorLabel", y="ChurnRate",
            color="SeniorLabel",
            color_discrete_sequence=["#27ae60", "#e74c3c"],
            text="ChurnRate",
        )
        fig8.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig8.update_layout(margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig8, use_container_width=True)


# ---- PAGE 3: Risk, Opportunity & Action ----

def page_risk_action(df, pred_df, fi_df, kpis, insights):
    st.title("🎯 Risk, Opportunity & Action")
    st.markdown("---")

    # ---- RISK SECTION ----
    section_header("⚠️ Churn Risk Analysis")

    high_risk = pred_df[pred_df["RiskCategory"] == "High Risk"]
    med_risk  = pred_df[pred_df["RiskCategory"] == "Medium Risk"]
    low_risk  = pred_df[pred_df["RiskCategory"] == "Low Risk"]
    revenue_at_risk = high_risk["MonthlyCharges"].sum()

    r1, r2, r3, r4 = st.columns(4)
    with r1: kpi_card("High-Risk Customers",  f"{len(high_risk):,}", "red")
    with r2: kpi_card("Medium-Risk Customers",f"{len(med_risk):,}",  "orange")
    with r3: kpi_card("Low-Risk Customers",   f"{len(low_risk):,}",  "green")
    with r4: kpi_card("Revenue at Risk ($/mo)",f"${revenue_at_risk:,.0f}", "red")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        section_header("Risk Distribution by Contract")
        risk_contract = pred_df.groupby(["Contract", "RiskCategory"]).size().reset_index(name="Count")
        fig_r1 = px.bar(
            risk_contract, x="Contract", y="Count", color="RiskCategory",
            color_discrete_map={"High Risk": "#e74c3c", "Medium Risk": "#f39c12", "Low Risk": "#27ae60"},
            barmode="stack",
        )
        fig_r1.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig_r1, use_container_width=True)

    with col_r2:
        section_header("Top 10 Churn Drivers (Feature Importance)")
        top10 = fi_df.head(10).sort_values("Importance")
        fig_r2 = px.bar(
            top10, x="Importance", y="OriginalFeature", orientation="h",
            color="Importance", color_continuous_scale="Blues",
            labels={"OriginalFeature": "Feature", "Importance": "Importance Score"},
        )
        fig_r2.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_r2, use_container_width=True)

    # Risk insights
    for risk_text in insights["risks"]:
        st.markdown(f'<div class="insight-box risk-box">⚠️ {risk_text}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ---- OPPORTUNITY SECTION ----
    section_header("✅ Retention Opportunities")

    col_o1, col_o2 = st.columns(2)

    with col_o1:
        section_header("Churn Rate by Paperless Billing")
        pb_churn = churn_rate_by(df, "PaperlessBilling")
        fig_o1 = px.bar(
            pb_churn, x="PaperlessBilling", y="ChurnRate",
            color="PaperlessBilling",
            color_discrete_sequence=["#27ae60", "#e74c3c"],
            text="ChurnRate",
        )
        fig_o1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_o1.update_layout(margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig_o1, use_container_width=True)

    with col_o2:
        section_header("Churn by Tenure Groups")
        bins = [0, 12, 24, 48, 72]
        labels_b = ["0–12 mo", "13–24 mo", "25–48 mo", "49–72 mo"]
        df_copy = df.copy()
        df_copy["TenureGroup"] = pd.cut(df_copy["tenure"], bins=bins, labels=labels_b, right=True)
        tg_churn = churn_rate_by(df_copy, "TenureGroup")
        fig_o2 = px.line(
            tg_churn, x="TenureGroup", y="ChurnRate",
            markers=True, text="ChurnRate",
            labels={"ChurnRate": "Churn Rate (%)", "TenureGroup": "Tenure Group"},
            color_discrete_sequence=["#3b6fd4"],
        )
        fig_o2.update_traces(texttemplate="%{text:.1f}%", textposition="top center")
        fig_o2.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig_o2, use_container_width=True)

    for oppty in insights["opportunities"]:
        st.markdown(f'<div class="insight-box oppty-box">✅ {oppty}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ---- ACTION SECTION ----
    section_header("🎯 Recommended Business Actions")
    for i, action in enumerate(insights["actions"], 1):
        st.markdown(f'<div class="insight-box action-box"><b>Action {i}:</b> {action}</div>', unsafe_allow_html=True)

    # High-Risk Customer Table
    st.markdown("---")
    section_header("High-Risk Customer Snapshot (Top 20 by Churn Probability)")
    display_cols = ["tenure", "Contract", "MonthlyCharges", "InternetService",
                    "PaymentMethod", "TechSupport", "ChurnProbability", "RiskCategory"]
    available_cols = [c for c in display_cols if c in high_risk.columns]
    top20 = high_risk.nlargest(20, "ChurnProbability")[available_cols].reset_index(drop=True)
    st.dataframe(top20.style.format({"ChurnProbability": "{:.1%}", "MonthlyCharges": "${:.2f}"}),
                 use_container_width=True)


# ---- PAGE 4: Individual Customer Prediction ----

def page_prediction(pipeline, fi_df):
    st.title("🔮 Individual Customer Churn Prediction")
    st.markdown("*Enter customer details to predict churn probability and get personalised retention advice.*")
    st.markdown("---")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            tenure          = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
            total_charges   = st.number_input("Total Charges ($)",   0.0, 10000.0,
                                              float(tenure * monthly_charges), step=1.0)
            senior          = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")

        with col2:
            contract        = st.selectbox("Contract",        ["Month-to-month", "One year", "Two year"])
            internet        = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            payment         = st.selectbox("Payment Method",  ["Electronic check", "Mailed check",
                                                                "Bank transfer (automatic)",
                                                                "Credit card (automatic)"])
            paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])

        with col3:
            tech_support    = st.selectbox("Tech Support",    ["Yes", "No", "No internet service"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup   = st.selectbox("Online Backup",   ["Yes", "No", "No internet service"])
            device_prot     = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            partner         = st.selectbox("Partner",         ["Yes", "No"])
            dependents      = st.selectbox("Dependents",      ["Yes", "No"])
            gender          = st.selectbox("Gender",          ["Male", "Female"])
            phone_service   = st.selectbox("Phone Service",   ["Yes", "No"])
            multiple_lines  = st.selectbox("Multiple Lines",  ["Yes", "No", "No phone service"])
            streaming_tv    = st.selectbox("Streaming TV",    ["Yes", "No", "No internet service"])
            streaming_movies= st.selectbox("Streaming Movies",["Yes", "No", "No internet service"])

        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            "gender"          : gender,
            "SeniorCitizen"   : senior,
            "Partner"         : partner,
            "Dependents"      : dependents,
            "tenure"          : tenure,
            "PhoneService"    : phone_service,
            "MultipleLines"   : multiple_lines,
            "InternetService" : internet,
            "OnlineSecurity"  : online_security,
            "OnlineBackup"    : online_backup,
            "DeviceProtection": device_prot,
            "TechSupport"     : tech_support,
            "StreamingTV"     : streaming_tv,
            "StreamingMovies" : streaming_movies,
            "Contract"        : contract,
            "PaperlessBilling": paperless,
            "PaymentMethod"   : payment,
            "MonthlyCharges"  : monthly_charges,
            "TotalCharges"    : total_charges,
        }])

        proba = pipeline.predict_proba(input_data)[0][1]

        if proba < LOW_RISK_THRESHOLD:
            risk_label = "🟢 Low Risk"
            risk_color = "#27ae60"
            action = "This customer shows low churn risk. Continue providing quality service and consider loyalty rewards."
        elif proba < MEDIUM_RISK_THRESHOLD:
            risk_label = "🟡 Medium Risk"
            risk_color = "#f39c12"
            action = "Monitor this customer closely. Consider a proactive outreach call and offer a service upgrade."
        else:
            risk_label = "🔴 High Risk"
            risk_color = "#e74c3c"
            action = "Immediate retention action required. Offer a personalised retention package, contract upgrade incentive, or discounted bundle."

        st.markdown("---")
        st.markdown("### Prediction Results")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{risk_color}">
                <div class="metric-value">{proba:.1%}</div>
                <div class="metric-label">Churn Probability</div>
            </div>""", unsafe_allow_html=True)
        with p2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{risk_color}">
                <div class="metric-value">{risk_label}</div>
                <div class="metric-label">Risk Category</div>
            </div>""", unsafe_allow_html=True)
        with p3:
            churn_pred = "Will Churn" if proba >= 0.5 else "Will Not Churn"
            pred_color = "#e74c3c" if proba >= 0.5 else "#27ae60"
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{pred_color}">
                <div class="metric-value">{churn_pred}</div>
                <div class="metric-label">Model Prediction (50% threshold)</div>
            </div>""", unsafe_allow_html=True)

        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={"text": "Churn Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 30],  "color": "#d5f5e3"},
                    {"range": [30, 60], "color": "#fef9e7"},
                    {"range": [60, 100],"color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": proba * 100,
                },
            },
            number={"suffix": "%", "font": {"size": 30}},
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Key factors
        st.markdown("### Key Factors Influencing This Prediction")
        top_factors = fi_df.head(5)
        for _, row in top_factors.iterrows():
            st.markdown(f"- **{row['OriginalFeature']}** — Importance Score: `{row['Importance']:.4f}`")

        # Recommended action
        st.markdown("### Recommended Retention Action")
        st.markdown(f'<div class="insight-box action-box">🎯 {action}</div>', unsafe_allow_html=True)

        # Risk threshold note
        st.info(
            f"*Note: Risk thresholds — Low Risk: < {LOW_RISK_THRESHOLD:.0%}  |  "
            f"Medium Risk: {LOW_RISK_THRESHOLD:.0%}–{MEDIUM_RISK_THRESHOLD:.0%}  |  "
            f"High Risk: > {MEDIUM_RISK_THRESHOLD:.0%}.*  "
            "Model prediction uses Churn Probability > 50% as decision boundary."
        )


# ---- PAGE 5: Model Performance ----

def page_model_performance(results, X_test, y_test, best_name):
    st.title("🤖 Machine Learning Model Performance")
    st.markdown("---")

    section_header("Model Comparison")
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        for m in metric_names:
            row[m] = metrics[m]
        rows.append(row)
    comparison_df = pd.DataFrame(rows)
    st.dataframe(
        comparison_df.style.highlight_max(subset=metric_names, color="#d5f5e3").format(
            {m: "{:.4f}" for m in metric_names}
        ),
        use_container_width=True,
    )

    best_metrics = results[best_name]
    st.success(
        f"✅ **Best Model: {best_name}** selected based on highest ROC-AUC "
        f"({best_metrics['ROC-AUC']:.4f}). "
        "ROC-AUC is prioritised over accuracy for churn prediction because the business "
        "cost of missing a churner (False Negative) is higher than a false alarm (False Positive)."
    )

    col1, col2 = st.columns(2)

    with col1:
        section_header("ROC Curves")
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dot", color="grey"))
        colors = ["#3b6fd4", "#e74c3c"]
        for (model_name, metrics), color in zip(results.items(), colors):
            fpr, tpr, _ = roc_curve(y_test, metrics["y_proba"])
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{model_name} (AUC={metrics['ROC-AUC']:.3f})",
                line=dict(color=color, width=2),
            ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            margin=dict(t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=0.02),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        section_header(f"Confusion Matrix — {best_name}")
        cm = confusion_matrix(y_test, best_metrics["y_pred"])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            x=["Predicted No Churn", "Predicted Churn"],
            y=["Actual No Churn",    "Actual Churn"],
        )
        fig_cm.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    section_header(f"Classification Report — {best_name}")
    report = classification_report(y_test, best_metrics["y_pred"],
                                   target_names=["No Churn", "Churn"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(
        report_df.style.format("{:.3f}").highlight_max(axis=0, color="#d5f5e3"),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# MAIN APPLICATION ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    setup_page()

    # ---- Sidebar Navigation ----
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
        width=80
    )
    st.sidebar.title("📡 Churn Intelligence")
    st.sidebar.markdown("*IBM SkillsBuild Internship*")
    st.sidebar.markdown("---")

    pages = {
        "🏠 Executive Overview"        : "overview",
        "📊 Churn & Customer Analysis" : "analysis",
        "🎯 Risk, Opportunity & Action": "risk",
        "🔮 Customer Prediction"       : "prediction",
        "🤖 Model Performance"         : "model",
    }
    selected = st.sidebar.radio("Navigate", list(pages.keys()))
    page_key = pages[selected]

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Dataset Path**")
    data_path = st.sidebar.text_input("CSV Path", value=DATASET_PATH)

    # ---- Load & Process Data ----
    try:
        raw_df = load_data(data_path)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    df = clean_data(raw_df)

    # ---- Train or load model ----
    pipeline = load_pipeline(MODEL_PATH)

    if pipeline is None:
        with st.spinner("Training models — this takes ~30 seconds on first run…"):
            X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols = preprocess_data(df)
            results, pipelines, best_name = train_models(X_train, X_test, y_train, y_test, preprocessor)
            pipeline = pipelines[best_name]
            save_pipeline({
                "pipeline"  : pipeline,
                "results"   : results,
                "best_name" : best_name,
                "cat_cols"  : cat_cols,
                "num_cols"  : num_cols,
                "X_test"    : X_test,
                "y_test"    : y_test,
            }, MODEL_PATH)
        st.success("✅ Model trained and saved. Subsequent loads will be instant.")
    else:
        saved = pipeline
        pipeline   = saved["pipeline"]
        results    = saved["results"]
        best_name  = saved["best_name"]
        cat_cols   = saved["cat_cols"]
        num_cols   = saved["num_cols"]
        X_test     = saved["X_test"]
        y_test     = saved["y_test"]

    # ---- Compute predictions, KPIs, insights ----
    pred_df  = generate_predictions(df, pipeline)
    kpis     = calculate_kpis(df)
    fi_df    = get_feature_importance(pipeline, cat_cols, num_cols)
    insights = generate_business_insights(df, kpis, fi_df, pred_df)

    # ---- Render selected page ----
    if   page_key == "overview"   : page_overview(df, kpis, pred_df, insights)
    elif page_key == "analysis"   : page_churn_analysis(df, pred_df)
    elif page_key == "risk"       : page_risk_action(df, pred_df, fi_df, kpis, insights)
    elif page_key == "prediction" : page_prediction(pipeline, fi_df)
    elif page_key == "model"      : page_model_performance(results, X_test, y_test, best_name)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small>© 2024 Shivam Singh | IBM SkillsBuild<br>"
        "Telecom Churn Intelligence v1.0</small>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
