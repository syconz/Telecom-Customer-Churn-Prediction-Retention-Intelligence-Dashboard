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

from dashboard_ui import (
    COLORS, CHURN_COLORS, RISK_COLORS, RISK_ORDER, setup_page, page_header,
    kpi_card, section_header, insight_card, chart, navigate_to,
    sidebar_brand, sidebar_status,
)

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
def clean_data(df: pd.DataFrame) -> tuple:
    """
    Cleaning steps with full audit trail:
      1. Strip whitespace from column names and string values.
      2. Convert TotalCharges to numeric (blank strings → NaN).
      3. Drop rows where TotalCharges is NaN after conversion.
         These are customers with tenure=0 who have never been billed —
         their TotalCharges field contains a single space ' ' in the CSV,
         which pandas cannot convert to float.
      4. Encode target variable: Churn Yes→1, No→0.
      5. Drop customerID (unique identifier, not a predictor).
      6. Remove exact duplicate rows (all 19 columns identical).
      7. Ensure SeniorCitizen is int.

    Returns: (cleaned_df, cleaning_log dict)
    The cleaning_log records every row-removal step so the UI can
    display an explicit data provenance table.
    """
    df = df.copy()
    log = {}
    log["raw_rows"] = len(df)

    # Step 1 – strip whitespace
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Step 2 – TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Step 3 – drop NaN TotalCharges
    # Customers with tenure=0 have a blank TotalCharges field in the
    # source CSV.  pd.to_numeric converts that blank to NaN, which we
    # then drop.  These 11 rows represent brand-new accounts that have
    # never completed a billing cycle, so they carry no churn signal.
    before_tc = len(df)
    df.dropna(subset=["TotalCharges"], inplace=True)
    log["dropped_blank_totalcharges"] = before_tc - len(df)
    log["after_totalcharges_drop"] = len(df)

    # Step 4 – encode target
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    # Step 5 – drop non-feature columns
    for col in DROP_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Step 6 – remove duplicates (all feature columns identical)
    before_dup = len(df)
    df.drop_duplicates(inplace=True)
    log["dropped_duplicates"] = before_dup - len(df)
    log["final_rows"] = len(df)

    # Step 7 – SeniorCitizen as int
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    return df, log

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
    Every numerical statement is computed from the cleaned dataset at
    runtime.  No values are hard-coded.  Where a claim involves the ML
    model (risk counts, revenue at risk) the text explicitly says so.
    """
    # --- Contract analysis ---
    contract_churn = churn_rate_by(df, "Contract")
    highest_churn_contract      = contract_churn.loc[contract_churn["ChurnRate"].idxmax(), "Contract"]
    highest_churn_contract_rate = contract_churn["ChurnRate"].max()
    # Lowest-churn contract for comparison
    lowest_churn_contract       = contract_churn.loc[contract_churn["ChurnRate"].idxmin(), "Contract"]
    lowest_churn_contract_rate  = contract_churn["ChurnRate"].min()

    # --- Tenure analysis ---
    df_lt12  = df[df["tenure"] <= 12]
    df_gt24  = df[df["tenure"] >  24]
    short_tenure_churn = df_lt12[TARGET_COL].mean() * 100
    long_tenure_churn  = df_gt24[TARGET_COL].mean() * 100
    short_tenure_n     = len(df_lt12)

    # --- Internet service analysis ---
    internet_churn = churn_rate_by(df, "InternetService")
    highest_internet_service = internet_churn.loc[internet_churn["ChurnRate"].idxmax(), "InternetService"]
    highest_internet_rate    = internet_churn["ChurnRate"].max()
    lowest_internet_service  = internet_churn.loc[internet_churn["ChurnRate"].idxmin(), "InternetService"]
    lowest_internet_rate     = internet_churn["ChurnRate"].min()

    # --- Senior citizen analysis ---
    senior_churn     = df[df["SeniorCitizen"] == 1][TARGET_COL].mean() * 100
    non_senior_churn = df[df["SeniorCitizen"] == 0][TARGET_COL].mean() * 100

    # --- Tech support / online security (actual churn rates, no vague words) ---
    ts_churn    = churn_rate_by(df, "TechSupport")
    ts_no_rate  = float(ts_churn.loc[ts_churn["TechSupport"] == "No",  "ChurnRate"].values[0]) \
                  if "No" in ts_churn["TechSupport"].values else float("nan")
    ts_yes_rate = float(ts_churn.loc[ts_churn["TechSupport"] == "Yes", "ChurnRate"].values[0]) \
                  if "Yes" in ts_churn["TechSupport"].values else float("nan")

    sec_churn    = churn_rate_by(df, "OnlineSecurity")
    sec_no_rate  = float(sec_churn.loc[sec_churn["OnlineSecurity"] == "No",  "ChurnRate"].values[0]) \
                   if "No" in sec_churn["OnlineSecurity"].values else float("nan")
    sec_yes_rate = float(sec_churn.loc[sec_churn["OnlineSecurity"] == "Yes", "ChurnRate"].values[0]) \
                   if "Yes" in sec_churn["OnlineSecurity"].values else float("nan")

    # --- Top churn driver (state model and method explicitly) ---
    top_feature = fi_df.iloc[0]["OriginalFeature"] if not fi_df.empty else "tenure"

    # --- Model-predicted risk counts and revenue ---
    # NOTE: these figures are outputs of the ML model, not observed churn counts.
    # "High Risk" = model-predicted churn probability > MEDIUM_RISK_THRESHOLD.
    high_risk       = pred_df[pred_df["RiskCategory"] == "High Risk"]
    revenue_at_risk = high_risk["MonthlyCharges"].sum()
    high_risk_count = len(high_risk)

    # --- Payment method churn ---
    pay_churn          = churn_rate_by(df, "PaymentMethod")
    highest_pay_method = pay_churn.loc[pay_churn["ChurnRate"].idxmax(), "PaymentMethod"]
    highest_pay_rate   = pay_churn["ChurnRate"].max()
    lowest_pay_method  = pay_churn.loc[pay_churn["ChurnRate"].idxmin(), "PaymentMethod"]
    lowest_pay_rate    = pay_churn["ChurnRate"].min()

    # --- Long-tenure retained customers ---
    loyal       = df[(df["tenure"] > 24) & (df[TARGET_COL] == 0)]
    loyal_count = len(loyal)

    insights = {
        "key_findings": [
            (f"Customers on {highest_churn_contract} contracts have the highest observed churn rate "
             f"at {highest_churn_contract_rate:.1f}% vs {lowest_churn_contract_rate:.1f}% for "
             f"{lowest_churn_contract} contracts (calculated from {len(df):,} cleaned records)."),
            (f"Of {short_tenure_n:,} customers with tenure ≤ 12 months, {short_tenure_churn:.1f}% "
             f"have churned, compared to {long_tenure_churn:.1f}% for customers with tenure > 24 months."),
            (f"{highest_internet_service} internet service customers show the highest observed churn "
             f"rate at {highest_internet_rate:.1f}%, compared to {lowest_internet_rate:.1f}% for "
             f"{lowest_internet_service} service."),
            (f"Senior citizens (SeniorCitizen=1) show a churn rate of {senior_churn:.1f}% vs "
             f"{non_senior_churn:.1f}% for non-seniors — a difference of "
             f"{abs(senior_churn - non_senior_churn):.1f} percentage points."),
            (f"'{top_feature}' has the highest aggregated feature importance score in the trained "
             f"model (scores reflect Mean Decrease in Impurity for Random Forest, or |coefficient| "
             f"for Logistic Regression — these indicate association, not causation)."),
        ],
        "risks": [
            (f"{high_risk_count:,} customers ({high_risk_count/len(df)*100:.1f}% of the dataset) "
             f"are classified as High Risk by the ML model (predicted churn probability "
             f"> {MEDIUM_RISK_THRESHOLD:.0%})."),
            (f"The combined monthly charges for all model-predicted High-Risk customers total "
             f"${revenue_at_risk:,.0f}/month — this is the revenue potentially at risk if these "
             f"customers were to churn (based on model predictions, not confirmed outcomes)."),
            (f"Customers paying via '{highest_pay_method}' have the highest observed churn rate "
             f"at {highest_pay_rate:.1f}%, compared to {lowest_pay_rate:.1f}% for "
             f"'{lowest_pay_method}' users."),
        ],
        "opportunities": [
            (f"{loyal_count:,} customers have tenure > 24 months and have not churned — "
             f"this segment ({loyal_count/len(df)*100:.1f}% of all customers) represents a "
             f"stable base for retention and upsell campaigns."),
            (f"Customers without TechSupport churn at {ts_no_rate:.1f}% vs {ts_yes_rate:.1f}% "
             f"with TechSupport. Customers without OnlineSecurity churn at {sec_no_rate:.1f}% "
             f"vs {sec_yes_rate:.1f}% with it. Adding these services is associated with lower "
             f"churn in the observed data (correlation, not confirmed causation)."),
            (f"Observed churn rate for {highest_churn_contract} contracts is "
             f"{highest_churn_contract_rate:.1f}% vs {lowest_churn_contract_rate:.1f}% for "
             f"{lowest_churn_contract} contracts. Customers successfully migrated to longer "
             f"contracts are associated with lower churn in this dataset."),
        ],
        # Pre-compute differences before rounding so printed values are
        # consistent — avoids e.g. "42.7% − 2.9% = 39.8%" when the true
        # difference rounds to 39.7%.
        "actions": [
            (f"Use the model's High-Risk flag (probability > {MEDIUM_RISK_THRESHOLD:.0%}) as a "
             f"prioritisation tool to identify the {high_risk_count:,} customers most worth "
             f"investigating for retention outreach — beginning with those on "
             f"{highest_churn_contract} contracts, which show the highest observed churn rate "
             f"({highest_churn_contract_rate:.1f}%) in this dataset. Validate through a "
             f"controlled experiment before rolling out at scale."),
            (f"Evaluate targeted TechSupport and OnlineSecurity offers for customers who currently "
             f"lack these services, and measure the retention impact through a controlled "
             f"experiment. The observed churn rates in this dataset are "
             f"{ts_no_rate:.1f}% (no TechSupport) vs {ts_yes_rate:.1f}% (with TechSupport) and "
             f"{sec_no_rate:.1f}% (no OnlineSecurity) vs {sec_yes_rate:.1f}% (with it) — "
             f"these are associations, not proven causal effects."),
            (f"Test contract-upgrade incentives among {highest_churn_contract} customers and "
             f"measure whether the intervention reduces churn. The observed churn gap between "
             f"{highest_churn_contract} ({highest_churn_contract_rate:.1f}%) and "
             f"{lowest_churn_contract} ({lowest_churn_contract_rate:.1f}%) contracts is "
             f"{round(highest_churn_contract_rate - lowest_churn_contract_rate, 1):.1f} "
             f"percentage points in this dataset."),
            (f"Investigate whether the higher churn rate among '{highest_pay_method}' users "
             f"({highest_pay_rate:.1f}%) reflects a billing experience problem or a customer "
             f"segment effect. The gap vs '{lowest_pay_method}' users ({lowest_pay_rate:.1f}%) "
             f"is {round(highest_pay_rate - lowest_pay_rate, 1):.1f} percentage points. "
             f"Design a targeted experiment before concluding a payment-method intervention "
             f"would reduce churn."),
        ],
    }
    return insights

# ---------------------------------------------------------------------------
# STREAMLIT DASHBOARD
# ---------------------------------------------------------------------------

def churn_rate_chart(df, column, title, horizontal=False):
    """A consistent chart card for observed churn across customer segments."""
    with st.container(border=True, key=f"panel-rate-{column}"):
        section_header(title, "Observed churn · share of customers in each segment")
        data = churn_rate_by(df, column).dropna(subset=["ChurnRate"])
        data["Segment"] = data[column].astype(str)
        if horizontal:
            data = data.sort_values("ChurnRate")
        fig = px.bar(
            data,
            x="ChurnRate" if horizontal else "Segment",
            y="Segment" if horizontal else "ChurnRate",
            orientation="h" if horizontal else "v",
            text="ChurnRate", custom_data=["Total"],
            labels={"Segment": "", "ChurnRate": "Churn rate (%)"},
        )
        fig.update_traces(
            marker_color=[COLORS["red"] if rate == data["ChurnRate"].max() else COLORS["teal"]
                          for rate in data["ChurnRate"]],
            marker_line_width=0, texttemplate="%{text:.1f}%", textposition="outside",
            cliponaxis=False,
            hovertemplate=("%{y}<br>Churn rate: %{x:.1f}%" if horizontal else
                           "%{x}<br>Churn rate: %{y:.1f}%")
                          + "<br>Customers: %{customdata[0]:,}<extra></extra>",
        )
        maximum = max(10, data["ChurnRate"].max() * 1.2) if not data.empty else 100
        if horizontal:
            fig.update_xaxes(range=[0, maximum])
        else:
            fig.update_yaxes(range=[0, maximum])
        fig.update_layout(showlegend=False, bargap=0.45)
        chart(fig, key=f"churn-rate-{column}")


# ---- PAGE 1: Executive Overview ----

def page_overview(df, kpis, pred_df, insights, cleaning_log):
    page_header("Workspace / Overview", "Your retention overview",
                "Understand customer churn, spot emerging risk, and focus your next retention move.")
    high_risk = pred_df[pred_df["RiskCategory"] == "High Risk"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Customers analysed", f"{len(df):,}", "blue", note="Cleaned customer records")
    with c2:
        kpi_card("Observed churn rate", f"{kpis['Churn Rate (%)']:.1f}", "red", "%",
                 note=f"{kpis['Churned Customers']:,} churned · {kpis['Retention Rate (%)']:.1f}% retained")
    with c3:
        kpi_card("Monthly charges base", f"${kpis['Total Monthly Revenue']/1000:,.1f}k", "teal",
                 note=f"${kpis['Avg Monthly Charges']:.2f} average per customer")
    with c4:
        kpi_card("High-risk customers", f"{len(high_risk):,}", "red",
                 note=f"{len(high_risk)/len(df):.1%} of portfolio · model-predicted")

    with st.container(key="retention-banner"):
        message, action = st.columns([3, 1], vertical_alignment="center")
        with message:
            st.markdown(
                '<div class="banner-eyebrow">YOUR NEXT RETENTION MOVE</div>'
                f'<div class="banner-title">{len(high_risk):,} customers worth a closer look.</div>'
                f'<p class="banner-note">${high_risk["MonthlyCharges"].sum():,.0f} in monthly charges '
                'across the high-risk segment. Explore the signals and prioritise outreach.</p>',
                unsafe_allow_html=True,
            )
        with action:
            st.button("Explore retention priorities →", width="stretch",
                      on_click=navigate_to, args=("Retention priorities",))

    col_a, col_b = st.columns(2)
    with col_a, st.container(border=True, key="panel-retention"):
        section_header("Customer retention", "Observed outcomes across the analytical dataset")
        counts = df[TARGET_COL].map({0: "Retained", 1: "Churned"}).value_counts().rename_axis("Status").reset_index(name="Customers")
        fig = px.pie(counts, values="Customers", names="Status", color="Status",
                     color_discrete_map=CHURN_COLORS, hole=0.76)
        fig.update_traces(textinfo="none", marker_line_color="white", marker_line_width=4,
                          hovertemplate="%{label}<br>%{value:,} customers · %{percent}<extra></extra>")
        fig.add_annotation(x=0.5, y=0.5, showarrow=False,
                           text=f"<b>{kpis['Retention Rate (%)']:.1f}%</b><br><span style='font-size:12px'>retained</span>",
                           font=dict(size=30, color=COLORS["ink"]))
        chart(fig)
    with col_b:
        churn_rate_chart(df, "Contract", "Churn by contract type")

    col_c, col_d = st.columns(2)
    with col_c, st.container(border=True, key="panel-risk-landscape"):
        section_header("The risk landscape", f"Model-predicted segments · {LOW_RISK_THRESHOLD:.0%} and {MEDIUM_RISK_THRESHOLD:.0%} probability cutoffs")
        counts = pred_df["RiskCategory"].value_counts().rename_axis("Risk").reset_index(name="Customers")
        fig = px.bar(counts, x="Risk", y="Customers", color="Risk", text="Customers",
                     color_discrete_map=RISK_COLORS, category_orders={"Risk": RISK_ORDER},
                     labels={"Risk": ""})
        fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, bargap=0.5)
        fig.update_yaxes(range=[0, counts["Customers"].max() * 1.2])
        chart(fig)
    with col_d, st.container(border=True, key="panel-monthly-charges"):
        section_header("Monthly charges & churn", "Billing distribution · sample of up to 1,000 customers")
        sample = df.sample(min(1000, len(df)), random_state=42).copy()
        sample["Status"] = sample[TARGET_COL].map({0: "Retained", 1: "Churned"})
        fig = px.box(sample, x="Status", y="MonthlyCharges", color="Status",
                     color_discrete_map=CHURN_COLORS, labels={"Status": "", "MonthlyCharges": "Monthly charges ($)"})
        fig.update_layout(showlegend=False)
        chart(fig)

    with st.container(border=True, key="panel-insights"):
        section_header("From insight to action", "Data-derived observations and retention hypotheses")
        tabs = st.tabs(["Key findings", "Risk signals", "Opportunities", "Recommended actions"])
        for tab, key, tone in zip(tabs, ["key_findings", "risks", "opportunities", "actions"],
                                  ["teal", "red", "green", "orange"]):
            with tab:
                for i, text in enumerate(insights[key], 1):
                    insight_card(text, i, tone)

    with st.expander("About this dataset · cleaning audit & KPI definitions"):
        st.markdown(f"""
| Preparation step | Records removed | Records remaining |
|:--|--:|--:|
| Raw CSV | — | {cleaning_log['raw_rows']:,} |
| Remove blank TotalCharges | {cleaning_log['dropped_blank_totalcharges']:,} | {cleaning_log['after_totalcharges_drop']:,} |
| Remove duplicates after excluding customerID | {cleaning_log['dropped_duplicates']:,} | {cleaning_log['final_rows']:,} |
""")
        st.caption(f"Average tenure: {kpis['Avg Tenure (months)']:.1f} months. Monthly charges base sums all analytical records, "
                   "including churned customers. Revenue at risk sums monthly charges for model-predicted high-risk customers; "
                   "it is potential exposure, not confirmed lost revenue. Feature importance describes association, not causation.")


# ---- PAGE 2: Churn & Customer Analysis ----

ANALYSIS_FILTERS = ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen", "gender", "RiskCategory"]


def reset_analysis_filters():
    for column in ANALYSIS_FILTERS:
        st.session_state[f"filter_{column}"] = "All"


def page_churn_analysis(df, pred_df):
    page_header("Workspace / Customer analysis", "Find the patterns behind churn",
                "Compare customer segments to understand where retention needs the most attention.")
    labels = ["Contract", "Internet service", "Payment method", "Senior citizen", "Gender", "Risk level"]
    selected = {}
    with st.expander("Filter customer segments", expanded=True):
        columns = st.columns(3)
        for i, (column, label) in enumerate(zip(ANALYSIS_FILTERS, labels)):
            options = (["Non-Senior", "Senior Citizen"] if column == "SeniorCitizen" else
                       RISK_ORDER if column == "RiskCategory" else sorted(df[column].unique().tolist()))
            with columns[i % 3]:
                selected[column] = st.selectbox(label, ["All"] + options, key=f"filter_{column}")
        active = sum(value != "All" for value in selected.values())
        summary, reset = st.columns([3, 1], vertical_alignment="center")
        with summary:
            st.caption(f"{active} active filter{'s' if active != 1 else ''} · charts update as you refine your segment")
        with reset:
            st.button("Reset filters", on_click=reset_analysis_filters, width="stretch")

    filtered = pred_df.copy()
    for column, value in selected.items():
        if value == "All":
            continue
        if column == "SeniorCitizen":
            value = 1 if value == "Senior Citizen" else 0
        filtered = filtered[filtered[column] == value]

    if filtered.empty:
        with st.container(border=True, key="panel-empty-segment"):
            section_header("No customers match these filters")
            st.write("Try a broader segment or reset your filters to explore all customers.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Customers in this segment", f"{len(filtered):,}", "blue", note=f"{len(filtered)/len(df):.1%} of the full dataset")
    with c2:
        kpi_card("Segment churn rate", f"{filtered[TARGET_COL].mean()*100:.1f}", "red", "%", note="Observed churn in the selected segment")
    with c3:
        kpi_card("High-risk customers", f"{(filtered['RiskCategory'] == 'High Risk').sum():,}", "red",
                 note=f"Model-predicted probability above {MEDIUM_RISK_THRESHOLD:.0%}")

    account, services, demographics = st.tabs(["Account & billing", "Services", "Demographics"])
    with account:
        left, right = st.columns(2)
        filtered["TenureBin"] = pd.cut(filtered["tenure"], [0, 12, 24, 36, 48, 60, 72],
                                       labels=["0–12", "13–24", "25–36", "37–48", "49–60", "61–72"], include_lowest=True)
        with left:
            churn_rate_chart(filtered, "TenureBin", "Churn by tenure (months)")
        with right:
            churn_rate_chart(filtered, "PaymentMethod", "Churn by payment method", horizontal=True)
        with st.container(border=True, key="panel-billing-distribution"):
            section_header("Monthly charges distribution", "Compare billing patterns among churned and retained customers")
            filtered["Status"] = filtered[TARGET_COL].map({0: "Retained", 1: "Churned"})
            fig = px.histogram(filtered, x="MonthlyCharges", color="Status", nbins=40,
                               color_discrete_map=CHURN_COLORS, barmode="overlay", opacity=0.75,
                               labels={"MonthlyCharges": "Monthly charges ($)", "count": "Customers"})
            chart(fig)
    with services:
        left, right = st.columns(2)
        with left:
            churn_rate_chart(filtered, "InternetService", "Internet service")
            churn_rate_chart(filtered, "TechSupport", "Technical support")
        with right:
            churn_rate_chart(filtered, "OnlineSecurity", "Online security")
    with demographics:
        left, right = st.columns(2)
        with left:
            churn_rate_chart(filtered, "gender", "Churn by gender")
        with right:
            filtered["SeniorLabel"] = filtered["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior Citizen"})
            churn_rate_chart(filtered, "SeniorLabel", "Senior citizen status")


# ---- PAGE 3: Risk, Opportunity & Action ----

def page_risk_action(df, pred_df, fi_df, kpis, insights):
    page_header("Workspace / Retention priorities", "Turn risk signals into action",
                "Explore your highest-risk segments and build a focused retention shortlist.")

    high_risk = pred_df[pred_df["RiskCategory"] == "High Risk"]
    med_risk  = pred_df[pred_df["RiskCategory"] == "Medium Risk"]
    low_risk  = pred_df[pred_df["RiskCategory"] == "Low Risk"]
    revenue_at_risk = high_risk["MonthlyCharges"].sum()

    r1, r2, r3, r4 = st.columns(4)
    with r1: kpi_card("High risk", f"{len(high_risk):,}", "red", note=f"Predicted probability > {MEDIUM_RISK_THRESHOLD:.0%}")
    with r2: kpi_card("Medium risk", f"{len(med_risk):,}", "orange", note=f"Predicted probability > {LOW_RISK_THRESHOLD:.0%} to {MEDIUM_RISK_THRESHOLD:.0%}")
    with r3: kpi_card("Low risk", f"{len(low_risk):,}", "green", note=f"Predicted probability ≤ {LOW_RISK_THRESHOLD:.0%}")
    with r4: kpi_card("Monthly exposure", f"${revenue_at_risk/1000:,.1f}k", "red", note="Monthly charges in the high-risk segment")
    st.caption(
        "Risk segments are model predictions. Monthly exposure represents potential revenue at risk, not confirmed lost revenue."
    )

    col_r1, col_r2 = st.columns(2)

    with col_r1, st.container(border=True, key="panel-risk-contract"):
        section_header("Where risk is concentrated", "Customer risk by contract type")
        risk_contract = pred_df.groupby(["Contract", "RiskCategory"], observed=False).size().reset_index(name="Count")
        fig_r1 = px.bar(
            risk_contract, x="Contract", y="Count", color="RiskCategory",
            color_discrete_map=RISK_COLORS, category_orders={"RiskCategory": RISK_ORDER},
            barmode="stack", labels={"Count": "Customers", "Contract": ""},
        )
        fig_r1.update_layout(bargap=0.45)
        chart(fig_r1, height=370)

    with col_r2, st.container(border=True, key="panel-feature-importance"):
        section_header("What the model pays attention to", "Top 10 predictive features · aggregated importance")
        top10 = fi_df.head(10).sort_values("Importance")
        fig_r2 = px.bar(
            top10, x="Importance", y="OriginalFeature", orientation="h",
            color_discrete_sequence=[COLORS["teal"]],
            labels={"OriginalFeature": "", "Importance": "Importance score"},
        )
        chart(fig_r2, height=370)
    st.caption("Feature importance uses Random Forest impurity reduction or Logistic Regression absolute coefficients. Scores indicate predictive association, not causation.")

    section_header("High-risk customer shortlist", "Top 20 records ranked by predicted churn probability")
    display_cols = ["tenure", "Contract", "MonthlyCharges", "InternetService",
                    "PaymentMethod", "TechSupport", "ChurnProbability"]
    ranked = high_risk.sort_values("ChurnProbability", ascending=False)[display_cols].copy()
    # Keep source row numbers so exported records can be located in the CSV.
    ranked.insert(0, "SourceRow", ranked.index + 2)
    st.dataframe(
        ranked.head(20), hide_index=True, width="stretch",
        column_config={
            "SourceRow": st.column_config.NumberColumn("CSV row", help="Row number in the source CSV, including its header", format="%d"),
            "tenure": st.column_config.NumberColumn("Tenure (mo)", format="%d"),
            "MonthlyCharges": st.column_config.NumberColumn("Monthly charges", format="$%.2f"),
            "InternetService": "Internet service", "PaymentMethod": "Payment method", "TechSupport": "Tech support",
            "ChurnProbability": st.column_config.ProgressColumn("Churn probability", min_value=0, max_value=1, format="percent"),
        },
    )
    st.download_button("Download all high-risk records (CSV)", ranked.to_csv(index=False).encode("utf-8"),
                       file_name="high_risk_customers.csv", mime="text/csv")

    actions, opportunities = st.tabs(["Recommended actions", "Retention opportunities"])
    with actions:
        for i, text in enumerate(insights["actions"], 1):
            insight_card(text, i, "orange")
    with opportunities:
        left, right = st.columns(2)
        with left:
            churn_rate_chart(df, "PaperlessBilling", "Paperless billing")
        with right:
            grouped = df.copy()
            grouped["TenureGroup"] = pd.cut(grouped["tenure"], [0, 12, 24, 48, 72],
                                             labels=["0–12 mo", "13–24 mo", "25–48 mo", "49–72 mo"], include_lowest=True)
            churn_rate_chart(grouped, "TenureGroup", "Tenure & retention")
        for i, text in enumerate(insights["opportunities"], 1):
            insight_card(text, i, "green")


# ---- PAGE 4: Individual Customer Prediction ----

def page_prediction(pipeline, fi_df):
    page_header("Workspace / Customer prediction", "A clearer view of customer risk",
                "Enter a customer profile to estimate churn probability and explore a recommended next step.")
    form_column, guide_column = st.columns([2, 1])
    with form_column, st.form("prediction_form"):
        section_header("Customer profile", "Review all three sections before generating a prediction.")
        account, services, profile = st.tabs(["Account & billing", "Services", "Customer details"])
        with account:
            col1, col2 = st.columns(2)
            with col1:
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                tenure = st.slider("Tenure (months)", 0, 72, 12)
                payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                           "Bank transfer (automatic)", "Credit card (automatic)"])
            with col2:
                monthly_charges = st.number_input("Monthly charges ($)", 0.0, 200.0, 65.0, step=0.5)
                total_charges = st.number_input("Total charges ($)", 0.0, 10000.0, 780.0, step=1.0,
                                                help="Enter the actual cumulative billed amount; this field is not calculated automatically.")
                paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        with services:
            col1, col2, col3 = st.columns(3)
            service_options = ["Yes", "No", "No internet service"]
            with col1:
                internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
                phone_service = st.selectbox("Phone service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
            with col2:
                tech_support = st.selectbox("Tech support", service_options)
                online_security = st.selectbox("Online security", service_options)
                online_backup = st.selectbox("Online backup", service_options)
            with col3:
                device_prot = st.selectbox("Device protection", service_options)
                streaming_tv = st.selectbox("Streaming TV", service_options)
                streaming_movies = st.selectbox("Streaming movies", service_options)
        with profile:
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            with col2:
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
        submitted = st.form_submit_button("Generate risk prediction →", type="primary", width="stretch")

    with guide_column, st.container(border=True, key="panel-prediction-guide"):
        section_header("Reading your prediction")
        st.markdown('<p class="prediction-guide">The model combines billing, service, and customer details into '
                    'a probability score. Use the risk band to help prioritise your next conversation.</p>'
                    f'<div class="risk-band tone-green"><strong>Low risk</strong><span>0–{LOW_RISK_THRESHOLD:.0%}</span></div>'
                    f'<div class="risk-band tone-orange"><strong>Medium risk</strong><span>Above {LOW_RISK_THRESHOLD:.0%} to {MEDIUM_RISK_THRESHOLD:.0%}</span></div>'
                    f'<div class="risk-band tone-red"><strong>High risk</strong><span>Above {MEDIUM_RISK_THRESHOLD:.0%}</span></div>',
                    unsafe_allow_html=True)
        st.caption("19 customer attributes · the saved model scores your profile when you submit.")

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

        if proba <= LOW_RISK_THRESHOLD:
            risk_label = "Low risk"
            risk_tone = "green"
            action = (
                f"This customer's predicted churn probability is {proba:.1%}, which is at or below the "
                f"{LOW_RISK_THRESHOLD:.0%} Low Risk threshold. No immediate intervention required. "
                "Continue standard service quality."
            )
        elif proba <= MEDIUM_RISK_THRESHOLD:
            risk_label = "Medium risk"
            risk_tone = "orange"
            # Build a context-aware note based on the customer's own inputs
            _risk_factors = []
            if contract == "Month-to-month":
                _risk_factors.append("month-to-month contract (highest observed churn contract type)")
            if internet == "Fiber optic":
                _risk_factors.append("Fiber optic service (highest observed churn internet type)")
            if tech_support == "No":
                _risk_factors.append("no TechSupport (associated with higher churn in training data)")
            if online_security == "No":
                _risk_factors.append("no OnlineSecurity (associated with higher churn in training data)")
            _factors_str = "; ".join(_risk_factors) if _risk_factors else "review the top feature drivers below"
            action = (
                f"Predicted churn probability {proba:.1%} — in the Medium Risk band "
                f"({LOW_RISK_THRESHOLD:.0%}–{MEDIUM_RISK_THRESHOLD:.0%}). "
                f"Flagged risk factors for this customer: {_factors_str}. "
                "Consider a proactive outreach call and targeted service offer."
            )
        else:
            risk_label = "High risk"
            risk_tone = "red"
            _risk_factors = []
            if contract == "Month-to-month":
                _risk_factors.append("month-to-month contract")
            if internet == "Fiber optic":
                _risk_factors.append("Fiber optic internet")
            if tech_support == "No":
                _risk_factors.append("no TechSupport")
            if online_security == "No":
                _risk_factors.append("no OnlineSecurity")
            if senior == 1:
                _risk_factors.append("Senior Citizen")
            _factors_str = "; ".join(_risk_factors) if _risk_factors else "review the top feature drivers below"
            action = (
                f"Predicted churn probability {proba:.1%} — above the {MEDIUM_RISK_THRESHOLD:.0%} "
                f"High Risk threshold. Customer-specific risk factors present: {_factors_str}. "
                "Immediate retention outreach recommended: consider a contract upgrade incentive "
                "and/or bundling TechSupport and OnlineSecurity."
            )

        section_header("Your prediction", "Results for the customer profile submitted above")
        p1, p2, p3 = st.columns(3)
        with p1:
            kpi_card("Churn probability", f"{proba:.1%}", risk_tone, note="Model-estimated likelihood")
        with p2:
            kpi_card("Risk category", risk_label, risk_tone, note=f"Based on {LOW_RISK_THRESHOLD:.0%} and {MEDIUM_RISK_THRESHOLD:.0%} cutoffs")
        with p3:
            kpi_card("Predicted outcome", "Churn" if proba >= 0.5 else "Retain",
                     "red" if proba >= 0.5 else "green", note="Classification threshold: ≥ 50%")

        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0, "ticksuffix": "%"},
                "bar": {"color": COLORS[risk_tone], "thickness": 0.65},
                "borderwidth": 0,
                "steps": [
                    {"range": [0, LOW_RISK_THRESHOLD * 100], "color": "#eaf5f2"},
                    {"range": [LOW_RISK_THRESHOLD * 100, MEDIUM_RISK_THRESHOLD * 100], "color": "#fbf5e8"},
                    {"range": [MEDIUM_RISK_THRESHOLD * 100, 100], "color": "#fdf0f1"},
                ],
                "threshold": {
                    "line": {"color": COLORS["ink"], "width": 2},
                    "thickness": 0.75,
                    "value": proba * 100,
                },
            },
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 38, "color": COLORS["ink"]}},
        ))
        left, right = st.columns(2)
        with left, st.container(border=True, key="panel-probability-gauge"):
            section_header("Churn probability")
            chart(fig_gauge, height=280)
        with right, st.container(border=True, key="panel-prediction-features"):
            section_header("Top model-wide predictors", "Overall importance, not an explanation of this individual score")
            factors = fi_df.head(5).sort_values("Importance")
            fig = px.bar(factors, x="Importance", y="OriginalFeature", orientation="h",
                         color_discrete_sequence=[COLORS["teal"]], labels={"OriginalFeature": "", "Importance": "Importance score"})
            chart(fig, height=255)
        section_header("Recommended next step")
        insight_card(action, tone=risk_tone)


# ---- PAGE 5: Model Performance ----

def page_model_performance(results, X_test, y_test, best_name):
    page_header("Workspace / Model performance", "Confidence, backed by evaluation",
                "Compare the models and understand how well they identify customers who churn.",
                badge=f"Held-out test set · {len(y_test):,} customers")
    best_metrics = results[best_name]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Selected model ROC-AUC", f"{best_metrics['ROC-AUC']:.3f}", "teal", note=best_name)
    with c2:
        kpi_card("Recall", f"{best_metrics['Recall']:.1%}", "blue", note="Share of actual churners identified")
    with c3:
        kpi_card("Precision", f"{best_metrics['Precision']:.1%}", "orange", note="Correct among predicted churners")
    with c4:
        kpi_card("F1 score", f"{best_metrics['F1 Score']:.3f}", "teal", note="Balance of precision and recall")

    section_header("Model comparison", "Selected by highest ROC-AUC on the held-out test set")
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        for m in metric_names:
            row[m] = metrics[m]
        rows.append(row)
    comparison_df = pd.DataFrame(rows)
    comparison_df.insert(1, "Selection", ["Selected" if name == best_name else "Baseline" for name in comparison_df["Model"]])
    st.dataframe(
        comparison_df.style.highlight_max(subset=metric_names, color="#eaf5f2").format(
            {m: "{:.4f}" for m in metric_names}
        ),
        width="stretch", hide_index=True,
    )

    col1, col2 = st.columns(2)

    with col1, st.container(border=True, key="panel-roc-curves"):
        section_header("Discrimination across thresholds", "ROC curves · closer to the upper-left corner is better")
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dot", color="#bdc9d0"))
        colors = [COLORS["teal"], COLORS["blue"]]
        for (model_name, metrics), color in zip(results.items(), colors):
            fpr, tpr, _ = roc_curve(y_test, metrics["y_proba"])
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{model_name} (AUC={metrics['ROC-AUC']:.3f})",
                line=dict(color=color, width=2.5),
            ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        chart(fig_roc, height=370)

    with col2, st.container(border=True, key="panel-confusion-matrix"):
        section_header("Predictions vs actual outcomes", f"Confusion matrix · {best_name}")
        cm = confusion_matrix(y_test, best_metrics["y_pred"])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale=["#edf6f3", COLORS["teal"]],
            x=["Predicted No Churn", "Predicted Churn"],
            y=["Actual No Churn",    "Actual Churn"],
        )
        fig_cm.update_layout(coloraxis_showscale=False)
        chart(fig_cm, height=370)

    report = classification_report(y_test, best_metrics["y_pred"],
                                   target_names=["No Churn", "Churn"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    with st.expander("Detailed classification report"):
        st.dataframe(report_df.style.format("{:.3f}"), width="stretch")
    with st.expander("How to interpret these results"):
        st.write("ROC-AUC measures how well a model ranks churners above retained customers across thresholds. "
                 "Both classifiers use balanced class weights and an 80/20 stratified train–test split. "
                 "Recall measures how many actual churners are found; precision describes the reliability of churn alerts.")
        st.caption("The 50% classification threshold is a default. Retention costs and the relative cost of missed churners "
                   "are not measured in this dataset; threshold tuning should use actual programme costs.")


# ---------------------------------------------------------------------------
# MAIN APPLICATION ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    setup_page()

    # ---- Sidebar Navigation ----
    sidebar_brand()

    pages = {
        "Overview"             : "overview",
        "Customer analysis"    : "analysis",
        "Retention priorities" : "risk",
        "Customer prediction"  : "prediction",
        "Model performance"    : "model",
    }
    selected = st.sidebar.radio("Workspace navigation", list(pages.keys()),
                                key="workspace_navigation", label_visibility="collapsed")
    page_key = pages[selected]

    with st.sidebar.expander("Data source"):
        data_path = st.text_input("CSV path", value=DATASET_PATH,
                                  help="Path to the Telco Customer Churn CSV on this machine.")
        st.caption("IBM Telco Customer Churn · sample dataset")

    # ---- Load & Process Data ----
    try:
        raw_df = load_data(data_path)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    df, cleaning_log = clean_data(raw_df)

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
        st.success("Model trained and saved. Your workspace is ready.")
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
    sidebar_status(best_name, results[best_name]["ROC-AUC"], len(df))

    # ---- Render selected page ----
    if   page_key == "overview"   : page_overview(df, kpis, pred_df, insights, cleaning_log)
    elif page_key == "analysis"   : page_churn_analysis(df, pred_df)
    elif page_key == "risk"       : page_risk_action(df, pred_df, fi_df, kpis, insights)
    elif page_key == "prediction" : page_prediction(pipeline, fi_df)
    elif page_key == "model"      : page_model_performance(results, X_test, y_test, best_name)

    # Footer
    st.sidebar.markdown(
        '<div class="sidebar-footer"><strong>IBM SkillsBuild</strong><br>'
        'Data Analytics with AI<br>Built by Shivam Singh</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
