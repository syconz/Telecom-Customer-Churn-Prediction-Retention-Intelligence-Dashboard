"""
generate_report_assets.py
Generates all charts, tables, and model outputs for the project report.
Run once: python generate_report_assets.py
All images saved to ./report_assets/
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    ConfusionMatrixDisplay
)

OUT = "report_assets"
os.makedirs(OUT, exist_ok=True)

# ── consistent style ──────────────────────────────────────────────────────────
BLUE   = "#3b82d4"
RED    = "#ef4444"
GREEN  = "#22c55e"
ORANGE = "#f97316"
PURPLE = "#7c5cd8"
GREY   = "#94a3b8"
BG     = "#f8fafc"
PALETTE = [BLUE, RED, GREEN, ORANGE, PURPLE, GREY]

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   "#cbd5e1",
    "axes.labelcolor":  "#1e293b",
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.titlecolor":  "#1e293b",
    "xtick.color":      "#475569",
    "ytick.color":      "#475569",
    "text.color":       "#1e293b",
    "grid.color":       "#e2e8f0",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
})

TARGET = "Churn"
CATEGORICAL_COLS = [
    "gender","Partner","Dependents","PhoneService","MultipleLines",
    "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
    "TechSupport","StreamingTV","StreamingMovies","Contract",
    "PaperlessBilling","PaymentMethod",
]
NUMERICAL_COLS = ["tenure","MonthlyCharges","TotalCharges","SeniorCitizen"]

# ── load & clean ──────────────────────────────────────────────────────────────
print("Loading dataset…")
csv_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
if not os.path.exists(csv_path):
    sys.exit(f"ERROR: {csv_path} not found. Place the CSV in the project folder.")

raw = pd.read_csv(csv_path)
raw_rows = len(raw)

df = raw.copy()
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
blank_tc = df["TotalCharges"].isna().sum()
df.dropna(subset=["TotalCharges"], inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df.drop(columns=["customerID"], inplace=True)
before_dup = len(df)
df.drop_duplicates(inplace=True)
dupes_dropped = before_dup - len(df)
df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
clean_rows = len(df)

print(f"  Raw: {raw_rows} → Clean: {clean_rows} ({blank_tc} blank TotalCharges + {dupes_dropped} duplicates removed)")

churn_rate = df[TARGET].mean() * 100
retain_rate = 100 - churn_rate
avg_mc = df["MonthlyCharges"].mean()
avg_tenure = df["tenure"].mean()
total_rev = df["MonthlyCharges"].sum()
churned_n = df[TARGET].sum()

def save(fig, name, tight=True):
    path = os.path.join(OUT, name)
    if tight:
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    else:
        fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"  ✓  {name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Churn distribution donut
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4.5))
sizes = [100 - churn_rate, churn_rate]
labels = [f"Retained\n{retain_rate:.1f}%", f"Churned\n{churn_rate:.1f}%"]
wedges, texts = ax.pie(sizes, labels=labels, colors=[GREEN, RED],
                       startangle=90, wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2))
for t in texts:
    t.set_fontsize(11)
    t.set_fontweight("bold")
ax.set_title("Figure 1 — Overall Churn Distribution", pad=12)
fig.patch.set_facecolor(BG)
save(fig, "fig01_churn_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Churn by Contract Type (bar)
# ─────────────────────────────────────────────────────────────────────────────
ct = df.groupby("Contract")[TARGET].mean().mul(100).reset_index()
ct.columns = ["Contract", "ChurnRate"]
ct = ct.sort_values("ChurnRate", ascending=False)
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(ct["Contract"], ct["ChurnRate"], color=[RED, ORANGE, GREEN], edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, ct["ChurnRate"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Contract Type")
ax.set_title("Figure 2 — Churn Rate by Contract Type")
ax.set_ylim(0, ct["ChurnRate"].max() * 1.25)
ax.yaxis.grid(True); ax.set_axisbelow(True)
save(fig, "fig02_churn_by_contract.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Churn by Tenure (histogram)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
churned    = df[df[TARGET] == 1]["tenure"]
retained   = df[df[TARGET] == 0]["tenure"]
ax.hist(retained, bins=30, alpha=0.65, color=GREEN,  label=f"Retained (n={len(retained):,})", edgecolor="white")
ax.hist(churned,  bins=30, alpha=0.80, color=RED,    label=f"Churned  (n={len(churned):,})",  edgecolor="white")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Number of Customers")
ax.set_title("Figure 3 — Tenure Distribution by Churn Status")
ax.legend(framealpha=0.8)
ax.yaxis.grid(True); ax.set_axisbelow(True)
save(fig, "fig03_churn_by_tenure.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Churn by Internet Service
# ─────────────────────────────────────────────────────────────────────────────
inet = df.groupby("InternetService")[TARGET].mean().mul(100).reset_index()
inet.columns = ["InternetService", "ChurnRate"]
inet = inet.sort_values("ChurnRate", ascending=False)
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(inet["InternetService"], inet["ChurnRate"],
              color=[RED, ORANGE, GREEN][:len(inet)], edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, inet["ChurnRate"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Internet Service Type")
ax.set_title("Figure 4 — Churn Rate by Internet Service")
ax.set_ylim(0, inet["ChurnRate"].max() * 1.25)
ax.yaxis.grid(True); ax.set_axisbelow(True)
save(fig, "fig04_churn_by_internet.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Churn by Payment Method
# ─────────────────────────────────────────────────────────────────────────────
pm = df.groupby("PaymentMethod")[TARGET].mean().mul(100).reset_index()
pm.columns = ["PaymentMethod","ChurnRate"]
pm = pm.sort_values("ChurnRate", ascending=False)
pm["PaymentMethod"] = pm["PaymentMethod"].str.replace(" (automatic)", "\n(automatic)", regex=False)
fig, ax = plt.subplots(figsize=(7, 4))
colors = [RED, ORANGE, BLUE, GREEN]
bars = ax.barh(pm["PaymentMethod"], pm["ChurnRate"], color=colors[:len(pm)], edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, pm["ChurnRate"]):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontweight="bold", fontsize=10)
ax.set_xlabel("Churn Rate (%)")
ax.set_title("Figure 5 — Churn Rate by Payment Method")
ax.set_xlim(0, pm["ChurnRate"].max() * 1.3)
ax.xaxis.grid(True); ax.set_axisbelow(True)
save(fig, "fig05_churn_by_payment.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — Monthly Charges distribution (churned vs retained)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df[df[TARGET]==0]["MonthlyCharges"], bins=35, alpha=0.65,
        color=GREEN, label="Retained", edgecolor="white")
ax.hist(df[df[TARGET]==1]["MonthlyCharges"], bins=35, alpha=0.80,
        color=RED,   label="Churned",  edgecolor="white")
ax.set_xlabel("Monthly Charges ($)")
ax.set_ylabel("Number of Customers")
ax.set_title("Figure 6 — Monthly Charges Distribution by Churn Status")
ax.legend(framealpha=0.8)
ax.yaxis.grid(True); ax.set_axisbelow(True)
save(fig, "fig06_monthly_charges.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — Churn by Senior Citizen & Gender (grouped bar)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# Senior citizen
sc = df.groupby("SeniorCitizen")[TARGET].mean().mul(100).reset_index()
sc["Label"] = sc["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
axes[0].bar(sc["Label"], sc[TARGET], color=[BLUE, RED], edgecolor="white", linewidth=1.5)
for i, val in enumerate(sc[TARGET]):
    axes[0].text(i, val + 0.5, f"{val:.1f}%", ha="center", fontweight="bold")
axes[0].set_ylabel("Churn Rate (%)")
axes[0].set_title("Churn by Senior Citizen Status")
axes[0].set_ylim(0, sc[TARGET].max() * 1.3)
axes[0].yaxis.grid(True); axes[0].set_axisbelow(True)
# Gender
gn = df.groupby("gender")[TARGET].mean().mul(100).reset_index()
axes[1].bar(gn["gender"], gn[TARGET], color=[PURPLE, ORANGE], edgecolor="white", linewidth=1.5)
for i, val in enumerate(gn[TARGET]):
    axes[1].text(i, val + 0.5, f"{val:.1f}%", ha="center", fontweight="bold")
axes[1].set_ylabel("Churn Rate (%)")
axes[1].set_title("Churn by Gender")
axes[1].set_ylim(0, gn[TARGET].max() * 1.3)
axes[1].yaxis.grid(True); axes[1].set_axisbelow(True)
fig.suptitle("Figure 7 — Churn by Demographics", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "fig07_churn_demographics.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — Churn by TechSupport & OnlineSecurity (grouped bar)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col, title in [(axes[0], "TechSupport", "Tech Support"),
                       (axes[1], "OnlineSecurity", "Online Security")]:
    grp = df[df[col] != "No internet service"].groupby(col)[TARGET].mean().mul(100).reset_index()
    grp.columns = [col, "ChurnRate"]
    grp = grp.sort_values("ChurnRate", ascending=False)
    clrs = [RED if v == grp["ChurnRate"].max() else GREEN for v in grp["ChurnRate"]]
    ax.bar(grp[col], grp["ChurnRate"], color=clrs, edgecolor="white", linewidth=1.5)
    for i, (_, row) in enumerate(grp.iterrows()):
        ax.text(i, row["ChurnRate"] + 0.5, f"{row['ChurnRate']:.1f}%",
                ha="center", fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title(f"Churn by {title}")
    ax.set_ylim(0, grp["ChurnRate"].max() * 1.3)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
fig.suptitle("Figure 8 — Churn by Service Subscription", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "fig08_churn_services.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 — Correlation heatmap (numerical features)
# ─────────────────────────────────────────────────────────────────────────────
num_df = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", TARGET]]
corr = num_df.corr()
fig, ax = plt.subplots(figsize=(6, 5))
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True  # show lower triangle
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            ax=ax, linewidths=0.5, linecolor="#e2e8f0",
            cbar_kws={"shrink": 0.8}, annot_kws={"size": 10})
ax.set_title("Figure 9 — Correlation Matrix (Numerical Features)")
plt.tight_layout()
save(fig, "fig09_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 — ML pipeline architecture diagram
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 2.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
steps = [
    ("Raw CSV\n7,043 rows", 0.6, BLUE),
    ("Data\nCleaning", 1.8, ORANGE),
    ("EDA &\nInsights", 3.0, GREEN),
    ("Feature\nEngineering", 4.2, PURPLE),
    ("Train / Test\nSplit (80/20)", 5.4, BLUE),
    ("Model\nTraining", 6.6, ORANGE),
    ("Evaluation\n& Selection", 7.8, RED),
    ("Dashboard\n& Deployment", 9.0, GREEN),
]
for label, x, color in steps:
    rect = mpatches.FancyBboxPatch((x - 0.52, 0.7), 1.04, 1.4,
        boxstyle="round,pad=0.08", facecolor=color, alpha=0.18,
        edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x, 1.42, label, ha="center", va="center", fontsize=7.5,
            fontweight="bold", color=color, multialignment="center")
    if x < 9.0:
        ax.annotate("", xy=(x + 0.62, 1.42), xytext=(x + 0.52, 1.42),
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.8))
ax.set_title("Figure 10 — End-to-End Machine Learning Pipeline", fontsize=12, fontweight="bold", pad=6)
save(fig, "fig10_ml_pipeline.png")

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN MODELS for FIG 11, 12, 13
# ─────────────────────────────────────────────────────────────────────────────
print("Training models for evaluation charts…")
cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
num_cols = [c for c in NUMERICAL_COLS  if c in df.columns]
X = df[[c for c in df.columns if c != TARGET]]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                  max_depth=10, random_state=42, n_jobs=-1),
}
results = {}
pipelines = {}
for name, clf in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    results[name] = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1 Score":  f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_proba),
        "y_pred":    y_pred,
        "y_proba":   y_proba,
    }
    pipelines[name] = pipe
    print(f"  {name}: AUC={results[name]['ROC-AUC']:.3f}  F1={results[name]['F1 Score']:.3f}")

best_name = max(results, key=lambda k: results[k]["ROC-AUC"])
print(f"  Best model: {best_name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 11 — Model Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
for i, (name, color) in enumerate(zip(results.keys(), [BLUE, ORANGE])):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i*width - width/2, vals, width, label=name, color=color,
                  edgecolor="white", linewidth=1.2, alpha=0.9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.18)
ax.set_title("Figure 11 — Model Performance Comparison")
ax.legend(framealpha=0.8)
ax.yaxis.grid(True); ax.set_axisbelow(True)
best_patch = mpatches.Patch(color="none", label=f"★ Best by ROC-AUC: {best_name}")
ax.legend(handles=ax.get_legend().legend_handles + [best_patch], framealpha=0.8)
save(fig, "fig11_model_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 12 — Confusion Matrices (side by side)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
fig.suptitle("Figure 12 — Confusion Matrices", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "fig12_confusion_matrices.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 13 — ROC Curves
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
for name, color in zip(results.keys(), [BLUE, ORANGE]):
    fpr, tpr, _ = roc_curve(y_test, results[name]["y_proba"])
    auc = results[name]["ROC-AUC"]
    ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{name}  (AUC = {auc:.3f})")
ax.plot([0,1],[0,1], "k--", lw=1.2, label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.06, color=BLUE)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Figure 13 — ROC Curves")
ax.legend(loc="lower right", framealpha=0.85)
ax.yaxis.grid(True); ax.xaxis.grid(True); ax.set_axisbelow(True)
save(fig, "fig13_roc_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 14 — Feature Importance (best model)
# ─────────────────────────────────────────────────────────────────────────────
best_pipe = pipelines[best_name]
clf       = best_pipe.named_steps["classifier"]
ohe       = best_pipe.named_steps["preprocessor"].named_transformers_["cat"]
ohe_cols  = list(ohe.get_feature_names_out(cat_cols))
all_feats = num_cols + ohe_cols
if hasattr(clf, "feature_importances_"):
    imps = clf.feature_importances_
else:
    imps = np.abs(clf.coef_[0])
fi = pd.DataFrame({"Feature": all_feats, "Importance": imps})
fi["OriginalFeature"] = fi["Feature"].apply(
    lambda x: x.split("_")[0] if any(c in x for c in cat_cols) else x)
fi_agg = fi.groupby("OriginalFeature")["Importance"].sum().reset_index()
fi_agg = fi_agg.sort_values("Importance", ascending=True).tail(10)

fig, ax = plt.subplots(figsize=(7, 5))
colors = [RED if i >= len(fi_agg)-3 else BLUE for i in range(len(fi_agg))]
bars = ax.barh(fi_agg["OriginalFeature"], fi_agg["Importance"],
               color=colors, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, fi_agg["Importance"]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9, fontweight="bold")
ax.set_xlabel("Importance Score (|coefficient| aggregated per feature)")
ax.set_title(f"Figure 14 — Top 10 Churn Drivers\n({best_name} — association only, not causation)")
ax.xaxis.grid(True); ax.set_axisbelow(True)
red_patch   = mpatches.Patch(color=RED,  label="Top 3 drivers")
blue_patch  = mpatches.Patch(color=BLUE, label="Other top drivers")
ax.legend(handles=[red_patch, blue_patch], framealpha=0.8)
save(fig, "fig14_feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 15 — Risk Distribution (pie)
# ─────────────────────────────────────────────────────────────────────────────
proba_all = best_pipe.predict_proba(X)[:, 1]
low    = (proba_all <  0.30).sum()
medium = ((proba_all >= 0.30) & (proba_all <= 0.60)).sum()
high   = (proba_all >  0.60).sum()
total  = len(proba_all)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sizes  = [low, medium, high]
labels = [f"Low Risk\n{low:,}  ({100*low/total:.1f}%)",
          f"Medium Risk\n{medium:,}  ({100*medium/total:.1f}%)",
          f"High Risk\n{high:,}  ({100*high/total:.1f}%)"]
wedges, texts = ax.pie(sizes, labels=labels, colors=[GREEN, ORANGE, RED],
                       startangle=90, wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2))
for t in texts:
    t.set_fontsize(9.5)
ax.set_title("Figure 15 — Customer Risk Distribution\n(ML Model Predictions)", pad=10)
save(fig, "fig15_risk_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 16 — KPI summary table screenshot
# ─────────────────────────────────────────────────────────────────────────────
high_risk_rev = df[proba_all > 0.60]["MonthlyCharges"].sum()
kpi_rows = [
    ["Total Customers",               f"{clean_rows:,}"],
    ["Churned Customers",             f"{int(churned_n):,}"],
    ["Retained Customers",            f"{clean_rows - int(churned_n):,}"],
    ["Churn Rate",                    f"{churn_rate:.1f}%"],
    ["Retention Rate",                f"{retain_rate:.1f}%"],
    ["Avg Monthly Charges",           f"${avg_mc:.2f}"],
    ["Avg Tenure",                    f"{avg_tenure:.1f} months"],
    ["Total Monthly Revenue",         f"${total_rev:,.0f}"],
    ["High-Risk Customers (ML)",      f"{high:,}"],
    ["Model-Predicted Revenue at Risk",f"${high_risk_rev:,.0f}/mo"],
]
fig, ax = plt.subplots(figsize=(7, 4))
ax.axis("off")
tbl = ax.table(
    cellText=kpi_rows,
    colLabels=["KPI", "Value"],
    cellLoc="left", loc="center",
    colWidths=[0.62, 0.38],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.55)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cbd5e1")
    if row == 0:
        cell.set_facecolor(BLUE)
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#eff6ff")
    else:
        cell.set_facecolor("white")
    if col == 1:
        cell.get_text().set_ha("right")
ax.set_title("Figure 16 — Business KPI Summary", fontsize=12, fontweight="bold", pad=8)
save(fig, "fig16_kpi_table.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 17 — Model Metrics Table
# ─────────────────────────────────────────────────────────────────────────────
metric_rows = []
for name, r in results.items():
    marker = " ★ BEST" if name == best_name else ""
    metric_rows.append([
        f"{name}{marker}",
        f"{r['Accuracy']:.4f}",
        f"{r['Precision']:.4f}",
        f"{r['Recall']:.4f}",
        f"{r['F1 Score']:.4f}",
        f"{r['ROC-AUC']:.4f}",
    ])
fig, ax = plt.subplots(figsize=(10, 2.2))
ax.axis("off")
tbl = ax.table(
    cellText=metric_rows,
    colLabels=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
    cellLoc="center", loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.0)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cbd5e1")
    if row == 0:
        cell.set_facecolor(BLUE)
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#eff6ff")
    else:
        cell.set_facecolor("white")
ax.set_title("Figure 17 — Model Evaluation Metrics (Test Set)", fontsize=12, fontweight="bold", pad=6)
save(fig, "fig17_model_metrics_table.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 18 — Dashboard Page 1 mockup (Executive Overview)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 7))
fig.patch.set_facecolor("#0f172a")
# Top KPI cards
kpi_items = [
    ("Total Customers", f"{clean_rows:,}", BLUE),
    ("Churn Rate",      f"{churn_rate:.1f}%", RED),
    ("Retention Rate",  f"{retain_rate:.1f}%", GREEN),
    ("Monthly Revenue", f"${total_rev/1e6:.2f}M", PURPLE),
    ("Avg Monthly Charge", f"${avg_mc:.0f}", ORANGE),
    ("High-Risk Customers",f"{high:,}", RED),
]
for i, (label, val, color) in enumerate(kpi_items):
    ax = fig.add_axes([0.02 + i*0.163, 0.76, 0.148, 0.20])
    ax.set_facecolor("#1e293b"); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.text(0.5, 0.72, val,   ha="center", va="center", fontsize=16, fontweight="bold", color=color)
    ax.text(0.5, 0.25, label, ha="center", va="center", fontsize=8,  color="#94a3b8", wrap=True)
    for spine in ["bottom"]:
        ax.axhline(0.02, color=color, lw=3, xmin=0.1, xmax=0.9)
# Donut (left bottom)
ax2 = fig.add_axes([0.03, 0.06, 0.28, 0.65])
ax2.set_facecolor("#1e293b"); ax2.axis("off")
ax2.text(0.5, 0.95, "Churn Distribution", ha="center", fontsize=10,
         fontweight="bold", color="white", transform=ax2.transAxes)
wedge_angles = [retain_rate/100 * 360, churn_rate/100 * 360]
colors_w = [GREEN, RED]
start = 90
for angle, c in zip(wedge_angles, colors_w):
    theta = np.linspace(np.radians(start), np.radians(start - angle), 100)
    x_in  = 0.28 * np.cos(theta) + 0.5
    y_in  = 0.28 * np.sin(theta) * 0.9 + 0.47
    x_out = 0.42 * np.cos(theta) + 0.5
    y_out = 0.42 * np.sin(theta) * 0.9 + 0.47
    ax2.fill(np.concatenate([x_in, x_out[::-1]]),
             np.concatenate([y_in, y_out[::-1]]), color=c, alpha=0.85)
    start -= angle
ax2.text(0.5, 0.47, f"{churn_rate:.1f}%\nChurn", ha="center", va="center",
         fontsize=13, fontweight="bold", color=RED, transform=ax2.transAxes)
# Bar chart (right bottom)
ax3 = fig.add_axes([0.36, 0.06, 0.30, 0.65])
ax3.set_facecolor("#1e293b")
ct_vals = df.groupby("Contract")[TARGET].mean().mul(100)
ypos = np.arange(len(ct_vals))
bars = ax3.barh(ypos, ct_vals.values, color=[RED, ORANGE, GREEN], edgecolor="none")
ax3.set_yticks(ypos); ax3.set_yticklabels(ct_vals.index, color="#cbd5e1", fontsize=8)
ax3.set_xlabel("Churn Rate (%)", color="#94a3b8", fontsize=8)
ax3.set_title("Churn by Contract", color="white", fontsize=9, fontweight="bold")
ax3.tick_params(colors="#94a3b8"); ax3.xaxis.grid(True, color="#334155", alpha=0.5)
ax3.set_facecolor("#1e293b"); ax3.spines[:].set_color("#334155")
for bar, val in zip(bars, ct_vals.values):
    ax3.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}%", va="center", color="white", fontsize=8)
# Risk pie (far right)
ax4 = fig.add_axes([0.70, 0.06, 0.28, 0.65])
ax4.set_facecolor("#1e293b"); ax4.axis("off")
ax4.text(0.5, 0.95, "Risk Segmentation", ha="center", fontsize=10,
         fontweight="bold", color="white", transform=ax4.transAxes)
risk_sizes = [low/total, medium/total, high/total]
start = 90
for frac, c in zip(risk_sizes, [GREEN, ORANGE, RED]):
    angle = frac * 360
    theta = np.linspace(np.radians(start), np.radians(start - angle), 100)
    x_in  = 0.28 * np.cos(theta) + 0.5
    y_in  = 0.28 * np.sin(theta) * 0.9 + 0.47
    x_out = 0.42 * np.cos(theta) + 0.5
    y_out = 0.42 * np.sin(theta) * 0.9 + 0.47
    ax4.fill(np.concatenate([x_in, x_out[::-1]]),
             np.concatenate([y_in, y_out[::-1]]), color=c, alpha=0.85)
    start -= angle
for label, c, ypos in [("● High", RED, 0.18), ("● Medium", ORANGE, 0.12), ("● Low", GREEN, 0.06)]:
    ax4.text(0.5, ypos, label, ha="center", fontsize=8, color=c, transform=ax4.transAxes)
fig.text(0.5, 0.99, "Dashboard Page 1 — Executive Overview", ha="center", va="top",
         fontsize=12, fontweight="bold", color="white")
save(fig, "fig18_dashboard_page1.png")

print(f"\n✅ All {len(os.listdir(OUT))} report assets saved to ./{OUT}/")
print("Assets list:")
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(os.path.join(OUT, f)) / 1024
    print(f"  {f}  ({size:.0f} KB)")
