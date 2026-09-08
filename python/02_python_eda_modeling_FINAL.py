"""
============================================================================
EMPLOYEE BURNOUT ANALYSIS — PHASE 2 & 3: PYTHON EDA + PREDICTIVE MODELING
============================================================================
Input : employee_burnout_clean.csv  (output of Phase 1 SQL script,
        18,590 rows, snake_case columns, zero missing values —
        complete-case filtered on burn_rate, resource_allocation,
        AND mental_fatigue_score)
Output: chart PNGs, model comparison table, feature importance table,
        and a ranked high-risk employee watchlist (CSV)
============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve, auc)

# ----------------------------------------------------------------------------
# Brand palette — kept identical to the Phase 4 Power BI theme so every chart
# in this portfolio (SQL exports, Python EDA, and the dashboard) reads as one
# consistent visual product rather than three disconnected tools.
# ----------------------------------------------------------------------------
PRIMARY, SECONDARY, SUCCESS = '#1F3A5F', '#4F9DDE', '#2E8B57'
WARNING, CRITICAL, BG, TEXT = '#F4A261', '#D62828', '#F5F7FA', '#333333'

os.makedirs('charts', exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': 'white', 'axes.edgecolor': '#CCCCCC',
    'text.color': TEXT, 'axes.labelcolor': TEXT, 'xtick.color': TEXT, 'ytick.color': TEXT,
    'font.size': 11, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
})

df = pd.read_csv('employee_burnout_clean.csv')

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

# ----------------------------------------------------------------------------
# NOTE ON employee_id: earlier versions of this pipeline sourced from a raw
# file where IDs arrived as mangled UTF-16-hex strings (e.g.
# "fffe32003000360033003200") requiring a byte-level decode step. The
# current source file (employee_burnout_postgres_fixed.csv) already
# provides clean, sequential IDs like "EMP000001" directly, so no decoding
# happens here — employee_burnout_clean.csv's employee_id column is passed
# through unchanged from Phase 1 SQL. All 18,590 values remain unique.
# ----------------------------------------------------------------------------


# ============================================================================
# PHASE 2 — DATA VALIDATION
# Confirms the SQL-cleaned dataset carries forward correctly: no duplicate
# IDs, all values within their logical ranges, and the complete-case
# guarantee established in Phase 1 (zero missing model inputs).
# ============================================================================
assert df['employee_id'].duplicated().sum() == 0, "Unexpected duplicate IDs"
assert df['burn_rate'].between(0, 1).all(), "burn_rate out of expected range"
print("Data validation passed.")
print(df.isnull().sum())


# ============================================================================
# PHASE 2 — CORRELATION ANALYSIS
# Correlation analysis on the complete-case analytical table.
# No imputation or pairwise-missing handling is required because Phase 1
# removed rows missing the numeric model inputs.
# ============================================================================
num_cols = ['designation', 'resource_allocation', 'mental_fatigue_score', 'burn_rate']
corr = df[num_cols].corr()
print(corr.round(3))
# mental_fatigue_score <-> burn_rate : ~0.945 (strongest)
# resource_allocation  <-> burn_rate : ~0.856
# designation           <-> burn_rate : ~0.738


# ============================================================================
# PHASE 2 — VISUALIZATION SUITE (9 charts)
# Each chart pairs a business objective with the figure; titles/axes are
# set explicitly rather than left to pandas defaults, since this is a
# portfolio deliverable meant to be read by a non-technical stakeholder.
# ============================================================================

# 1. Burn Rate Distribution — overall shape + where the High Risk line falls
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['burn_rate'], bins=30, color=SECONDARY, edgecolor='white')
ax.axvline(0.70, color=CRITICAL, linestyle='--', linewidth=2, label='High Risk threshold (0.70)')
ax.set_title('Burn Rate Distribution Across Workforce')
ax.set_xlabel('Burn Rate'); ax.set_ylabel('Number of Employees'); ax.legend()
plt.tight_layout(); plt.savefig('charts/01_burn_rate_distribution.png', dpi=140); plt.close()

# 2-4. Burn Rate by Gender / Company Type / WFH — simple grouped-mean bar charts
for col, fname, title in [
    ('gender', '02_burn_rate_by_gender.png', 'Average Burn Rate by Gender'),
    ('company_type', '03_burn_rate_by_company_type.png', 'Average Burn Rate by Company Type'),
    ('wfh_setup_available', '04_burn_rate_by_wfh.png', 'Average Burn Rate: WFH Available vs Not'),
]:
    means = df.groupby(col)['burn_rate'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(means.index, means.values, color=[PRIMARY, SECONDARY])
    for i, v in enumerate(means.values):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    ax.set_title(title); ax.set_ylabel('Average Burn Rate'); ax.set_ylim(0, 0.6)
    plt.tight_layout(); plt.savefig(f'charts/{fname}', dpi=140); plt.close()

# 5. Burn Rate by Designation — reveals the near-linear seniority effect
means = df.groupby('designation')['burn_rate'].mean().sort_index()
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(means.index.astype(str), means.values, color=SECONDARY)
ax.set_title('Average Burn Rate by Designation (Seniority Level)')
ax.set_xlabel('Designation (0 = Junior, 5 = Senior)'); ax.set_ylabel('Average Burn Rate')
plt.tight_layout(); plt.savefig('charts/05_burn_rate_by_designation.png', dpi=140); plt.close()

# 6. Mental Fatigue vs Burn Rate — scatter + trend line, visualizes r=0.945
sample = df.dropna(subset=['mental_fatigue_score', 'burn_rate']).sample(3000, random_state=42)
z = np.polyfit(df['mental_fatigue_score'].dropna(), df.loc[df['mental_fatigue_score'].notna(), 'burn_rate'], 1)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(sample['mental_fatigue_score'], sample['burn_rate'], alpha=0.25, s=12, color=PRIMARY)
xs = np.linspace(0, 10, 100)
ax.plot(xs, z[0]*xs + z[1], color=CRITICAL, linewidth=2, label='Trend (r=0.945)')
ax.set_title('Mental Fatigue Score vs Burn Rate')
ax.set_xlabel('Mental Fatigue Score'); ax.set_ylabel('Burn Rate'); ax.legend()
plt.tight_layout(); plt.savefig('charts/06_fatigue_vs_burnrate.png', dpi=140); plt.close()

# 7. Resource Allocation vs Burn Rate — near-monotonic workload effect
means = df.groupby('resource_allocation')['burn_rate'].mean().sort_index()
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(means.index, means.values, marker='o', color=PRIMARY, markerfacecolor=SECONDARY)
ax.set_title('Resource Allocation vs Average Burn Rate')
ax.set_xlabel('Resource Allocation (Workload Score)'); ax.set_ylabel('Average Burn Rate')
plt.tight_layout(); plt.savefig('charts/07_resource_vs_burnrate.png', dpi=140); plt.close()

# 8. Correlation Heatmap — single view of every numeric relationship
fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', ax=ax, vmin=0, vmax=1, linewidths=0.5, linecolor='white')
ax.set_title('Correlation Heatmap — Numeric Features')
plt.tight_layout(); plt.savefig('charts/08_correlation_heatmap.png', dpi=140); plt.close()

# 9. Boxplots — confirms the Phase 1 SQL IQR outlier findings visually
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
for i, col in enumerate(['resource_allocation', 'mental_fatigue_score', 'burn_rate']):
    axes[i].boxplot(df[col].dropna(), patch_artist=True,
                     boxprops=dict(facecolor=SECONDARY, color=PRIMARY),
                     medianprops=dict(color=CRITICAL, linewidth=2))
    axes[i].set_title(col.replace('_', ' ').title()); axes[i].set_xticks([])
fig.suptitle('Boxplots — Outlier Detection', fontweight='bold')
plt.tight_layout(); plt.savefig('charts/09_boxplots_outliers.png', dpi=140); plt.close()


# ============================================================================
# PHASE 3 — PREDICTIVE MODELING
# ============================================================================

# ---- Feature preparation ----
# Employee ID is intentionally excluded: it's a unique identifier with zero
# predictive signal, and including it risks the model "memorizing" rows.
features = ['gender', 'company_type', 'wfh_setup_available', 'designation',
            'resource_allocation', 'mental_fatigue_score']
# employee_id is kept alongside (not as a feature) purely so the final
# watchlist can identify *who* to follow up with — it is never passed to X.
model_df = df[['employee_id'] + features + ['risk']].copy()

# NOTE ON MISSING VALUES: earlier versions of this pipeline median-imputed
# resource_allocation and mental_fatigue_score here, since Phase 1 SQL only
# dropped rows missing burn_rate. Phase 1 now performs complete-case
# deletion at the source (dropping any row missing burn_rate,
# resource_allocation, OR mental_fatigue_score), so employee_burnout_clean.csv
# is guaranteed to have zero NULLs in every numeric field. Imputation is no
# longer needed; the assertion below simply confirms that guarantee holds
# rather than silently masking a regression if it doesn't.
assert model_df[['resource_allocation', 'mental_fatigue_score']].isnull().sum().sum() == 0, \
    "Unexpected missing values — Phase 1 SQL should have already dropped these rows"

# Encode categoricals — tree models don't need scaling, but Logistic
# Regression does, so scaling is applied conditionally below.
le_gender, le_company, le_wfh = LabelEncoder(), LabelEncoder(), LabelEncoder()
model_df['Gender'] = le_gender.fit_transform(model_df['gender'])
model_df['CompanyType'] = le_company.fit_transform(model_df['company_type'])
model_df['WFH_Setup'] = le_wfh.fit_transform(model_df['wfh_setup_available'])
model_df['target'] = (model_df['risk'] == 'High Risk').astype(int)

X = model_df[['Gender', 'CompanyType', 'WFH_Setup', 'designation', 'resource_allocation', 'mental_fatigue_score']]
X.columns = ['Gender', 'CompanyType', 'WFH_Setup', 'Designation', 'ResourceAllocation', 'MentalFatigueScore']
y = model_df['target']
ids = model_df['employee_id']

# Class balance check — 11.1% High Risk means accuracy alone would be a
# misleading metric; class_weight='balanced' is used on every model below
# so the minority class isn't ignored, and Recall/F1/ROC-AUC are the
# primary evaluation metrics rather than raw accuracy.
print(y.value_counts(normalize=True))

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, ids, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- Train all three models ----
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

dt = DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42)
dt.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# ---- Evaluate ----
def evaluate(name, model, X_eval, y_true):
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]
    return {
        'Model': name,
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred), 4),
        'Recall': round(recall_score(y_true, y_pred), 4),
        'F1': round(f1_score(y_true, y_pred), 4),
        'ROC_AUC': round(roc_auc_score(y_true, y_proba), 4),
    }, y_pred, y_proba

res_lr, pred_lr, proba_lr = evaluate('Logistic Regression', lr, X_test_scaled, y_test)
res_dt, pred_dt, proba_dt = evaluate('Decision Tree', dt, X_test, y_test)
res_rf, pred_rf, proba_rf = evaluate('Random Forest', rf, X_test, y_test)

# Full Random Forest prediction results for Power BI
rf_predictions = df.loc[X_test.index, ['employee_id']].copy()

rf_predictions['actual_risk'] = y_test.values
rf_predictions['predicted_risk'] = pred_rf
rf_predictions['burnout_probability'] = proba_rf

rf_predictions.to_csv('random_forest_predictions.csv', index=False)

print("\nSaved: random_forest_predictions.csv")
print(rf_predictions.head())

results_df = pd.DataFrame([res_lr, res_dt, res_rf])
print(results_df.to_string(index=False))
"""
Expected current model comparison (complete-case test set, n=4,648):
              Model  Accuracy  Precision  Recall     F1  ROC_AUC
Logistic Regression    0.9402     0.6592  0.9595  0.7814   0.9903
Decision Tree          0.9288     0.6128  0.9807  0.7543   0.9900
Random Forest          0.9380     0.6474  0.9749  0.7781   0.9894   <- SELECTED OPERATIONAL MODEL

Selection rationale: Random Forest prioritizes High-Risk recall (97.49%) while
maintaining stronger precision/F1 than the single Decision Tree and ensemble
stability. Logistic Regression remains the strongest interpretable benchmark.
"""

print("Random Forest confusion matrix:\n", confusion_matrix(y_test, pred_rf))
# [[3855  275]
#  [  13  505]]  -> TN=3855, FP=275, FN=13, TP=505; High-Risk recall=97.49%

# ---- Feature importance ----
fi = pd.DataFrame({
    'Feature': X.columns, 'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
print(fi.to_string(index=False))
# MentalFatigueScore 52.6% | ResourceAllocation 32.9% | Designation 12.6%
# WFH_Setup 1.1% | Gender 0.4% | CompanyType 0.3%

# ---- Probability of burnout / High-risk employee watchlist ----

watchlist = df.loc[X_test.index, [
    'employee_id',
    'gender',
    'company_type',
    'wfh_setup_available',
    'designation',
    'resource_allocation',
    'mental_fatigue_score',
    'burn_rate',
    'risk'
]].copy()

watchlist['actual_risk'] = y_test.values
watchlist['predicted_risk'] = pred_rf
watchlist['burnout_probability'] = proba_rf

watchlist = watchlist[watchlist['predicted_risk'] == 1].sort_values(
    'burnout_probability',
    ascending=False
)

watchlist.to_csv('high_risk_watchlist.csv', index=False)
# watchlist now leads with employee_id, e.g. EMP23341, so HR can act on it directly

print(f"Model flags {int((pred_rf==1).sum())} of {len(y_test)} test employees as High Risk "
      f"({(pred_rf==1).mean()*100:.1f}%).")

# ============================================================================
# END OF PHASE 2 & 3
# ============================================================================
