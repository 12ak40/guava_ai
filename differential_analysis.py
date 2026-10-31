import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import shap
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🧬 DIFFERENTIAL METABOLITE ANALYSIS")
print("   Using Only Significant Differential Metabolites")
print("="*70)

# ============================================
# 1. LOAD THE DIFFERENTIAL METABOLITES FILE
# ============================================
print("\n📂 Step 1: Loading Differentially Expressed Metabolites")
print("-" * 40)

# Load the differential metabolites file
diff_file = '01.Statistical Analysis/Differentially Expressed Metabolites.xlsx'
xls = pd.ExcelFile(diff_file)
print(f"   Sheets: {xls.sheet_names}")

# Load the main differential metabolites sheet
df_diff = pd.read_excel(diff_file, sheet_name='A1vsB1vsC1vsF1vsP1')
print(f"   Shape: {df_diff.shape}")
print(f"   Columns (first 10): {df_diff.columns.tolist()[:10]}")

# ============================================
# 2. LOAD MEAN DATA AND MERGE
# ============================================
print("\n📂 Step 2: Loading Mean Data")
print("-" * 40)

df_mean = pd.read_excel('01.Statistical Analysis/Mean.xlsx', sheet_name='Mean')
print(f"   Mean data shape: {df_mean.shape}")

# Get the significant differential metabolites (from the paper's criteria)
# The differential file likely contains VIP, log2FC, p-value columns
print(f"\n   Differential file columns: {df_diff.columns.tolist()}")

# ============================================
# 3. FILTER TO SIGNIFICANT DIFFERENTIAL METABOLITES
# ============================================
print("\n🔬 Step 3: Filtering to Significant Metabolites")
print("-" * 40)

# Try to find the significance columns
# Common columns: VIP, log2FC, p-value, FDR
sig_cols = [c for c in df_diff.columns if any(k in c.lower() for k in ['vip', 'log2', 'fc', 'pval', 'p-value', 'fdr', 'sig'])]
print(f"   Significance columns found: {sig_cols}")

# Use compound name to merge
if 'compound name' in df_diff.columns:
    diff_names = df_diff['compound name'].dropna().unique().tolist()
    print(f"   Differential metabolites: {len(diff_names)}")
    
    # Filter mean data to only these metabolites
    df_filtered = df_mean[df_mean['compound name'].isin(diff_names)]
    print(f"   After filtering: {df_filtered.shape[0]} metabolites")

# ============================================
# 4. DEFINE SAMPLES AND ORIGINS
# ============================================
print("\n📊 Step 4: Defining Features and Targets")
print("-" * 40)

sample_cols = ['A1_1', 'A1_2', 'A1_3',
               'B1_1', 'B1_2', 'B1_3',
               'C1_1', 'C1_2', 'C1_3',
               'F1_1', 'F1_2', 'F1_3',
               'P1_1', 'P1_2', 'P1_3']

origin_map = {
    'A1': 'Lowland', 'B1': 'Lowland',
    'C1': 'Amazonian', 'F1': 'Amazonian', 'P1': 'Amazonian'
}

origins = [origin_map[col.split('_')[0]] for col in sample_cols]

# ============================================
# 5. PREPARE FEATURE MATRIX
# ============================================
print("\n🔄 Step 5: Preparing Feature Matrix")
print("-" * 40)

X = df_filtered[sample_cols].T
X.columns = [str(c) for c in df_filtered['compound name'].astype(str).values]

if X.columns.duplicated().any():
    X.columns = [f"{c}_{i}" if i > 0 else c for i, c in enumerate(X.columns)]

y = pd.Series(origins, index=sample_cols, name='Origin')

print(f"   Feature matrix shape: {X.shape}")

# Clean
X = X.replace(0, np.nan)
X = X.loc[:, X.notna().sum() >= len(X) * 0.5]
X = X.fillna(X.median())
X_log = np.log1p(X)

print(f"   After cleaning: {X_log.shape}")

# ============================================
# 6. TRAIN WITH LEAVE-ONE-OUT CROSS-VALIDATION
# ============================================
print("\n📊 Step 6: Leave-One-Out Cross-Validation")
print("-" * 40)

model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)

# Leave-One-Out is more honest for small datasets
loo = LeaveOneOut()
predictions = []
actuals = []

for train_idx, test_idx in loo.split(X_log):
    X_train, X_test = X_log.iloc[train_idx], X_log.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)[0]
    predictions.append(pred)
    actuals.append(y_test.values[0])

loo_accuracy = accuracy_score(actuals, predictions)
print(f"   ✅ Leave-One-Out Accuracy: {loo_accuracy:.4f}")

# ============================================
# 7. FEATURE IMPORTANCE (with simple model)
# ============================================
print("\n🌲 Step 7: Feature Importance")
print("-" * 40)

# Train on all data for feature importance
model.fit(X_log, y)
importance = model.feature_importances_
feature_names = list(X_log.columns)
top_idx = np.argsort(importance)[-20:][::-1]

print("\n   🏆 Top 20 Significant Differential Markers:")
print(f"   {'Rank':<6} {'Metabolite':<50} {'Importance':<12}")
print("   " + "-" * 70)
for i, idx in enumerate(top_idx):
    name = str(feature_names[idx])[:48]
    print(f"   {i+1:<6} {name:<50} {importance[idx]:.6f}")

# ============================================
# 8. SHAP ANALYSIS
# ============================================
print("\n🔍 Step 8: SHAP Analysis")
print("-" * 40)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_log)

if isinstance(shap_values, list):
    shap_use = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_use = shap_values[:, :, 1]
else:
    shap_use = shap_values

mean_shap = np.abs(shap_use).mean(axis=0)
top_idx_shap = np.argsort(mean_shap)[-20:][::-1]

print("\n   🏆 Top 20 Markers by SHAP:")
for i, idx in enumerate(top_idx_shap):
    name = str(feature_names[idx])[:48]
    print(f"   {i+1:<6} {name:<50} {mean_shap[idx]:.6f}")

# ============================================
# 9. SAVE RESULTS
# ============================================
print("\n💾 Step 9: Saving Results")
print("-" * 40)

top_markers_df = pd.DataFrame({
    'Rank': range(1, 21),
    'Metabolite': [str(feature_names[i]) for i in top_idx_shap],
    'SHAP_Importance': [mean_shap[i] for i in top_idx_shap],
    'RF_Importance': [importance[i] for i in top_idx_shap]
})
top_markers_df.to_csv('differential_top_20_markers.csv', index=False)
print("   ✅ Saved: differential_top_20_markers.csv")

# ============================================
# 10. VISUALIZATIONS
# ============================================
print("\n📈 Step 10: Generating Visualizations")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

top_plot = top_markers_df.head(15)
labels = [str(n)[:38] for n in top_plot['Metabolite']]
axes[0].barh(range(len(top_plot)), top_plot['SHAP_Importance'], color='steelblue')
axes[0].set_yticks(range(len(top_plot)))
axes[0].set_yticklabels(labels, fontsize=8)
axes[0].set_xlabel('SHAP Importance', fontsize=12)
axes[0].set_title(f'Top 15 Differential Markers\nLOO Accuracy: {loo_accuracy:.1%}', fontsize=13)
axes[0].invert_yaxis()

# Class distribution
origin_counts = pd.Series(origins).value_counts()
axes[1].pie(origin_counts.values, labels=origin_counts.keys(),
            autopct='%1.0f%%', colors=['#2B6CB0', '#68D391'])
axes[1].set_title('Sample Distribution by Origin', fontsize=13)

plt.tight_layout()
plt.savefig('differential_classification_results.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: differential_classification_results.png")

# SHAP summary
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_use, X_log, feature_names=feature_names,
                  max_display=20, show=False)
plt.title('SHAP Summary: Top 20 Differential Markers', fontsize=14)
plt.tight_layout()
plt.savefig('differential_shap_summary.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: differential_shap_summary.png")

print("\n" + "="*70)
print("✅ DIFFERENTIAL ANALYSIS COMPLETE!")
print("="*70)
print(f"\n📊 Summary:")
print(f"   • Differential metabolites: {X_log.shape[1]}")
print(f"   • Samples: {X_log.shape[0]}")
print(f"   • Leave-One-Out accuracy: {loo_accuracy:.2%}")
print(f"   • Top marker: {str(feature_names[top_idx_shap[0]])}")
