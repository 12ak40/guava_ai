import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import shap
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🧬 REAL GUAVA PHENOLIC ANALYSIS")
print("   Filtered to Phenolic Metabolites Only")
print("="*70)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n📂 Step 1: Loading Data")
print("-" * 40)

df = pd.read_excel('01.Statistical Analysis/Mean.xlsx', sheet_name='Mean')
print(f"   ✅ Loaded: {df.shape[0]} metabolites × {df.shape[1]} columns")

# ============================================
# 2. FILTER TO PHENOLIC METABOLITES ONLY
# ============================================
print("\n🔬 Step 2: Filtering to Phenolic Metabolites")
print("-" * 40)

# The paper focused on phenolics - filter using Superclass and Class
# Phenolic classes from the paper: Shikimates and Phenylpropanoids, Flavonoids

phenolic_keywords = [
    'Shikimates', 'Phenylpropanoids',
    'Flavonoids', 'Phenolic', 'Phenol',
    'Flavone', 'Flavonol', 'Flavanone', 'Flavanonol',
    'Isoflavonoid', 'Isoflavone', 'Anthocyanin',
    'Chalcone', 'Dihydrochalcone', 'Stilbene',
    'Coumarin', 'Lignan', 'Tannin',
    'Phenolic acid', 'Hydroxycinnamic', 'Benzoic'
]

# Filter by Superclass or Class containing phenolic keywords
mask = pd.Series([False] * len(df), index=df.index)
for kw in phenolic_keywords:
    mask |= df['Superclass'].astype(str).str.contains(kw, case=False, na=False)
    mask |= df['Class'].astype(str).str.contains(kw, case=False, na=False)
    mask |= df['Subclass'].astype(str).str.contains(kw, case=False, na=False)

df_phenolic = df[mask].copy()
print(f"   Total metabolites: {df.shape[0]}")
print(f"   Phenolic metabolites: {df_phenolic.shape[0]}")

# Remove unannotated
df_phenolic = df_phenolic.dropna(subset=['compound name'])
df_phenolic = df_phenolic[df_phenolic['compound name'].astype(str).str.strip() != '']
df_phenolic = df_phenolic.drop_duplicates(subset=['compound name'], keep='first')
print(f"   After removing unannotated: {df_phenolic.shape[0]}")

# ============================================
# 3. DEFINE SAMPLE COLUMNS AND ORIGINS
# ============================================
print("\n📊 Step 3: Defining Features and Targets")
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
# 4. PREPARE FEATURE MATRIX
# ============================================
print("\n🔄 Step 4: Preparing Feature Matrix")
print("-" * 40)

X = df_phenolic[sample_cols].T
X.columns = [str(c) for c in df_phenolic['compound name'].astype(str).values]

# Handle duplicate names
if X.columns.duplicated().any():
    X.columns = [f"{c}_{i}" if i > 0 else c for i, c in enumerate(X.columns)]

y = pd.Series(origins, index=sample_cols, name='Origin')

print(f"   Feature matrix shape: {X.shape}")

# ============================================
# 5. CLEAN DATA
# ============================================
print("\n🧹 Step 5: Cleaning Data")
print("-" * 40)

X = X.replace(0, np.nan)
X = X.loc[:, X.notna().sum() >= len(X) * 0.5]
X = X.fillna(X.median())
X_log = np.log1p(X)

print(f"   After cleaning: {X_log.shape}")

# ============================================
# 6. TRAIN CLASSIFIER (with cross-validation)
# ============================================
print("\n📊 Step 6: Training Origin Classifier")
print("-" * 40)

# Use cross-validation for more reliable estimate
cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    X_log, y, cv=5
)
print(f"   ✅ Cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Train on full data
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_log, y)
y_pred = model.predict(X_log)
accuracy = accuracy_score(y, y_pred)
print(f"   ✅ Training accuracy: {accuracy:.4f}")

# ============================================
# 7. FEATURE IMPORTANCE
# ============================================
print("\n🌲 Step 7: Feature Importance")
print("-" * 40)

importance = model.feature_importances_
feature_names = list(X_log.columns)
top_idx = np.argsort(importance)[-20:][::-1]

print("\n   🏆 Top 20 Phenolic Markers:")
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

print("\n   🏆 Top 20 Phenolic Markers (SHAP):")
print(f"   {'Rank':<6} {'Metabolite':<50} {'SHAP':<12}")
print("   " + "-" * 70)
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
top_markers_df.to_csv('phenolic_top_20_markers.csv', index=False)
print("   ✅ Saved: phenolic_top_20_markers.csv")

# ============================================
# 10. VISUALIZATIONS
# ============================================
print("\n📈 Step 10: Generating Visualizations")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Top markers plot
top_plot = top_markers_df.head(15)
labels = [str(n)[:38] for n in top_plot['Metabolite']]
axes[0].barh(range(len(top_plot)), top_plot['SHAP_Importance'], color='steelblue')
axes[0].set_yticks(range(len(top_plot)))
axes[0].set_yticklabels(labels, fontsize=8)
axes[0].set_xlabel('SHAP Importance', fontsize=12)
axes[0].set_title('Top 15 Phenolic Markers for Origin Classification', fontsize=13)
axes[0].invert_yaxis()

# Class distribution
origin_counts = pd.Series(origins).value_counts()
axes[1].pie(origin_counts.values, labels=origin_counts.index,
            autopct='%1.0f%%', colors=['#2B6CB0', '#68D391'])
axes[1].set_title('Sample Distribution by Origin', fontsize=13)

plt.tight_layout()
plt.savefig('phenolic_classification_results.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: phenolic_classification_results.png")

# SHAP summary
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_use, X_log, feature_names=feature_names,
                  max_display=20, show=False)
plt.title('SHAP Summary: Top 20 Phenolic Markers', fontsize=14)
plt.tight_layout()
plt.savefig('phenolic_shap_summary.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: phenolic_shap_summary.png")

print("\n" + "="*70)
print("✅ PHENOLIC ANALYSIS COMPLETE!")
print("="*70)
print(f"\n📊 Summary:")
print(f"   • Phenolic metabolites analyzed: {X_log.shape[1]}")
print(f"   • Samples: {X_log.shape[0]}")
print(f"   • Cross-validation accuracy: {cv_scores.mean():.2%}")
print(f"   • Top marker: {str(feature_names[top_idx_shap[0]])}")
