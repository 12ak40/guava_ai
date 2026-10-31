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
print("🧬 REAL GUAVA METABOLOMICS ANALYSIS")
print("   Origin Classification & Key Marker Identification")
print("="*70)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n📂 Step 1: Loading Data")
print("-" * 40)

df = pd.read_excel('01.Statistical Analysis/Mean.xlsx', sheet_name='Mean')
print(f"   ✅ Loaded: {df.shape[0]} metabolites × {df.shape[1]} columns")

# ============================================
# 2. FILTER TO ANNOTATED METABOLITES
# ============================================
print("\n🔬 Step 2: Filtering to Annotated Metabolites")
print("-" * 40)

# Keep only metabolites with a valid compound name (not NaN, not empty)
df_clean = df.dropna(subset=['compound name'])
df_clean = df_clean[df_clean['compound name'].astype(str).str.strip() != '']
df_clean = df_clean[df_clean['compound name'].astype(str) != 'nan']

# Remove duplicates by compound name (keep first)
df_clean = df_clean.drop_duplicates(subset=['compound name'], keep='first')

print(f"   Original: {df.shape[0]} features")
print(f"   After removing unannotated: {df_clean.shape[0]} features")

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

origins = []
for col in sample_cols:
    site = col.split('_')[0]
    origins.append(origin_map[site])

print(f"   Sample columns: {len(sample_cols)}")
print(f"   Origins: {origins}")

# ============================================
# 4. PREPARE FEATURE MATRIX
# ============================================
print("\n🔄 Step 4: Preparing Feature Matrix")
print("-" * 40)

# Extract metabolite intensities and transpose
X = df_clean[sample_cols].T
X.columns = df_clean['compound name'].astype(str).values

# Make sure column names are unique strings
X.columns = [str(c) for c in X.columns]
# Handle duplicates
if X.columns.duplicated().any():
    X.columns = [f"{c}_{i}" if i > 0 else c for i, c in enumerate(X.columns)]

y = pd.Series(origins, index=sample_cols, name='Origin')

print(f"   Feature matrix shape: {X.shape}")
print(f"   Target shape: {y.shape}")

# ============================================
# 5. CLEAN DATA
# ============================================
print("\n🧹 Step 5: Cleaning Data")
print("-" * 40)

# Replace zeros with NaN and then impute
X = X.replace(0, np.nan)

# Remove features with >50% missing values
X = X.loc[:, X.notna().sum() >= len(X) * 0.5]
print(f"   After removing sparse features: {X.shape}")

# Fill remaining NaN with column median
X = X.fillna(X.median())

# Log transform
X_log = np.log1p(X)

print(f"   ✅ Data cleaned: {X_log.shape}")

# ============================================
# 6. TRAIN CLASSIFICATION MODEL
# ============================================
print("\n📊 Step 6: Training Origin Classifier")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X_log, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"   ✅ Accuracy: {accuracy:.4f}")
print(f"   ✅ Cross-validation (5-fold): {cross_val_score(model, X_log, y, cv=5).mean():.4f}")

cm = confusion_matrix(y_test, y_pred, labels=['Amazonian', 'Lowland'])
print(f"\n   Confusion Matrix:")
print(f"                 Predicted")
print(f"                 Amazonian  Lowland")
print(f"   Actual Amazonian    {cm[0][0]:>3}       {cm[0][1]:>3}")
print(f"   Actual Lowland      {cm[1][0]:>3}       {cm[1][1]:>3}")

# ============================================
# 7. FEATURE IMPORTANCE (Random Forest)
# ============================================
print("\n🌲 Step 7: Random Forest Feature Importance")
print("-" * 40)

importance = model.feature_importances_
top_idx_rf = np.argsort(importance)[-20:][::-1]
feature_names = list(X_log.columns)

print("\n   🏆 Top 20 Markers (Random Forest Importance):")
print(f"   {'Rank':<6} {'Metabolite':<45} {'Importance':<12}")
print("   " + "-" * 65)
for i, idx in enumerate(top_idx_rf):
    name = str(feature_names[idx])[:43]
    print(f"   {i+1:<6} {name:<45} {importance[idx]:.6f}")

# ============================================
# 8. SHAP ANALYSIS
# ============================================
print("\n🔍 Step 8: SHAP Analysis")
print("-" * 40)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_log)

# Handle different SHAP output formats
if isinstance(shap_values, list):
    shap_use = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_use = shap_values[:, :, 1]
else:
    shap_use = shap_values

mean_shap = np.abs(shap_use).mean(axis=0)
top_idx_shap = np.argsort(mean_shap)[-20:][::-1]

print("\n   🏆 Top 20 Markers (SHAP Importance):")
print(f"   {'Rank':<6} {'Metabolite':<45} {'SHAP':<12}")
print("   " + "-" * 65)
for i, idx in enumerate(top_idx_shap):
    name = str(feature_names[idx])[:43]
    print(f"   {i+1:<6} {name:<45} {mean_shap[idx]:.6f}")

# ============================================
# 9. SAVE RESULTS
# ============================================
print("\n💾 Step 9: Saving Results")
print("-" * 40)

# Top markers (combined RF + SHAP)
top_markers_df = pd.DataFrame({
    'Rank': range(1, 21),
    'Metabolite': [str(feature_names[i]) for i in top_idx_shap],
    'SHAP_Importance': [mean_shap[i] for i in top_idx_shap],
    'RF_Importance': [importance[i] for i in top_idx_shap]
})
top_markers_df.to_csv('top_20_markers.csv', index=False)
print("   ✅ Saved: top_20_markers.csv")

# Classification results
results_df = pd.DataFrame({
    'Sample': X_test.index,
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Correct': y_test.values == y_pred
})
results_df.to_csv('classification_results.csv', index=False)
print("   ✅ Saved: classification_results.csv")

# ============================================
# 10. VISUALIZATIONS
# ============================================
print("\n📈 Step 10: Generating Visualizations")
print("-" * 40)

# Plot 1: Confusion Matrix + Top Markers
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Amazonian', 'Lowland'],
            yticklabels=['Amazonian', 'Lowland'],
            ax=axes[0], cbar=False, annot_kws={'size': 16})
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('Actual', fontsize=12)
axes[0].set_title(f'Confusion Matrix\nAccuracy: {accuracy:.2%}', fontsize=14)

top_plot = top_markers_df.head(15)
labels = [str(n)[:35] for n in top_plot['Metabolite']]
axes[1].barh(range(len(top_plot)), top_plot['SHAP_Importance'], color='steelblue')
axes[1].set_yticks(range(len(top_plot)))
axes[1].set_yticklabels(labels, fontsize=9)
axes[1].set_xlabel('SHAP Importance', fontsize=12)
axes[1].set_title('Top 15 Key Markers for Origin Classification', fontsize=14)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('origin_classification_results.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: origin_classification_results.png")

# Plot 2: SHAP summary
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_use, X_log, feature_names=feature_names,
                  max_display=20, show=False)
plt.title('SHAP Summary: Top 20 Markers for Origin Classification', fontsize=14)
plt.tight_layout()
plt.savefig('shap_summary_real.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: shap_summary_real.png")

# Plot 3: RF Feature importance
plt.figure(figsize=(12, 10))
shap.summary_plot(importance.reshape(1, -1), X_log,
                  feature_names=feature_names, plot_type='bar',
                  max_display=20, show=False)
plt.title('Random Forest Feature Importance', fontsize=14)
plt.tight_layout()
plt.savefig('rf_importance_real.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: rf_importance_real.png")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE!")
print("="*70)
print(f"\n📊 Summary:")
print(f"   • Annotated metabolites: {X_log.shape[1]}")
print(f"   • Samples: {X_log.shape[0]}")
print(f"   • Classification accuracy: {accuracy:.2%}")
print(f"   • Top marker: {str(feature_names[top_idx_shap[0]])}")
print(f"\n📁 Files generated:")
print(f"   • top_20_markers.csv")
print(f"   • classification_results.csv")
print(f"   • origin_classification_results.png")
print(f"   • shap_summary_real.png")
print(f"   • rf_importance_real.png")
