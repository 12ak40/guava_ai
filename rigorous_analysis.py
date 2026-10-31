import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import shap
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🧬 RIGOROUS ANALYSIS: PCA + Simple Classifier")
print("   Addressing the dimensionality problem")
print("="*70)

# ============================================
# 1. LOAD DIFFERENTIAL METABOLITES (with Q-VALUE)
# ============================================
print("\n📂 Step 1: Loading Differential Metabolites")
print("-" * 40)

df_diff = pd.read_excel(
    '01.Statistical Analysis/Differentially Expressed Metabolites.xlsx',
    sheet_name='A1vsB1vsC1vsF1vsP1'
)
print(f"   Total features: {df_diff.shape[0]}")

# Filter by Q-VALUE < 0.05 (statistically significant)
df_sig = df_diff[df_diff['Q-VALUE'] < 0.05].copy()
print(f"   After Q-VALUE < 0.05: {df_sig.shape[0]} features")

# Also filter by phenolic classes
phenolic_keywords = ['Shikimates', 'Phenylpropanoids', 'Flavonoids', 'Phenolic']
mask = pd.Series([False] * len(df_sig), index=df_sig.index)
for kw in phenolic_keywords:
    mask |= df_sig['Superclass'].astype(str).str.contains(kw, case=False, na=False)
    mask |= df_sig['Class'].astype(str).str.contains(kw, case=False, na=False)

df_final = df_sig[mask].copy()
print(f"   After phenolic filter: {df_final.shape[0]} features")

# Remove duplicates
df_final = df_final.drop_duplicates(subset=['compound name'], keep='first')
print(f"   Final features: {df_final.shape[0]}")

# ============================================
# 2. PREPARE FEATURE MATRIX
# ============================================
print("\n📊 Step 2: Preparing Feature Matrix")
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

X = df_final[sample_cols].T
X.columns = [str(c) for c in df_final['compound name'].astype(str).values]
if X.columns.duplicated().any():
    X.columns = [f"{c}_{i}" if i > 0 else c for i, c in enumerate(X.columns)]

y = pd.Series(origins, index=sample_cols, name='Origin')

X = X.replace(0, np.nan)
X = X.fillna(X.median())
X_log = np.log1p(X)

print(f"   Feature matrix: {X_log.shape}")

# ============================================
# 3. PCA REDUCTION
# ============================================
print("\n🔬 Step 3: PCA Dimensionality Reduction")
print("-" * 40)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

print(f"   Original features: {X_log.shape[1]}")
print(f"   After PCA: {X_pca.shape[1]} components")
print(f"   Explained variance: {pca.explained_variance_ratio_}")
print(f"   Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# ============================================
# 4. LEAVE-ONE-OUT CLASSIFICATION ON PCA
# ============================================
print("\n📊 Step 4: Leave-One-Out Classification (PCA components)")
print("-" * 40)

model = LogisticRegression(max_iter=1000, random_state=42)
loo = LeaveOneOut()
predictions, actuals = [], []

for train_idx, test_idx in loo.split(X_pca):
    X_train, X_test = X_pca[train_idx], X_pca[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)[0]
    predictions.append(pred)
    actuals.append(y_test.values[0])

loo_accuracy = accuracy_score(actuals, predictions)
print(f"   ✅ LOO Accuracy (PCA): {loo_accuracy:.4f} ({loo_accuracy*15:.0f}/15 correct)")

# Confusion matrix
cm = confusion_matrix(actuals, predictions, labels=['Amazonian', 'Lowland'])
print(f"\n   Confusion Matrix:")
print(f"                 Predicted")
print(f"                 Amazonian  Lowland")
print(f"   Actual Amazonian    {cm[0][0]:>3}       {cm[0][1]:>3}")
print(f"   Actual Lowland      {cm[1][0]:>3}       {cm[1][1]:>3}")

# ============================================
# 5. IDENTIFY KEY MARKERS (from PCA loadings)
# ============================================
print("\n🔍 Step 5: Key Markers from PCA Loadings")
print("-" * 40)

# Get top loadings for PC1 (most variance)
pc1_loadings = pd.Series(pca.components_[0], index=X_log.columns)
top_pc1 = pc1_loadings.abs().sort_values(ascending=False).head(20)

print("\n   🏆 Top 20 Markers Driving PC1 (Origin Separation):")
for i, (name, loading) in enumerate(top_pc1.items()):
    print(f"   {i+1:<6} {str(name)[:48]:<50} {loading:.4f}")

# ============================================
# 6. VISUALIZE PCA
# ============================================
print("\n📈 Step 6: Generating Visualizations")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA scatter plot
colors = ['#2B6CB0' if o == 'Amazonian' else '#68D391' for o in y]
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=150, alpha=0.8)
for i, (x, yv) in enumerate(zip(X_pca[:, 0], X_pca[:, 1])):
    axes[0].annotate(y.index[i], (x, yv), fontsize=8, ha='center', va='center')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
axes[0].set_title('PCA: Origin Separation', fontsize=13)
axes[0].grid(True, alpha=0.3)
axes[0].legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2B6CB0', markersize=10, label='Amazonian'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#68D391', markersize=10, label='Lowland')
])

# Top markers bar chart
top_plot = top_pc1.head(15)
labels = [str(n)[:38] for n in top_plot.index]
axes[1].barh(range(len(top_plot)), top_plot.values, color='steelblue')
axes[1].set_yticks(range(len(top_plot)))
axes[1].set_yticklabels(labels, fontsize=8)
axes[1].set_xlabel('PC1 Loading', fontsize=12)
axes[1].set_title(f'Top 15 Markers Driving Origin Separation\nLOO Accuracy: {loo_accuracy:.1%}', fontsize=12)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('rigorous_pca_analysis.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: rigorous_pca_analysis.png")

print("\n" + "="*70)
print("✅ RIGOROUS ANALYSIS COMPLETE!")
print("="*70)
print(f"\n📊 Summary:")
print(f"   • Original features: {X_log.shape[1]}")
print(f"   • PCA components: {X_pca.shape[1]}")
print(f"   • Variance explained: {sum(pca.explained_variance_ratio_):.1%}")
print(f"   • LOO accuracy: {loo_accuracy:.1%} ({loo_accuracy*15:.0f}/15 correct)")
print(f"\n📁 Files generated:")
print(f"   • rigorous_pca_analysis.png")
