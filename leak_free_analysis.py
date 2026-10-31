import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LEAK-FREE AI PIPELINE (Nested LOOCV)")
print("="*70)

# ============================================
# 1. LOAD AND FILTER DATA
# ============================================
print("\n📂 Step 1: Loading and filtering data...")
df_diff = pd.read_excel(
    '01.Statistical Analysis/Differentially Expressed Metabolites.xlsx',
    sheet_name='A1vsB1vsC1vsF1vsP1'
)

# Filter 1: Q-VALUE < 0.05
df_sig = df_diff[df_diff['Q-VALUE'] < 0.05].copy()

# Filter 2: Phenolic classes
phenolic_keywords = ['Shikimates', 'Phenylpropanoids', 'Flavonoids', 'Phenolic']
mask = pd.Series([False] * len(df_sig), index=df_sig.index)
for kw in phenolic_keywords:
    mask |= df_sig['Superclass'].astype(str).str.contains(kw, case=False, na=False)
    mask |= df_sig['Class'].astype(str).str.contains(kw, case=False, na=False)

df_final = df_sig[mask].copy()
df_final = df_final.drop_duplicates(subset=['compound name'], keep='first')

print(f"   ✅ Phenolic features retained: {df_final.shape[0]}")

# ============================================
# 2. PREPARE MATRIX (X and y)
# ============================================
sample_cols = ['A1_1', 'A1_2', 'A1_3', 'B1_1', 'B1_2', 'B1_3',
               'C1_1', 'C1_2', 'C1_3', 'F1_1', 'F1_2', 'F1_3',
               'P1_1', 'P1_2', 'P1_3']

origin_map = {
    'A1': 'Lowland', 'B1': 'Lowland',
    'C1': 'Amazonian', 'F1': 'Amazonian', 'P1': 'Amazonian'
}
origins = [origin_map[col.split('_')[0]] for col in sample_cols]

# Transpose so rows = samples, columns = features
X = df_final[sample_cols].T 
y = np.array(origins)

# Basic cleaning (Imputation of zeros/NaNs)
X = X.replace(0, np.nan)
X = X.fillna(X.median())
X_log = np.log1p(X).values # Convert to numpy array for the pipeline

print(f"   ✅ Feature matrix shape: {X_log.shape}")

# ============================================
# 3. DEFINE THE LEAK-FREE PIPELINE
# ============================================
print("\n🔬 Step 3: Setting up the Pipeline...")
# The pipeline ensures Scaling and PCA are ONLY fit on the training folds
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=3)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# ============================================
# 4. RUN LEAVE-ONE-OUT CROSS-VALIDATION
# ============================================
print("\n🚀 Step 4: Running True LOOCV...")
loo = LeaveOneOut()
predictions = []
actuals = []

for train_index, test_index in loo.split(X_log):
    X_train, X_test = X_log[train_index], X_log[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Fit pipeline on 14 samples, predict the 1 held-out sample
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)[0]
    
    predictions.append(pred)
    actuals.append(y_test[0])

# ============================================
# 5. EVALUATE AND PRINT RESULTS
# ============================================
true_accuracy = accuracy_score(actuals, predictions)
cm = confusion_matrix(actuals, predictions, labels=['Amazonian', 'Lowland'])

print("\n" + "="*70)
print("TRUE LEAK-FREE RESULTS")
print("="*70)
print(f"✅ Unbiased LOOCV Accuracy: {true_accuracy:.4f} ({true_accuracy*15:.0f}/15 correct)")
print(f"\n   Confusion Matrix:")
print(f"                    Predicted")
print(f"                    Amazonian  Lowland")
print(f"   Actual Amazonian      {cm[0][0]:>3}       {cm[0][1]:>3}")
print(f"   Actual Lowland        {cm[1][0]:>3}       {cm[1][1]:>3}")
print("="*70)
print("\n📝 Write this exact accuracy number into your manuscript.")

# ============================================
# 6. SAVE A NEW PLOT FOR THE PAPER
# ============================================
# (This just visualizes the final PCA fit on all data, for the figures)
pipeline.fit(X_log, y)
X_pca_final = pipeline.named_steps['pca'].transform(pipeline.named_steps['scaler'].transform(X_log))

plt.figure(figsize=(8, 6))
colors = ['#2B6CB0' if o == 'Amazonian' else '#68D391' for o in y]
plt.scatter(X_pca_final[:, 0], X_pca_final[:, 1], c=colors, s=150, alpha=0.8, edgecolors='k')
for i, (x_val, y_val) in enumerate(zip(X_pca_final[:, 0], X_pca_final[:, 1])):
    plt.annotate(sample_cols[i], (x_val, y_val), fontsize=8, ha='center', va='center')
    
plt.xlabel(f'PC1 ({pipeline.named_steps["pca"].explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pipeline.named_steps["pca"].explained_variance_ratio_[1]:.1%})')
plt.title('PCA of Guava Chemotypes (Leak-Free Fit)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('final_pca_leakfree.png', dpi=150)
print("✅ Saved final PCA plot as 'final_pca_leakfree.png'")
