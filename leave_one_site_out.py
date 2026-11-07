import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STRICT LEAVE-ONE-SITE-OUT CROSS-VALIDATION (LOSOCV)")
print("="*70)

# 1. Load and filter data
df_diff = pd.read_excel('01.Statistical Analysis/Differentially Expressed Metabolites.xlsx', sheet_name='A1vsB1vsC1vsF1vsP1')
df_sig = df_diff[df_diff['Q-VALUE'] < 0.05].copy()

phenolic_keywords = ['Shikimates', 'Phenylpropanoids', 'Flavonoids', 'Phenolic']
mask = pd.Series([False] * len(df_sig), index=df_sig.index)
for kw in phenolic_keywords:
    mask |= df_sig['Superclass'].astype(str).str.contains(kw, case=False, na=False)
    mask |= df_sig['Class'].astype(str).str.contains(kw, case=False, na=False)

df_final = df_sig[mask].drop_duplicates(subset=['compound name'], keep='first')

# 2. Prepare matrix (samples as rows)
sample_cols = ['A1_1', 'A1_2', 'A1_3', 'B1_1', 'B1_2', 'B1_3', 
               'C1_1', 'C1_2', 'C1_3', 'F1_1', 'F1_2', 'F1_3', 
               'P1_1', 'P1_2', 'P1_3']

# Extract the site identifier (e.g., 'A1', 'B1') for grouping
sites = [col.split('_')[0] for col in sample_cols]
origin_map = {'A1': 'Lowland', 'B1': 'Lowland', 'C1': 'Amazonian', 'F1': 'Amazonian', 'P1': 'Amazonian'}
y = np.array([origin_map[site] for site in sites])
groups = np.array(sites)

X = df_final[sample_cols].T.replace(0, np.nan)
X = X.fillna(X.median())
X_log = np.log1p(X).values

# 3. Define strict Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=3)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# 4. Run Leave-One-Site-Out Cross-Validation
logo = LeaveOneGroupOut()
predictions = []
actuals = []
tested_sites = []

print("\n🚀 Running LOSOCV (testing on completely unseen sites)...")
for train_idx, test_idx in logo.split(X_log, y, groups):
    X_train, X_test = X_log[train_idx], X_log[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    group_test = groups[test_idx][0] # Which site is being tested
    
    # Fit on 4 sites, predict the 5th
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    
    # Calculate accuracy for this specific held-out site
    site_acc = accuracy_score(y_test, pred)
    print(f"   Held-out Site {group_test}: Accuracy = {site_acc*100:.1f}% ({sum(y_test == pred)}/3 correct)")
    
    predictions.extend(pred)
    actuals.extend(y_test)
    tested_sites.append(group_test)

# 5. Evaluate overall results
losocv_accuracy = accuracy_score(actuals, predictions)
cm = confusion_matrix(actuals, predictions, labels=['Amazonian', 'Lowland'])

print("\n" + "="*70)
print("TRUE LEAVE-ONE-SITE-OUT RESULTS")
print("="*70)
print(f"✅ Unbiased LOSOCV Accuracy: {losocv_accuracy*100:.1f}% ({sum(np.array(actuals) == np.array(predictions))}/15 correct)")
print(f"\n   Confusion Matrix:")
print(f"                    Predicted")
print(f"                    Amazonian  Lowland")
print(f"   Actual Amazonian      {cm[0][0]:>3}       {cm[0][1]:>3}")
print(f"   Actual Lowland        {cm[1][0]:>3}       {cm[1][1]:>3}")
print("="*70)
