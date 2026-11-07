import pandas as pd
import numpy as np
import itertools
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SITE-RESTRICTED PERMUTATION TEST")
print("="*70)

# 1. Load Data (using the globally filtered 1218 features for speed, 
# as the permutation test evaluates the final model's ability to separate sites)
df_diff = pd.read_excel('01.Statistical Analysis/Differentially Expressed Metabolites.xlsx', sheet_name='A1vsB1vsC1vsF1vsP1')
df_sig = df_diff[df_diff['Q-VALUE'] < 0.05].copy()
phenolic_keywords = ['Shikimates', 'Phenylpropanoids', 'Flavonoids', 'Phenolic']
mask = pd.Series([False] * len(df_sig), index=df_sig.index)
for kw in phenolic_keywords:
    mask |= df_sig['Superclass'].astype(str).str.contains(kw, case=False, na=False)
    mask |= df_sig['Class'].astype(str).str.contains(kw, case=False, na=False)
df_final = df_sig[mask].drop_duplicates(subset=['compound name'], keep='first')

sample_cols = ['A1_1', 'A1_2', 'A1_3', 'B1_1', 'B1_2', 'B1_3', 'C1_1', 'C1_2', 'C1_3', 'F1_1', 'F1_2', 'F1_3', 'P1_1', 'P1_2', 'P1_3']
X = df_final[sample_cols].T.replace(0, np.nan).fillna(df_final[sample_cols].T.median()).values

# 2. Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=3)),
    ('clf', LogisticRegression(max_iter=1000))
])

# The 5 sites
sites = ['A1', 'B1', 'C1', 'F1', 'P1']
# The true labels
true_labels = ['Lowland', 'Lowland', 'Amazonian', 'Amazonian', 'Amazonian']

# 3. Generate all 10 possible unique site permutations (choose 2 for Lowland)
# Because there are 5 sites, there are 10 ways to pick 2 of them to be "Lowland".
permutations = list(itertools.combinations(range(5), 2))
results = []

print(f"Testing all {len(permutations)} possible site-level assignments...")

for i, lowland_indices in enumerate(permutations):
    # Create labels for this permutation
    perm_labels = []
    for site_idx in range(5):
        if site_idx in lowland_indices:
            perm_labels.append('Lowland')
        else:
            perm_labels.append('Amazonian')
    
    # Expand site labels to sample labels
    y_perm = []
    for label in perm_labels:
        y_perm.extend([label] * 3) # 3 replicates per site
    y_perm = np.array(y_perm)
    
    # We are NOT doing LOSOCV here, we are doing standard LOOCV to evaluate 
    # if this specific label assignment can be learned.
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    preds = []
    actuals = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_perm[train_idx], y_perm[test_idx]
        
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)[0]
        preds.append(pred)
        actuals.append(y_test[0])
        
    acc = accuracy_score(actuals, preds)
    
    # Check if this permutation matches the true labels
    is_true = (perm_labels == true_labels)
    results.append((perm_labels, acc, is_true))
    
    print(f"  Perm {i+1}: {perm_labels} -> Accuracy: {acc*100:.1f}% {'(TRUE LABELS)' if is_true else ''}")

# 4. Calculate p-value
# p-value = proportion of permutations that achieve accuracy >= true accuracy
true_acc = [r[1] for r in results if r[2]][0]
extreme_count = sum(1 for r in results if r[1] >= true_acc)
p_value = extreme_count / len(results)

print("\n" + "="*70)
print(f"True Model Accuracy: {true_acc*100:.1f}%")
print(f"Number of permutations achieving >= {true_acc*100:.1f}%: {extreme_count}")
print(f"Site-Restricted Permutation p-value: {p_value:.4f}")
print("="*70)
