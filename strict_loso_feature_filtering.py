import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STRICT FOLD-SPECIFIC FEATURE FILTERING (LOSOCV)")
print("="*70)

# 1. Load the RAW data (all 7,458 features)
df_raw = pd.read_excel('01.Statistical Analysis/Mean.xlsx', sheet_name='Mean')
df_raw = df_raw.dropna(subset=['compound name']).drop_duplicates(subset=['compound name']).set_index('compound name')

# Filter to phenolic classes FIRST (this is unsupervised, so it can be done globally)
phenolic_keywords = ['Shikimates', 'Phenylpropanoids', 'Flavonoids', 'Phenolic']
mask = pd.Series([False] * len(df_raw), index=df_raw.index)
for kw in phenolic_keywords:
    mask |= df_raw['Superclass'].astype(str).str.contains(kw, case=False, na=False)
    mask |= df_raw['Class'].astype(str).str.contains(kw, case=False, na=False)
df_phenolic = df_raw[mask].copy()

print(f"Total raw metabolites: {len(df_raw)}")
print(f"Phenolic metabolites (unsupervised filter): {len(df_phenolic)}")

# Define samples
sample_cols = ['A1_1', 'A1_2', 'A1_3', 'B1_1', 'B1_2', 'B1_3', 'C1_1', 'C1_2', 'C1_3', 'F1_1', 'F1_2', 'F1_3', 'P1_1', 'P1_2', 'P1_3']
origin_map = {'A1': 'Lowland', 'B1': 'Lowland', 'C1': 'Amazonian', 'F1': 'Amazonian', 'P1': 'Amazonian'}
sites = np.array([col.split('_')[0] for col in sample_cols])
origins = np.array([origin_map[s] for s in sites])

X = df_phenolic[sample_cols].T.replace(0, np.nan).fillna(df_phenolic[sample_cols].T.median()).values
y = origins

# 2. Custom Transformer for Fold-Specific Supervised Filtering
class FoldSpecificQValueFilter(BaseEstimator, TransformerMixin):
    def __init__(self, q_threshold=0.05):
        self.q_threshold = q_threshold
        
    def fit(self, X, y):
        # X is (n_samples, n_features) for the TRAINING set
        # y is the labels for the TRAINING set
        
        # Calculate p-values for each feature using a t-test
        p_values = []
        for i in range(X.shape[1]):
            group1 = X[y == 'Amazonian', i]
            group2 = X[y == 'Lowland', i]
            # If a feature is all zero or identical, p=1
            if np.std(group1) == 0 and np.std(group2) == 0:
                p_values.append(1.0)
            else:
                stat, p = ttest_ind(group1, group2, equal_var=False)
                p_values.append(p)
                
        # Calculate Q-values (BH correction)
        _, q_values, _, _ = multipletests(p_values, alpha=self.q_threshold, method='fdr_bh')
        self.q_values_ = q_values
        
        # Find which features pass the threshold
        self.selected_indices_ = np.where(q_values < self.q_threshold)[0]
        
        # If no features pass, keep the top 10 to avoid crashing PCA
        if len(self.selected_indices_) == 0:
            self.selected_indices_ = np.argsort(q_values)[:10]
            
        print(f"  [Fold] Selected {len(self.selected_indices_)} features out of {X.shape[1]}")
        return self
        
    def transform(self, X):
        return X[:, self.selected_indices_]

# 3. Define the Pipeline
# Note: The filter is FIRST, so it only sees training data.
pipeline = Pipeline([
    ('filter', FoldSpecificQValueFilter(q_threshold=0.05)),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=3)),
    ('clf', LogisticRegression(max_iter=1000))
])

# 4. Run LOSOCV
logo = LeaveOneGroupOut()
predictions = []
actuals = []

print("\nRunning LOSOCV with fold-specific filtering...")
for train_idx, test_idx in logo.split(X, y, sites):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_site = sites[test_idx][0]
    
    # Fit the ENTIRE pipeline (including the filter!) on the training data
    pipeline.fit(X_train, y_train)
    
    # Predict the held-out site
    pred = pipeline.predict(X_test)
    predictions.extend(pred)
    actuals.extend(y_test)
    
    acc = accuracy_score(y_test, pred)
    print(f"  Site {test_site}: Accuracy = {acc*100:.1f}%")

# 5. Final Metrics
cm = confusion_matrix(actuals, predictions, labels=['Amazonian', 'Lowland'])
tn, fp, fn, tp = cm.ravel()
acc = accuracy_score(actuals, predictions)
bal_acc = balanced_accuracy_score(actuals, predictions)

print("\n" + "="*70)
print("TRUE LEAK-FREE RESULTS (Feature selection inside folds)")
print("="*70)
print(f"Accuracy: {acc*100:.1f}% ({sum(np.array(actuals)==np.array(predictions))}/15)")
print(f"Balanced Accuracy: {bal_acc*100:.1f}%")
print(f"Confusion Matrix:\n{cm}")
