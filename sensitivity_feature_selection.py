import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SENSITIVITY ANALYSIS: ALL PHENOLIC FEATURES (NO SUPERVISED FILTERING)")
print("="*70)

# Load raw data and filter to phenolic classes only
df_raw = pd.read_excel('01.Statistical Analysis/Mean.xlsx', sheet_name='Mean')
df_raw = df_raw.dropna(subset=['compound name']).drop_duplicates(subset=['compound name']).set_index('compound name')

phenolic_keywords = ['Shikimates', 'Phenylpropanoids', 'Flavonoids', 'Phenolic']
mask = pd.Series([False] * len(df_raw), index=df_raw.index)
for kw in phenolic_keywords:
    mask |= df_raw['Superclass'].astype(str).str.contains(kw, case=False, na=False)
    mask |= df_raw['Class'].astype(str).str.contains(kw, case=False, na=False)
df_phenolic = df_raw[mask].copy()

sample_cols = ['A1_1', 'A1_2', 'A1_3', 'B1_1', 'B1_2', 'B1_3', 'C1_1', 'C1_2', 'C1_3', 'F1_1', 'F1_2', 'F1_3', 'P1_1', 'P1_2', 'P1_3']
origin_map = {'A1': 'Lowland', 'B1': 'Lowland', 'C1': 'Amazonian', 'F1': 'Amazonian', 'P1': 'Amazonian'}
sites = np.array([col.split('_')[0] for col in sample_cols])
origins = np.array([origin_map[s] for s in sites])

X = df_phenolic[sample_cols].T.replace(0, np.nan).fillna(df_phenolic[sample_cols].T.median()).values
y = origins

# Pipeline WITHOUT supervised filtering
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=3)),
    ('clf', LogisticRegression(max_iter=1000))
])

logo = LeaveOneGroupOut()
predictions = []
actuals = []

for train_idx, test_idx in logo.split(X, y, sites):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_site = sites[test_idx][0]
    
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    predictions.extend(pred)
    actuals.extend(y_test)
    
    acc = accuracy_score(y_test, pred)
    print(f"  Site {test_site}: Accuracy = {acc*100:.1f}%")

acc = accuracy_score(actuals, predictions)
bal_acc = balanced_accuracy_score(actuals, predictions)

print("\n" + "="*70)
print("RESULTS WITHOUT SUPERVISED FILTERING (All 1,372 phenolics)")
print("="*70)
print(f"Accuracy: {acc*100:.1f}% ({sum(np.array(actuals)==np.array(predictions))}/15)")
print(f"Balanced Accuracy: {bal_acc*100:.1f}%")
