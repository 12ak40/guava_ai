import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ADVANCED ML COMPARISON (Nested LOSOCV)")
print("="*70)

# 1. Load Data
df_diff = pd.read_excel('01.Statistical Analysis/Differentially Expressed Metabolites.xlsx', sheet_name='A1vsB1vsC1vsF1vsP1')
df_sig = df_diff[df_diff['Q-VALUE'] < 0.05].copy()
phenolic_keywords = ['Shikimates', 'Phenylpropanoids', 'Flavonoids', 'Phenolic']
mask = pd.Series([False] * len(df_sig), index=df_sig.index)
for kw in phenolic_keywords:
    mask |= df_sig['Superclass'].astype(str).str.contains(kw, case=False, na=False)
    mask |= df_sig['Class'].astype(str).str.contains(kw, case=False, na=False)
df_final = df_sig[mask].drop_duplicates(subset=['compound name'], keep='first')

sample_cols = ['A1_1', 'A1_2', 'A1_3', 'B1_1', 'B1_2', 'B1_3', 'C1_1', 'C1_2', 'C1_3', 'F1_1', 'F1_2', 'F1_3', 'P1_1', 'P1_2', 'P1_3']
origin_map = {'A1': 'Lowland', 'B1': 'Lowland', 'C1': 'Amazonian', 'F1': 'Amazonian', 'P1': 'Amazonian'}
sites = np.array([col.split('_')[0] for col in sample_cols])
origins = np.array([origin_map[s] for s in sites])

X = df_final[sample_cols].T.replace(0, np.nan).fillna(df_final[sample_cols].T.median()).values
y = origins

# 2. Define Nested CV Setup
logo = LeaveOneGroupOut()

# We will test 3 models
models = {
    'Logistic (Unpenalized)': {
        'pipeline': Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=3)), ('clf', LogisticRegression(penalty=None, max_iter=1000))]),
        'params': {}
    },
    'Logistic (L1 Penalized)': {
        'pipeline': Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=3)), ('clf', LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000))]),
        'params': {'clf__C': [0.01, 0.1, 1.0, 10.0]}
    },
    'Random Forest (Non-linear)': {
        'pipeline': Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=3)), ('clf', RandomForestClassifier(random_state=42))]),
        'params': {'clf__n_estimators': [10, 50], 'clf__max_depth': [2, 5]}
    }
}

results = {}

# 3. Run Nested LOSOCV for each model
for name, config in models.items():
    print(f"\nRunning {name}...")
    predictions = []
    actuals = []
    
    for train_idx, test_idx in logo.split(X, y, sites):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inner CV for hyperparameter tuning
        if config['params']:
            grid = GridSearchCV(config['pipeline'], config['params'], cv=3, scoring='accuracy')
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        else:
            best_model = config['pipeline']
            best_model.fit(X_train, y_train)
            
        pred = best_model.predict(X_test)
        predictions.extend(pred)
        actuals.extend(y_test)

    # 4. Calculate Metrics
    cm = confusion_matrix(actuals, predictions, labels=['Amazonian', 'Lowland'])
    tn, fp, fn, tp = cm.ravel()
    
    # Note: Treating Amazonian as Positive class
    sensitivity = recall_score(actuals, predictions, pos_label='Amazonian')
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    bal_acc = balanced_accuracy_score(actuals, predictions)
    acc = accuracy_score(actuals, predictions)
    
    results[name] = {
        'Accuracy': acc,
        'Balanced Acc': bal_acc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Confusion Matrix': cm
    }

# 5. Print Results
print("\n" + "="*70)
print("FINAL NESTED LOSOCV RESULTS")
print("="*70)
print(f"{'Model':<25} | {'Accuracy':<9} | {'Bal. Acc':<9} | {'Sens.':<6} | {'Spec.':<6}")
print("-" * 70)
for name, res in results.items():
    print(f"{name:<25} | {res['Accuracy']*100:>7.1f}% | {res['Balanced Acc']*100:>7.1f}% | {res['Sensitivity']*100:>4.0f}% | {res['Specificity']*100:>4.0f}%")

print("\nConfusion Matrices (Rows=True, Cols=Pred):")
for name, res in results.items():
    print(f"\n{name}:")
    print(f"  Amazonian {res['Confusion Matrix'][0]}")
    print(f"  Lowland   {res['Confusion Matrix'][1]}")
