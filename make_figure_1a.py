import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

print("Generating Figure 1A...")

# 1. Load and filter data
df_diff = pd.read_excel('01.Statistical Analysis/Differentially Expressed Metabolites.xlsx', sheet_name='A1vsB1vsC1vsF1vsP1')
df_sig = df_diff[df_diff['Q-VALUE'] < 0.05].copy()

phenolic_keywords = ['Shikimates', 'Phenylpropanoids', 'Flavonoids', 'Phenolic']
mask = pd.Series([False] * len(df_sig), index=df_sig.index)
for kw in phenolic_keywords:
    mask |= df_sig['Superclass'].astype(str).str.contains(kw, case=False, na=False)
    mask |= df_sig['Class'].astype(str).str.contains(kw, case=False, na=False)

df_final = df_sig[mask].drop_duplicates(subset=['compound name'], keep='first')

# 2. Define samples and origins
sample_cols = ['A1_1', 'A1_2', 'A1_3', 'B1_1', 'B1_2', 'B1_3', 
               'C1_1', 'C1_2', 'C1_3', 'F1_1', 'F1_2', 'F1_3', 
               'P1_1', 'P1_2', 'P1_3']
origin_map = {'A1': 'Lowland', 'B1': 'Lowland', 'C1': 'Amazonian', 'F1': 'Amazonian', 'P1': 'Amazonian'}
origins = [origin_map[col.split('_')[0]] for col in sample_cols]

# 3. Clean and scale
X = df_final[sample_cols].T.replace(0, np.nan)
X = X.fillna(X.median())
X_log = np.log1p(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# 4. PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# 5. Create Plot
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
colors = ['#2B6CB0' if o == 'Amazonian' else '#68D391' for o in origins]

ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=180, alpha=0.85, edgecolors='k', linewidth=1.5)

# Add sample labels
for i, (x_val, y_val) in enumerate(zip(X_pca[:, 0], X_pca[:, 1])):
    ax.annotate(sample_cols[i], (x_val, y_val), fontsize=9, ha='center', va='center', fontweight='bold')

# Formatting
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12, fontweight='bold')
ax.set_title('PCA of Guava Phenolic Chemotypes', fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')

# Legend
legend_elements = [
    Patch(facecolor='#2B6CB0', edgecolor='k', label='Amazonian'),
    Patch(facecolor='#68D391', edgecolor='k', label='Lowland')
]
ax.legend(handles=legend_elements, loc='best', fontsize=11, framealpha=1)

plt.tight_layout()
plt.savefig('Figure_1A_PCA_Plot.png', dpi=300, bbox_inches='tight')
print("✅ Figure 1A saved successfully as 'Figure_1A_PCA_Plot.png' at 300 DPI.")
