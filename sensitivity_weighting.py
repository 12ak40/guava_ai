import pandas as pd
import numpy as np
from scipy.stats import pearsonr

print("="*60)
print("WEIGHTING SENSITIVITY ANALYSIS")
print("="*60)

# Load data
df = pd.read_excel('01.Statistical Analysis/Mean.xlsx', sheet_name='Mean')
df['compound_lower'] = df['compound name'].astype(str).str.lower()

sample_cols = ['A1_1', 'A1_2', 'A1_3', 'B1_1', 'B1_2', 'B1_3',
               'C1_1', 'C1_2', 'C1_3', 'F1_1', 'F1_2', 'F1_3',
               'P1_1', 'P1_2', 'P1_3']

# The 20 markers and their weights
markers_weights = {
    'Norwogonin': 1.0, 'Jaceidin': 1.0, 'Luteolin': 0.9, 'Quercitrin': 0.8, 
    'Myricitrin': 0.8, 'Ellagic acid': 0.8, 'Apigenin': 0.7, 'Eriodictyol': 0.7, 
    'Isorhamnetin': 0.7, 'Kaempferol': 0.7, 'Gallic acid': 0.7, 'Chlorogenic acid': 0.7, 
    'Caffeic acid': 0.7, 'Catechin': 0.6, 'Epicatechin': 0.6, 'Rutin': 0.6, 
    'Hyperin': 0.6, 'Isoquercitrin': 0.6, 'Avicularin': 0.6, 'Guaijaverin': 0.6
}

# Extract data for detected markers
marker_data = {}
for marker in markers_weights.keys():
    mask = df['compound_lower'].str.contains(marker.lower(), na=False)
    if mask.any():
        marker_data[marker] = df[mask].iloc[0][sample_cols].values.astype(float)

# Calculate Weighted Score
weighted_score = np.zeros(len(sample_cols))
total_weight = 0
for marker, vals in marker_data.items():
    w = markers_weights[marker]
    vals_norm = (vals - vals.min()) / (vals.max() - vals.min()) if vals.max() > vals.min() else np.zeros_like(vals)
    weighted_score += vals_norm * w
    total_weight += w
weighted_score = weighted_score / total_weight

# Calculate Unweighted Score (all weights = 1.0)
unweighted_score = np.zeros(len(sample_cols))
total_unweighted = 0
for marker, vals in marker_data.items():
    vals_norm = (vals - vals.min()) / (vals.max() - vals.min()) if vals.max() > vals.min() else np.zeros_like(vals)
    unweighted_score += vals_norm
    total_unweighted += 1
unweighted_score = unweighted_score / total_unweighted

# Calculate Pearson Correlation
r_value, p_value = pearsonr(weighted_score, unweighted_score)

print(f"Number of markers detected: {len(marker_data)}")
print(f"Pearson Correlation (Weighted vs Unweighted): r = {r_value:.4f}")
print(f"P-value: {p_value:.4e}")
print("="*60)
print("Interpretation:")
if r_value > 0.95:
    print("✅ The correlation is almost perfect (r > 0.95).")
    print("This proves the result is driven by the biology (metabolite abundance),")
    print("NOT by the manual weights we assigned.")
else:
    print("⚠️ Correlation is lower than 0.95. The weights do influence the result.")
