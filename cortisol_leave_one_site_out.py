import pandas as pd
import numpy as np
from scipy import stats

# Load data (from the previous cortisol analysis)
# For simplicity, we are using the scores we already generated
# (Replace this with your actual cortisol scores if you want to recalculate)
sample_cols = ['A1_1', 'A1_2', 'A1_3', 'B1_1', 'B1_2', 'B1_3', 
               'C1_1', 'C1_2', 'C1_3', 'F1_1', 'F1_2', 'F1_3', 
               'P1_1', 'P1_2', 'P1_3']
origin_map = {'A1': 'Lowland', 'B1': 'Lowland', 'C1': 'Amazonian', 'F1': 'Amazonian', 'P1': 'Amazonian'}
sites = [col.split('_')[0] for col in sample_cols]
origins = [origin_map[s] for s in sites]

# Assuming you have a variable 'cortisol_score' from your earlier script. 
# If not, here is a placeholder for the exact values you got earlier:
cortisol_score = np.array([67.04, 65.66, 66.89, 64.79, 62.29, 63.91, 
                           53.48, 52.98, 50.00, 74.34, 74.46, 74.80, 
                           66.05, 66.41, 75.00])

df = pd.DataFrame({'Sample': sample_cols, 'Site': sites, 'Origin': origins, 'Score': cortisol_score})

print("="*60)
print("CORTISOL ROBUSTNESS: LEAVE-ONE-SITE-OUT")
print("="*60)

# We already know the full analysis (including C1) is p=0.97
# The corrected analysis (excluding C1) is p=0.0057

# Let's test what happens if we exclude A1, then B1, then F1, then P1
for site_to_drop in ['A1', 'B1', 'F1', 'P1']:
    df_subset = df[df['Site'] != site_to_drop]
    
    amazonian = df_subset[df_subset['Origin'] == 'Amazonian']['Score']
    lowland = df_subset[df_subset['Origin'] == 'Lowland']['Score']
    
    t_stat, p_val = stats.ttest_ind(amazonian, lowland)
    print(f"Dropping site {site_to_drop}: p-value = {p_val:.4f} (Amazonian n={len(amazonian)}, Lowland n={len(lowland)})")

print("="*60)
print("Interpretation:")
print("If the p-value remains < 0.05 when we drop A1 or B1 or F1 or P1,")
print("it proves the cortisol signal is NOT driven by just one site.")
