import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

# Load the data
sample_cols = ['A1_1', 'A1_2', 'A1_3', 'B1_1', 'B1_2', 'B1_3', 
               'C1_1', 'C1_2', 'C1_3', 'F1_1', 'F1_2', 'F1_3', 
               'P1_1', 'P1_2', 'P1_3']
origin_map = {'A1': 'Lowland', 'B1': 'Lowland', 'C1': 'Amazonian', 'F1': 'Amazonian', 'P1': 'Amazonian'}
sites = [col.split('_')[0] for col in sample_cols]
origins = [origin_map[s] for s in sites]

# The exact cortisol scores from your previous run
cortisol_score = np.array([67.04, 65.66, 66.89, 64.79, 62.29, 63.91, 
                           53.48, 52.98, 50.00, 74.34, 74.46, 74.80, 
                           66.05, 66.41, 75.00])

df = pd.DataFrame({'Sample': sample_cols, 'Site': sites, 'Origin': origins, 'Score': cortisol_score})

# Exclude the transitional site C1 for the main comparison
df_filtered = df[df['Site'] != 'C1']

# AGREGATE TO SITE LEVEL (True independent unit)
site_means = df_filtered.groupby(['Site', 'Origin'])['Score'].mean().reset_index()

print("="*60)
print("SITE-LEVEL CORTISOL SCORES (n=4 independent sites)")
print("="*60)
print(site_means.to_string(index=False))
print("="*60)

# Extract the site-level scores
amazonian_sites = site_means[site_means['Origin'] == 'Amazonian']['Score'].values
lowland_sites = site_means[site_means['Origin'] == 'Lowland']['Score'].values

print(f"\nLowland Sites (A1, B1): {lowland_sites}")
print(f"Amazonian Sites (F1, P1): {amazonian_sites}")
print(f"Mean difference: {np.mean(amazonian_sites) - np.mean(lowland_sites):.2f}")

# Run a non-parametric test (Mann-Whitney U) for small samples
stat, p_val = mannwhitneyu(amazonian_sites, lowland_sites, alternative='greater')
print(f"\nMann-Whitney U test (Site-level): p = {p_val:.4f}")
print("="*60)
print("Note: Because we only have 2 sites per group (n=4 total),")
print("traditional p-values are limited, but the effect size is clear.")
