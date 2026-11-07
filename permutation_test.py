import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# The exact cortisol scores you generated
sample_cols = ['A1_1', 'A1_2', 'A1_3', 'B1_1', 'B1_2', 'B1_3', 
               'C1_1', 'C1_2', 'C1_3', 'F1_1', 'F1_2', 'F1_3', 
               'P1_1', 'P1_2', 'P1_3']
origin_map = {'A1': 'Lowland', 'B1': 'Lowland', 'C1': 'Amazonian', 'F1': 'Amazonian', 'P1': 'Amazonian'}
sites = [col.split('_')[0] for col in sample_cols]
origins = np.array([origin_map[s] for s in sites])

cortisol_score = np.array([67.04, 65.66, 66.89, 64.79, 62.29, 63.91, 
                           53.48, 52.98, 50.00, 74.34, 74.46, 74.80, 
                           66.05, 66.41, 75.00])

# Exclude C1 (transitional site)
mask = np.array([s != 'C1' for s in sites])
scores_filtered = cortisol_score[mask]
origins_filtered = origins[mask]

# Observed difference
amazonian_scores = scores_filtered[origins_filtered == 'Amazonian']
lowland_scores = scores_filtered[origins_filtered == 'Lowland']
observed_diff = np.mean(amazonian_scores) - np.mean(lowland_scores)

print("="*60)
print("PERMUTATION TEST (10,000 iterations)")
print("="*60)
print(f"Observed difference (Amazonian - Lowland): {observed_diff:.2f}")

# Permutation test
np.random.seed(42)
n_permutations = 10000
count_extreme = 0

for _ in range(n_permutations):
    # Shuffle the labels randomly
    shuffled_origins = np.random.permutation(origins_filtered)
    
    # Calculate mean difference for this random shuffle
    shuf_amazonian = scores_filtered[shuffled_origins == 'Amazonian']
    shuf_lowland = scores_filtered[shuffled_origins == 'Lowland']
    shuf_diff = np.mean(shuf_amazonian) - np.mean(shuf_lowland)
    
    # Count how many times the random difference is larger than observed
    if abs(shuf_diff) >= abs(observed_diff):
        count_extreme += 1

p_value_perm = count_extreme / n_permutations

print(f"Permutation p-value: {p_value_perm:.4f}")
print("="*60)
print("\n💡 Note: A permutation p-value < 0.05 is highly robust.")
