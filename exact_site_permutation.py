import numpy as np
import pandas as pd
import itertools
from scipy import stats

print("="*60)
print("EXACT SITE-RESTRICTED PERMUTATION TEST")
print("="*60)

# Site-level mean cortisol scores (from your previous analysis)
site_data = {
    'A1': {'Origin': 'Lowland', 'Score': 66.53},
    'B1': {'Origin': 'Lowland', 'Score': 63.66},
    'F1': {'Origin': 'Amazonian', 'Score': 74.53},
    'P1': {'Origin': 'Amazonian', 'Score': 69.15}
}

scores = np.array([66.53, 63.66, 74.53, 69.15])
# The actual labels
actual_labels = ['Lowland', 'Lowland', 'Amazonian', 'Amazonian']

# Observed difference
actual_diff = np.mean(scores[2:]) - np.mean(scores[:2])
print(f"Observed difference (Amazonian - Lowland): {actual_diff:.2f}")

# Exact permutation (because we have 4 sites, there are only 6 unique combinations)
# We shuffle the labels: how many ways can we pick 2 sites for Lowland out of 4?
combinations = list(itertools.combinations(range(4), 2))
extreme_count = 0
total_perms = len(combinations)

print(f"\nTesting all {total_perms} possible site permutations:")

for combo in combinations:
    # Assign these indices to Lowland, the rest to Amazonian
    lowland_idx = list(combo)
    amazonian_idx = [i for i in range(4) if i not in lowland_idx]
    
    perm_diff = np.mean(scores[amazonian_idx]) - np.mean(scores[lowland_idx])
    
    # We are doing a two-tailed test
    if abs(perm_diff) >= abs(actual_diff):
        extreme_count += 1
        print(f"  Permutation {lowland_idx} vs {amazonian_idx}: Diff = {perm_diff:.2f} (Extreme)")
    else:
        print(f"  Permutation {lowland_idx} vs {amazonian_idx}: Diff = {perm_diff:.2f}")

# Exact p-value
exact_p = extreme_count / total_perms
print("\n" + "="*60)
print(f"EXACT SITE-LEVEL P-VALUE: {exact_p:.4f}")
print("="*60)

if exact_p < 0.05:
    print("✅ Statistically significant, even at the strict site level.")
else:
    print("⚠️ Not significant at the strict site level (which is expected with n=2 sites per group).")
    print("This should be reported as a descriptive effect size, not a definitive p-value.")
