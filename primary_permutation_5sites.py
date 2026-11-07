import numpy as np
import itertools

# Site-level mean cortisol scores
site_data = {
    'A1': {'Origin': 'Lowland', 'Score': 66.53},
    'B1': {'Origin': 'Lowland', 'Score': 63.66},
    'C1': {'Origin': 'Amazonian', 'Score': 51.51}, # Mean of 50.0, 53.48, 52.98
    'F1': {'Origin': 'Amazonian', 'Score': 74.53},
    'P1': {'Origin': 'Amazonian', 'Score': 69.15}
}

scores = np.array([66.53, 63.66, 51.51, 74.53, 69.15])
true_labels = ['Lowland', 'Lowland', 'Amazonian', 'Amazonian', 'Amazonian']

# Observed difference (Amazonian - Lowland)
true_diff = np.mean(scores[[2,3,4]]) - np.mean(scores[[0,1]])
print(f"Observed difference (5-site primary analysis): {true_diff:.2f}")

# Enumerate all 10 possible assignments (2 Lowland, 3 Amazonian)
permutations = list(itertools.combinations(range(5), 2))
extreme_count = 0

for combo in permutations:
    lowland_idx = list(combo)
    amazonian_idx = [i for i in range(5) if i not in lowland_idx]
    perm_diff = np.mean(scores[amazonian_idx]) - np.mean(scores[lowland_idx])
    
    if abs(perm_diff) >= abs(true_diff):
        extreme_count += 1

print(f"Number of permutations with extreme diff: {extreme_count} / {len(permutations)}")
print(f"Exact site-restricted p-value (5-site primary): {extreme_count/len(permutations):.4f}")
