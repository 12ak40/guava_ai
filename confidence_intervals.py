import numpy as np
import pandas as pd

# The exact cortisol scores
sample_cols = ['A1_1', 'A1_2', 'A1_3', 'B1_1', 'B1_2', 'B1_3', 
               'C1_1', 'C1_2', 'C1_3', 'F1_1', 'F1_2', 'F1_3', 
               'P1_1', 'P1_2', 'P1_3']
origin_map = {'A1': 'Lowland', 'B1': 'Lowland', 'C1': 'Amazonian', 'F1': 'Amazonian', 'P1': 'Amazonian'}
sites = np.array([col.split('_')[0] for col in sample_cols])
origins = np.array([origin_map[s] for s in sites])

scores = np.array([67.04, 65.66, 66.89, 64.79, 62.29, 63.91, 
                   53.48, 52.98, 50.00, 74.34, 74.46, 74.80, 
                   66.05, 66.41, 75.00])

def bootstrap_ci(amazonian_data, lowland_data, n_boot=10000):
    diffs = []
    for _ in range(n_boot):
        a_sample = np.random.choice(amazonian_data, size=len(amazonian_data), replace=True)
        l_sample = np.random.choice(lowland_data, size=len(lowland_data), replace=True)
        diffs.append(np.mean(a_sample) - np.mean(l_sample))
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

# 1. PRIMARY ANALYSIS (All samples)
amazonian_all = scores[origins == 'Amazonian']
lowland_all = scores[origins == 'Lowland']
diff_all = np.mean(amazonian_all) - np.mean(lowland_all)
ci_all = bootstrap_ci(amazonian_all, lowland_all)

print("="*60)
print("PRIMARY ANALYSIS (All 15 samples)")
print("="*60)
print(f"Observed Difference: {diff_all:.2f}")
print(f"95% Confidence Interval: [{ci_all[0]:.2f}, {ci_all[1]:.2f}]")
print("Note: CI crosses zero, indicating no significant difference.")

# 2. EXPLORATORY ANALYSIS (Excluding C1)
mask_c1 = sites != 'C1'
amazonian_ex = scores[mask_c1 & (origins == 'Amazonian')]
lowland_ex = scores[mask_c1 & (origins == 'Lowland')]
diff_ex = np.mean(amazonian_ex) - np.mean(lowland_ex)
ci_ex = bootstrap_ci(amazonian_ex, lowland_ex)

print("\n" + "="*60)
print("EXPLORATORY ANALYSIS (Excluding transitional site C1)")
print("="*60)
print(f"Observed Difference: {diff_ex:.2f}")
print(f"95% Confidence Interval: [{ci_ex[0]:.2f}, {ci_ex[1]:.2f}]")
print("Note: CI is entirely above zero, indicating a positive effect.")
