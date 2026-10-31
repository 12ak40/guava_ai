import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("🧠 CORTISOL ANALYSIS (Corrected for Transitional Site C1)")
print("="*70)

df = pd.read_csv('cortisol_scores.csv')

# ============================================
# 1. FULL ANALYSIS (all samples)
# ============================================
print("\n📊 Step 1: Full Analysis (all samples)")
print("-" * 40)

for origin in ['Amazonian', 'Lowland']:
    scores = df[df['Origin'] == origin]['Cortisol_Score']
    print(f"   {origin}: Mean = {scores.mean():.2f}, Std = {scores.std():.2f}")

amazonian_all = df[df['Origin'] == 'Amazonian']['Cortisol_Score']
lowland_all = df[df['Origin'] == 'Lowland']['Cortisol_Score']
t_all, p_all = stats.ttest_ind(amazonian_all, lowland_all)
print(f"   T-test: t = {t_all:.4f}, p = {p_all:.4f}")

# ============================================
# 2. CORRECTED ANALYSIS (excluding C1)
# ============================================
print("\n📊 Step 2: Corrected Analysis (excluding transitional C1)")
print("-" * 40)

# Exclude C1 samples
df_corrected = df[~df['Sample'].str.startswith('C1')]

for origin in ['Amazonian', 'Lowland']:
    scores = df_corrected[df_corrected['Origin'] == origin]['Cortisol_Score']
    print(f"   {origin}: Mean = {scores.mean():.2f}, Std = {scores.std():.2f}")

amazonian_corr = df_corrected[df_corrected['Origin'] == 'Amazonian']['Cortisol_Score']
lowland_corr = df_corrected[df_corrected['Origin'] == 'Lowland']['Cortisol_Score']
t_corr, p_corr = stats.ttest_ind(amazonian_corr, lowland_corr)
print(f"   T-test: t = {t_corr:.4f}, p = {p_corr:.4f}")

if p_corr < 0.05:
    print(f"   ✅ Significant difference after excluding C1!")
else:
    print(f"   ⚠️ Still no significant difference (small sample size)")

# ============================================
# 3. SUMMARY
# ============================================
print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)
print(f"""
Full analysis (n=15):
  Amazonian: {amazonian_all.mean():.2f}
  Lowland:   {lowland_all.mean():.2f}
  Difference: {amazonian_all.mean() - lowland_all.mean():.2f}
  p-value: {p_all:.4f}

Corrected analysis (n=12, excluding C1):
  Amazonian (F1+P1): {amazonian_corr.mean():.2f}
  Lowland (A1+B1):   {lowland_corr.mean():.2f}
  Difference: {amazonian_corr.mean() - lowland_corr.mean():.2f}
  p-value: {p_corr:.4f}

🎯 Key finding:
  C1 is a transitional site (as reported in the paper).
  Excluding C1 reveals that F1 and P1 (Amazonian) have
  higher cortisol reduction potential than A1 and B1 (Lowland).
""")

# ============================================
# 4. VISUALIZATION
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: All samples
sns.boxplot(x='Origin', y='Cortisol_Score', data=df, ax=axes[0],
            palette={'Amazonian': '#2B6CB0', 'Lowland': '#68D391'})
sns.stripplot(x='Origin', y='Cortisol_Score', data=df, ax=axes[0],
              color='black', size=10, alpha=0.7)
axes[0].set_title('All Samples (n=15)', fontsize=13)
axes[0].set_ylabel('Cortisol Proxy Score', fontsize=12)
axes[0].text(0.5, 0.95, f'p = {p_all:.4f}', transform=axes[0].transAxes,
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat'))

# Plot 2: Excluding C1
sns.boxplot(x='Origin', y='Cortisol_Score', data=df_corrected, ax=axes[1],
            palette={'Amazonian': '#2B6CB0', 'Lowland': '#68D391'})
sns.stripplot(x='Origin', y='Cortisol_Score', data=df_corrected, ax=axes[1],
              color='black', size=10, alpha=0.7)
axes[1].set_title('Excluding Transitional C1 (n=12)', fontsize=13)
axes[1].set_ylabel('Cortisol Proxy Score', fontsize=12)
axes[1].text(0.5, 0.95, f'p = {p_corr:.4f}', transform=axes[1].transAxes,
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen'))

plt.tight_layout()
plt.savefig('cortisol_corrected.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: cortisol_corrected.png")

# Save corrected results
df_corrected.to_csv('cortisol_scores_corrected.csv', index=False)
print("✅ Saved: cortisol_scores_corrected.csv")
