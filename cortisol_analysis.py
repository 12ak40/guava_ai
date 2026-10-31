import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🧠 CORTISOL REDUCTION POTENTIAL ANALYSIS")
print("   Pathway-Based Proxy Scoring from Guava Phenolics")
print("="*70)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n📂 Step 1: Loading Data")
print("-" * 40)

df = pd.read_excel('01.Statistical Analysis/Mean.xlsx', sheet_name='Mean')
print(f"   ✅ Loaded: {df.shape[0]} metabolites × {df.shape[1]} columns")

# ============================================
# 2. DEFINE CORTISOL-RELATED PATHWAY MARKERS
# ============================================
print("\n🧬 Step 2: Defining Cortisol-Related Pathway Markers")
print("-" * 40)

# Metabolites from the paper targeting AGE-RAGE, PI3K/Akt, TNF, MAPK14, RELA
# These are the key stress/inflammation-related compounds
cortisol_markers = {
    # Strong evidence (from paper's docking results)
    'Norwogonin': {'weight': 1.0, 'pathway': 'AGE-RAGE/PI3K/Akt/TNF'},
    'Jaceidin': {'weight': 1.0, 'pathway': 'MAPK14/RELA'},
    'Quercitrin': {'weight': 0.8, 'pathway': 'AGE-RAGE'},
    'Myricitrin': {'weight': 0.8, 'pathway': 'AGE-RAGE'},
    'Luteolin': {'weight': 0.9, 'pathway': 'TNF/IL-6'},
    'Apigenin': {'weight': 0.7, 'pathway': 'AGE-RAGE'},
    'Catechin': {'weight': 0.6, 'pathway': 'Antioxidant'},
    'Epicatechin': {'weight': 0.6, 'pathway': 'Antioxidant'},
    'Eriodictyol': {'weight': 0.7, 'pathway': 'Antioxidant'},
    'Isorhamnetin': {'weight': 0.7, 'pathway': 'AGE-RAGE'},
    'Kaempferol': {'weight': 0.7, 'pathway': 'AGE-RAGE'},
    'Rutin': {'weight': 0.6, 'pathway': 'Antioxidant'},
    'Hyperin': {'weight': 0.6, 'pathway': 'Antioxidant'},
    'Isoquercitrin': {'weight': 0.6, 'pathway': 'Antioxidant'},
    'Avicularin': {'weight': 0.6, 'pathway': 'Antioxidant'},
    'Guaijaverin': {'weight': 0.6, 'pathway': 'Antioxidant'},
    'Ellagic acid': {'weight': 0.8, 'pathway': 'AGE-RAGE'},
    'Gallic acid': {'weight': 0.7, 'pathway': 'AGE-RAGE'},
    'Chlorogenic acid': {'weight': 0.7, 'pathway': 'AGE-RAGE'},
    'Caffeic acid': {'weight': 0.7, 'pathway': 'AGE-RAGE'},
}

# Find which markers are in the dataset
df['compound_lower'] = df['compound name'].astype(str).str.lower()
found_markers = []
for marker in cortisol_markers:
    mask = df['compound_lower'].str.contains(marker.lower(), na=False)
    if mask.any():
        found_markers.append(marker)

print(f"   Cortisol markers defined: {len(cortisol_markers)}")
print(f"   Found in dataset: {len(found_markers)}")
print(f"   Markers: {found_markers[:10]}...")

# ============================================
# 3. CREATE CORTISOL PROXY SCORE
# ============================================
print("\n📊 Step 3: Creating Cortisol Reduction Proxy Score")
print("-" * 40)

# Sample columns
sample_cols = ['A1_1', 'A1_2', 'A1_3',
               'B1_1', 'B1_2', 'B1_3',
               'C1_1', 'C1_2', 'C1_3',
               'F1_1', 'F1_2', 'F1_3',
               'P1_1', 'P1_2', 'P1_3']

origin_map = {
    'A1': 'Lowland', 'B1': 'Lowland',
    'C1': 'Amazonian', 'F1': 'Amazonian', 'P1': 'Amazonian'
}
origins = [origin_map[col.split('_')[0]] for col in sample_cols]

# Extract cortisol marker data
marker_data = {}
for marker in found_markers:
    mask = df['compound_lower'].str.contains(marker.lower(), na=False)
    marker_rows = df[mask]
    if len(marker_rows) > 0:
        # Take the first match
        values = marker_rows.iloc[0][sample_cols].values.astype(float)
        weight = cortisol_markers[marker]['weight']
        marker_data[marker] = {'values': values, 'weight': weight}

# Create weighted score (normalized)
cortisol_score = np.zeros(len(sample_cols))
total_weight = 0

for marker, data in marker_data.items():
    values = data['values']
    weight = data['weight']
    # Normalize values to 0-1 range
    if values.max() > values.min():
        values_norm = (values - values.min()) / (values.max() - values.min())
    else:
        values_norm = np.zeros_like(values)
    cortisol_score += values_norm * weight
    total_weight += weight

if total_weight > 0:
    cortisol_score = cortisol_score / total_weight

# Scale to 0-100
cortisol_score = 50 + 25 * (cortisol_score - cortisol_score.min()) / (cortisol_score.max() - cortisol_score.min() + 1e-10)

print(f"   Score range: {cortisol_score.min():.2f} - {cortisol_score.max():.2f}")

# ============================================
# 4. COMPARE CHEMOTYPES
# ============================================
print("\n📊 Step 4: Comparing Chemotypes")
print("-" * 40)

results_df = pd.DataFrame({
    'Sample': sample_cols,
    'Origin': origins,
    'Cortisol_Score': cortisol_score
})

# Calculate statistics by origin
for origin in ['Amazonian', 'Lowland']:
    scores = results_df[results_df['Origin'] == origin]['Cortisol_Score']
    print(f"   {origin}: Mean = {scores.mean():.2f}, Std = {scores.std():.2f}")

# T-test
amazonian_scores = results_df[results_df['Origin'] == 'Amazonian']['Cortisol_Score']
lowland_scores = results_df[results_df['Origin'] == 'Lowland']['Cortisol_Score']
t_stat, p_value = stats.ttest_ind(amazonian_scores, lowland_scores)
print(f"\n   T-test: t = {t_stat:.4f}, p = {p_value:.4f}")

if p_value < 0.05:
    print(f"   ✅ Significant difference between chemotypes!")
else:
    print(f"   ⚠️ No significant difference (p > 0.05)")

# ============================================
# 5. KEY MARKERS DRIVING CORTISOL SCORE
# ============================================
print("\n🔍 Step 5: Key Markers Driving Cortisol Score")
print("-" * 40)

# Calculate contribution of each marker
contributions = {}
for marker, data in marker_data.items():
    values = data['values']
    if values.max() > values.min():
        values_norm = (values - values.min()) / (values.max() - values.min())
    else:
        values_norm = np.zeros_like(values)
    contributions[marker] = values_norm.mean() * data['weight']

contrib_df = pd.DataFrame({
    'Metabolite': list(contributions.keys()),
    'Contribution': list(contributions.values())
}).sort_values('Contribution', ascending=False)

print("\n   🏆 Top 10 Markers Contributing to Cortisol Score:")
for i, row in contrib_df.head(10).iterrows():
    print(f"   {row.name+1:<4} {row['Metabolite']:<30} {row['Contribution']:.4f}")

# ============================================
# 6. VISUALIZATIONS
# ============================================
print("\n📈 Step 6: Generating Visualizations")
print("-" * 40)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Boxplot by origin
sns.boxplot(x='Origin', y='Cortisol_Score', data=results_df, ax=axes[0],
            palette={'Amazonian': '#2B6CB0', 'Lowland': '#68D391'})
sns.stripplot(x='Origin', y='Cortisol_Score', data=results_df, ax=axes[0],
              color='black', size=8, alpha=0.6)
axes[0].set_title('Cortisol Reduction Potential by Origin', fontsize=13)
axes[0].set_ylabel('Cortisol Proxy Score', fontsize=12)
axes[0].set_xlabel('')
axes[0].grid(True, alpha=0.3)

# Add p-value
axes[0].text(0.5, 0.95, f'p = {p_value:.4f}',
             transform=axes[0].transAxes, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Bar chart of contributions
top_contrib = contrib_df.head(10)
axes[1].barh(range(len(top_contrib)), top_contrib['Contribution'].values,
             color='#2B6CB0')
axes[1].set_yticks(range(len(top_contrib)))
axes[1].set_yticklabels(top_contrib['Metabolite'].values, fontsize=9)
axes[1].set_xlabel('Contribution to Cortisol Score', fontsize=12)
axes[1].set_title('Top 10 Markers Driving Cortisol Reduction', fontsize=13)
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3)

# Plot 3: Sample-level scores
colors = ['#2B6CB0' if o == 'Amazonian' else '#68D391' for o in results_df['Origin']]
axes[2].bar(range(len(results_df)), results_df['Cortisol_Score'], color=colors)
axes[2].set_xticks(range(len(results_df)))
axes[2].set_xticklabels(results_df['Sample'], rotation=45, ha='right', fontsize=8)
axes[2].set_ylabel('Cortisol Proxy Score', fontsize=12)
axes[2].set_title('Cortisol Reduction Potential per Sample', fontsize=13)
axes[2].grid(True, alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2B6CB0', label='Amazonian'),
    Patch(facecolor='#68D391', label='Lowland')
]
axes[2].legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('cortisol_analysis.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: cortisol_analysis.png")

# ============================================
# 7. SAVE RESULTS
# ============================================
print("\n💾 Step 7: Saving Results")
print("-" * 40)

results_df.to_csv('cortisol_scores.csv', index=False)
contrib_df.to_csv('cortisol_marker_contributions.csv', index=False)
print("   ✅ Saved: cortisol_scores.csv")
print("   ✅ Saved: cortisol_marker_contributions.csv")

print("\n" + "="*70)
print("✅ CORTISOL ANALYSIS COMPLETE!")
print("="*70)
print(f"\n📊 Summary:")
print(f"   • Markers analyzed: {len(marker_data)}")
print(f"   • Amazonian mean score: {amazonian_scores.mean():.2f}")
print(f"   • Lowland mean score: {lowland_scores.mean():.2f}")
print(f"   • Statistical significance: p = {p_value:.4f}")
print(f"   • Top driver: {contrib_df.iloc[0]['Metabolite']}")
