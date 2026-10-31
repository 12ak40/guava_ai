import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("Generating Figures 2 and 3 at 600 DPI...")

# ============================================
# LOAD DATA & RECREATE CORTISOL SCORES
# ============================================
df = pd.read_excel('01.Statistical Analysis/Mean.xlsx', sheet_name='Mean')
df['compound_lower'] = df['compound name'].astype(str).str.lower()

cortisol_markers = {
    'Norwogonin': 1.0, 'Jaceidin': 1.0, 'Quercitrin': 0.8, 'Myricitrin': 0.8, 
    'Luteolin': 0.9, 'Apigenin': 0.7, 'Catechin': 0.6, 'Epicatechin': 0.6, 
    'Eriodictyol': 0.7, 'Isorhamnetin': 0.7, 'Kaempferol': 0.7, 'Rutin': 0.6, 
    'Hyperin': 0.6, 'Isoquercitrin': 0.6, 'Avicularin': 0.6, 'Guaijaverin': 0.6, 
    'Ellagic acid': 0.8, 'Gallic acid': 0.7, 'Chlorogenic acid': 0.7, 'Caffeic acid': 0.7,
}

sample_cols = ['A1_1', 'A1_2', 'A1_3', 'B1_1', 'B1_2', 'B1_3',
               'C1_1', 'C1_2', 'C1_3', 'F1_1', 'F1_2', 'F1_3',
               'P1_1', 'P1_2', 'P1_3']
origin_map = {'A1': 'Lowland', 'B1': 'Lowland', 'C1': 'Amazonian', 'F1': 'Amazonian', 'P1': 'Amazonian'}
origins = [origin_map[col.split('_')[0]] for col in sample_cols]

marker_data = {}
for marker, weight in cortisol_markers.items():
    mask = df['compound_lower'].str.contains(marker.lower(), na=False)
    if mask.any():
        marker_data[marker] = {'values': df[mask].iloc[0][sample_cols].values.astype(float), 'weight': weight}

cortisol_score = np.zeros(len(sample_cols))
total_weight = sum(d['weight'] for d in marker_data.values())

contributions = {}
for marker, data in marker_data.items():
    vals = data['values']
    vals_norm = (vals - vals.min()) / (vals.max() - vals.min()) if vals.max() > vals.min() else np.zeros_like(vals)
    cortisol_score += vals_norm * data['weight']
    contributions[marker] = vals_norm.mean() * data['weight']

cortisol_score = (cortisol_score / total_weight)
cortisol_score = 50 + 25 * (cortisol_score - cortisol_score.min()) / (cortisol_score.max() - cortisol_score.min())

results_df = pd.DataFrame({'Sample': sample_cols, 'Origin': origins, 'Cortisol_Score': cortisol_score})
df_corrected = results_df[~results_df['Sample'].str.startswith('C1')]

# ============================================
# FIGURE 2: CORTISOL BOXPLOT (600 DPI)
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=600)

# Panel A: All Samples
sns.boxplot(x='Origin', y='Cortisol_Score', data=results_df, ax=axes[0], palette={'Amazonian': '#2B6CB0', 'Lowland': '#68D391'})
sns.stripplot(x='Origin', y='Cortisol_Score', data=results_df, ax=axes[0], color='black', size=7, alpha=0.7)
axes[0].set_title('All Samples (n=15)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Cortisol Proxy Score', fontsize=11, fontweight='bold')
axes[0].set_xlabel('')
axes[0].grid(True, alpha=0.3)

# Panel B: Excluding C1
sns.boxplot(x='Origin', y='Cortisol_Score', data=df_corrected, ax=axes[1], palette={'Amazonian': '#2B6CB0', 'Lowland': '#68D391'})
sns.stripplot(x='Origin', y='Cortisol_Score', data=df_corrected, ax=axes[1], color='black', size=7, alpha=0.7)
axes[1].set_title('Excluding Transitional C1 (n=12)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Cortisol Proxy Score', fontsize=11, fontweight='bold')
axes[1].set_xlabel('')
axes[1].grid(True, alpha=0.3)

# Recalculate p-value for the annotation
amazonian_corr = df_corrected[df_corrected['Origin'] == 'Amazonian']['Cortisol_Score']
lowland_corr = df_corrected[df_corrected['Origin'] == 'Lowland']['Cortisol_Score']
t_corr, p_corr = stats.ttest_ind(amazonian_corr, lowland_corr)

axes[1].text(0.5, 0.95, f'p = {p_corr:.4f}', transform=axes[1].transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('Figure_2_Cortisol_Boxplot.png', dpi=600, bbox_inches='tight')
print("✅ Saved Figure 2 as 'Figure_2_Cortisol_Boxplot.png' at 600 DPI")

# ============================================
# FIGURE 3: MARKER DRIVERS BAR CHART (600 DPI)
# ============================================
contrib_df = pd.DataFrame({
    'Metabolite': list(contributions.keys()),
    'Contribution': list(contributions.values())
}).sort_values('Contribution', ascending=False)

top_contrib = contrib_df.head(10)

fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
bars = ax.barh(range(len(top_contrib)), top_contrib['Contribution'].values, color='#2B6CB0', edgecolor='black')

ax.set_yticks(range(len(top_contrib)))
ax.set_yticklabels(top_contrib['Metabolite'].values, fontsize=11, fontweight='bold')
ax.set_xlabel('Contribution to Cortisol Proxy Score', fontsize=11, fontweight='bold')
ax.set_title('Top 10 Metabolites Driving Cortisol Reduction', fontsize=13, fontweight='bold', pad=15)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Add value labels to the bars
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
            ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('Figure_3_Marker_Drivers.png', dpi=600, bbox_inches='tight')
print("✅ Saved Figure 3 as 'Figure_3_Marker_Drivers.png' at 600 DPI")
