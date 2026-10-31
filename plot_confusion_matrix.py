import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# The exact results from your leak-free LOOCV run
# 9 Amazonian samples (all correct)
# 6 Lowland samples (all correct)
actuals = ['Amazonian']*9 + ['Lowland']*6
predictions = ['Amazonian']*9 + ['Lowland']*6

# Define the order of labels
labels = ['Amazonian', 'Lowland']

# Compute the confusion matrix
cm = confusion_matrix(actuals, predictions, labels=labels)

# Create the plot
fig, ax = plt.subplots(figsize=(6, 5))

# Plot using Seaborn heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=labels, yticklabels=labels,
            annot_kws={'size': 20, 'weight': 'bold'},
            linewidths=1, linecolor='black', ax=ax)

# Customize labels and title
ax.set_xlabel('Predicted Chemotype', fontsize=12, fontweight='bold')
ax.set_ylabel('True Chemotype', fontsize=12, fontweight='bold')
ax.set_title('Leak-Free LOOCV Classification (n=15)', fontsize=14, fontweight='bold', pad=15)

# Adjust tick label sizes
ax.tick_params(axis='both', labelsize=11)

# Save the figure at high resolution for publication
plt.tight_layout()
plt.savefig('Figure_1B_ConfusionMatrix.png', dpi=300, bbox_inches='tight')

print("✅ Figure 1B successfully saved as 'Figure_1B_ConfusionMatrix.png'")
