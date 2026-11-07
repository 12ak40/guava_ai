import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# The exact results from your Leave-One-Site-Out run
# Total samples: 15 (9 Amazonian, 6 Lowland)
# Actuals: A1(3 L), B1(3 L), C1(3 A), F1(3 A), P1(3 A)
# Predictions: C1 was incorrectly predicted as Lowland (3 errors), all others correct.

actuals = ['Amazonian']*9 + ['Lowland']*6
predictions = ['Lowland']*3 + ['Amazonian']*6 + ['Lowland']*6  

labels = ['Amazonian', 'Lowland']

# Compute the confusion matrix
cm = confusion_matrix(actuals, predictions, labels=labels)

# Create the plot (600 DPI for publication)
fig, ax = plt.subplots(figsize=(6, 5), dpi=600)

# Plot using Seaborn heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=labels, yticklabels=labels,
            annot_kws={'size': 20, 'weight': 'bold'},
            linewidths=1, linecolor='black', ax=ax)

# Customize labels and title
ax.set_xlabel('Predicted Chemotype', fontsize=12, fontweight='bold')
ax.set_ylabel('True Chemotype', fontsize=12, fontweight='bold')
ax.set_title('Leave-One-Site-Out Validation (n=15)', fontsize=14, fontweight='bold', pad=15)

# Adjust tick label sizes
ax.tick_params(axis='both', labelsize=11)

# Save the figure
plt.tight_layout()
plt.savefig('Figure_1B_ConfusionMatrix.png', dpi=600, bbox_inches='tight')

print("✅ Figure 1B successfully updated at 600 DPI to reflect LOSOCV (80% accuracy).")
print("The plot now shows the 3 misclassifications from the transitional site C1.")
