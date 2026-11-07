import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patheffects import withStroke

# Set up the figure: 5:2 aspect ratio (10x4 inches) at 300 DPI
fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off') # Hide axes

# Colors
bg_color = "#F7FAFC"
box_color = "#EBF4FF"
border_color = "#3182CE"
text_dark = "#1A365D"
accent_green = "#68D391"
accent_blue = "#2B6CB0"

# Add background rectangle
rect_bg = patches.Rectangle((0, 0), 10, 4, facecolor=bg_color, edgecolor='none')
ax.add_patch(rect_bg)

# Helper function to draw rounded boxes
def draw_box(x, y, w, h, title, text, color=box_color, border=border_color):
    box = patches.FancyBboxPatch((x, y), w, h,
                                 boxstyle="round,pad=0.1",
                                 edgecolor=border, facecolor=color, linewidth=2)
    ax.add_patch(box)
    # Title
    ax.text(x + w/2, y + h - 0.4, title, ha='center', va='center', 
            fontsize=12, fontweight='bold', color=text_dark, fontfamily='Arial')
    # Body text
    ax.text(x + w/2, y + h/2 - 0.2, text, ha='center', va='center', 
            fontsize=10, color=text_dark, fontfamily='Arial', linespacing=1.5)

# Helper function to draw arrows
def draw_arrow(x_start, x_end, y):
    ax.annotate('', xy=(x_end, y), xytext=(x_start, y),
                arrowprops=dict(arrowstyle='->', color=text_dark, lw=2, mutation_scale=20))

# --- Box 1: Input Data ---
draw_box(0.2, 0.5, 2.1, 3.0, "1. Input Data", "7,458 Metabolites\n(Small sample size)")

# Arrow 1
draw_arrow(2.4, 2.7, 2.0)

# --- Box 2: ML Pipeline ---
draw_box(2.8, 0.5, 2.1, 3.0, "2. Leak-Free Pipeline", "Fold-specific filtering\nLeave-One-Site-Out\nCross-Validation")

# Arrow 2
draw_arrow(5.0, 5.3, 2.0)

# --- Box 3: Classification ---
# Custom drawing for Box 3 to include a scatter plot visual inside
box3 = patches.FancyBboxPatch((5.4, 0.5), 2.1, 3.0, boxstyle="round,pad=0.1",
                              edgecolor=border_color, facecolor=box_color, linewidth=2)
ax.add_patch(box3)
ax.text(6.45, 3.1, "3. Chemotype Classification", ha='center', va='center', 
        fontsize=12, fontweight='bold', color=text_dark, fontfamily='Arial')

# Draw mini-scatter plot inside Box 3
# Lowland (Green)
ax.scatter([5.8, 5.9, 5.85], [2.3, 2.4, 2.2], color=accent_green, s=50, edgecolor='k', zorder=10)
# Amazonian (Blue)
ax.scatter([7.1, 7.0, 7.15], [2.3, 2.4, 2.2], color=accent_blue, s=50, edgecolor='k', zorder=10)
# C1 Transitional (Grey)
ax.scatter([6.45, 6.5, 6.4], [2.3, 2.4, 2.2], color='grey', s=50, edgecolor='k', zorder=10)

ax.text(6.45, 1.2, "80% Accuracy\n(C1 is transitional)", ha='center', va='center', 
        fontsize=10, color=text_dark, fontfamily='Arial')

# Arrow 3
draw_arrow(7.6, 7.9, 2.0)

# --- Box 4: Output ---
draw_box(8.0, 0.5, 1.8, 3.0, "4. Prioritization", "Norwogonin\n\nTop Candidate", color="#EBF4FF")

# Add a little star to Box 4 to highlight the result
ax.text(8.9, 1.5, "★", ha='center', va='center', fontsize=30, color="#D69E2E")

# Save the figure
plt.tight_layout()
plt.savefig('graphical_abstract.png', dpi=300, bbox_inches='tight', transparent=True)
print("✅ Graphical Abstract saved successfully as 'graphical_abstract.png' (300 DPI).")
