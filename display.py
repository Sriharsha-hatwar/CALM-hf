import matplotlib.pyplot as plt
import numpy as np

# Assume your list of losses is called 'losses'
losses = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the losses
ax.plot(range(len(losses)), losses, marker='o', markersize=8, linestyle='-', linewidth=2, color='r', label='Losses')

# Set the title and axis labels
ax.set_title('Loss Curve', fontsize=18)
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)

# Add a legend
ax.legend(fontsize=12)

# Set the x-axis tick labels
num_ticks = int(max(range(len(losses))) / 0.5) + 1
xtick_positions = np.arange(0, len(losses), 0.5)
xtick_labels = [f'{i:.1f}' for i in xtick_positions]
ax.set_xticks(xtick_positions)
ax.set_xticklabels(xtick_labels, fontsize=12, rotation=0)

# Set the y-axis tick labels
ax.set_yticklabels([f'{loss:.2f}' for loss in ax.get_yticks()], fontsize=12)

# Add grid lines
ax.grid(linestyle='--', alpha=0.5)

# Adjust the spacing between subplots
plt.tight_layout()

# Save the plot
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')