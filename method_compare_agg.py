import matplotlib.pyplot as plt
import numpy as np

# Data for the bar chart
categories = ['Unidirectional RNN', 'Bidirectional RNN']
methods = ['Last', 'Average Pooling', 'Max Pooling', 'Attention']
values = {
    'Unidirectional RNN': [0.662, 0.654, 0.657, 0.659],
    'Bidirectional RNN': [0.664, 0.662, 0.658, 0.69],
}

# Colors for each method
colors = ['#6bb9c7', '#ffb933', '#f35c5b', '#9b59b6']  # Custom colors

# Position of bars on x-axis
x = np.arange(len(categories))
width = 0.18  # Width of each bar

# Create subplots
fig, ax = plt.subplots(figsize=(10, 6))

# Adjust y-axis limits to focus on the data range
ax.set_ylim(0.65, 0.70)

# Plot each method's bars in each category
for i, method in enumerate(methods):
    ax.bar(x + i * width - width*1.5, [values[cat][i] for cat in categories], width, label=method, color=colors[i])

# Add labels, title, and custom x-axis tick labels
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('Comparison of Sentence Aggregation Methods on RNN Models', fontsize=14)

# Add gridlines for better readability
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

# Annotate each bar with the respective value
for i, method in enumerate(methods):
    for j, category in enumerate(categories):
        value = values[category][i]
        ax.text(j + i * width - width*1.5, value + 0.0005, f'{value:.3f}', ha='center', fontsize=10, color='black')

# Add legend
ax.legend(title="Methods", fontsize=10, title_fontsize=12)

# Tight layout for better spacing
plt.tight_layout()

# Display plot
plt.show()
