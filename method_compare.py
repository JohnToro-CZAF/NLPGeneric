import matplotlib.pyplot as plt
import numpy as np

# Data for the bar chart
categories = ['Word2vec-GRU', 'Word2vec-LSTM','Glove-GRU', 'Glove-LSTM']
methods = ['Using UNK', 'Average Sentence', 'Average Context']
values = {
    'Word2vec-GRU': [0.7758, 0.7758, 0.7683, 0.7767],
    'Word2vec-LSTM': [0.7786, 0.7627, 0.7655, 0.7824],
    'Glove-GRU': [0.7636, 0.7749, 0.7598, 0.7749],
    'Glove-LSTM': [0.7767, 0.7598, 0.7430, 0.7505],
}

# Colors for each model
colors = ['#6bb9c7', '#ffb933', '#f35c5b', '#9b59b6']  # Custom colors

# Position of bars on x-axis
x = np.arange(len(categories))
width = 0.18  # Width of each bar

# Create subplots
fig, ax = plt.subplots(figsize=(12, 6))
# Adjust y-axis limits to focus on the data range
ax.set_ylim(0.725, 0.8)

# Plot each method's bars in each category
for i, method in enumerate(methods):
    ax.bar(x + i * width - width*1.5, [values[cat][i] for cat in categories], width, label=method, color=colors[i])

# Add labels, title, and custom x-axis tick labels
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('Comparison of OOV Handling Methods on GRU-LSTM, Word2Vec-Glove Models', fontsize=14)

# Add gridlines for better readability
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

# Annotate each bar with the respective value
for i, method in enumerate(methods):
    for j, category in enumerate(categories):
        value = values[category][i]
        ax.text(j + i * width - width*1.5, value + 0.0005, f'{value:.3f}', ha='center', fontsize=10, color='black')

# Add legend
ax.legend(title="Models")

# Display plot
plt.tight_layout()
plt.show()
