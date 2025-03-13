# import tqdm 
# import torch
# from torch.utils.data import DataLoader, TensorDataset

# indices = torch.tensor([0, 1, 2])
# bs_indices = torch.arange(0, 3)
# # dim_indices = torch.arange(0, 3)

# hidden = torch.tensor([
#     [[1,2,3], [4,5,6], [7,8,9]],
#     [[10,11,12], [13,14,15], [16,17,18]],
#     [[19,20,21], [22,23,24], [25,26,27]]
# ]
# )

# # indices = indices.unsqueeze(2)
# # print(indices)
# # hidden_sliced = torch.gather(hidden, 1, indices).squeeze(1)
# hidden_sliced = hidden[bs_indices, indices]

# print(hidden_sliced)

import matplotlib.pyplot as plt
import numpy as np

# Sample data for demonstration
# Define x-axis as parameters (logarithmic scale)
# parameters_with_embedding = np.logspace(6, 9, 10)  # Change as per actual data
# parameters_non_embedding = np.logspace(3, 9, 10)   # Change as per actual data

# Sample test loss values for each layer configuration (to be replaced with actual values)
# test_loss_data_with_embedding = {
#     "0 Layer": np.linspace(7, 4, 10),
#     "1 Layer": np.linspace(6, 3.8, 10),
#     "2 Layers": np.linspace(5.8, 3.5, 10),
#     "3 Layers": np.linspace(5.6, 3.3, 10),
#     "6 Layers": np.linspace(5.2, 3.1, 10),
#     ">6 Layers": np.linspace(5, 3, 10)
# }

# test_loss_data_non_embedding = {
#     "1 Layer": np.linspace(6.5, 4.2, 10),
#     "2 Layers": np.linspace(6, 4, 10),
#     "3 Layers": np.linspace(5.8, 3.8, 10),
#     "6 Layers": np.linspace(5.4, 3.4, 10),
#     ">6 Layers": np.linspace(5, 3.2, 10)
# }

archs = {
    'BiGRU': [0.7739, 0.7805, 0.7889, 0.7920, 0.8201],
    'GRU': [0.7842, 0.7814, 0.7749, 0.7458, 0.7621],
    'BiLSTM': [0.7702, 0.7739, 0.7786, 0.7754, 0.7785],
    'LSTM': [0.7758, 0.7767, 0.7608, 0.773, 0.7758],
    'BiDeepRNN': [0.7308, 0.7111, 0.7167, 0.7111, 0.7242],
    'DeepRNN': [0.7364, 0.7242, 0.7392, 0.7064, 0.7317],
}
layers = [1, 2, 3, 4, 5]

# Colors for each line
colors = ['#9E1699', '#0E058D', '#EC764C', '#CD4575', 'red', '#FDB52C']

# Create the figure and axis for subplots
fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

# Plot for Parameters with Embedding
for idx, (label, test_loss) in enumerate(archs.items()):
    ax1.plot(layers, test_loss, label=label, color=colors[idx], marker='o', linestyle='-')

# ax1.set_xscale('log')
ax1.set_xlabel("Layers")
ax1.set_ylabel("Test Acc")
ax1.legend(title="Layers")

# Plot for Parameters without Embedding
# Adjust layout
plt.tight_layout()
plt.show()
# save to image
plt.savefig('test_loss.png')

# import matplotlib.pyplot as plt
# import numpy as np

# # Data for the bar chart
# categories = ['SSIM', 'PSNR', 'FPS', 'Train']
# models = ['Instant-NGP', 'MipNeRF360', 'Ours30K', 'Ours7K']
# values = {
#     'SSIM': [0.72, 0.80, 0.83, 0.78],
#     'PSNR': [24.92, 27.11, 26.91, 25.2],
#     'FPS': [9.0, 0.07, 137.3, 167.9],
#     'Train': [7.5, 48.0, 38.3, 6.1]
# }

# # Colors for each model
# colors = ['#6bb9c7', '#006d63', '#ffb933', '#f35c5b']  # Custom colors

# # Position of bars on x-axis
# x = np.arange(len(categories))
# width = 0.2  # Width of each bar

# # Create subplots
# fig, ax = plt.subplots(figsize=(12, 6))

# # Plot each model's bars in each category
# for i, model in enumerate(models):
#     ax.bar(x + i * width, [values[cat][i] for cat in categories], width, label=model, color=colors[i])

# # Add labels, title, and custom x-axis tick labels
# ax.set_xticks(x + width * 1.5)
# ax.set_xticklabels(categories, fontsize=12)
# ax.set_ylim(0, max(max(values[cat]) for cat in categories) + 10)
# ax.set_ylabel('Values')

# # Annotate each bar with the respective value
# for i, model in enumerate(models):
#     for j, category in enumerate(categories):
#         value = values[category][i]
#         ax.text(j + i * width, value + 0.5, f'{value}', ha='center', fontsize=10, color=colors[i])

# # Add legend
# ax.legend(title="Models")

# # Display plot
# plt.tight_layout()
# plt.show()
