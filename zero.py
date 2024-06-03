import torch

# Get all model weights
weights = []
for param in model.parameters():
    weights.extend(param.data.cpu().numpy().flatten())

# Count zero weights
zero_weights = (weights == 0).sum()

# Calculate percentage of zero weights
percentage_zero_weights = (zero_weights / len(weights)) * 100

print(f"Percentage of zero weights: {percentage_zero_weights:.2f}%")
