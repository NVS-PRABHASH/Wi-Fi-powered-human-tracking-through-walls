import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the CSV file into a pandas DataFrame
# Replace 'test_sample_sit.csv' with your actual CSV file path if needed.
df = pd.read_csv('data/test_samples/test_sample_stand.csv')

# Step 2: Extract CSI columns.
# Assume CSI columns are named like 'csi_0', 'csi_1', ..., 'csi_N'.
# We filter columns by the prefix 'csi_' to get all subcarrier data.
csi_columns = [col for col in df.columns if col.startswith('csi_')]
csi_data = df[csi_columns].to_numpy()  # Convert to NumPy array

# Now csi_data is a 2D array of shape (num_samples, num_subcarriers).
# For example, with one sample and 256 subcarriers, shape will be (1, 256).

# Step 3: Generate the heatmap.
plt.figure(figsize=(10, 6))
# Plot heatmap; cmap can be changed as needed (e.g., 'viridis', 'plasma', etc.).
sns.heatmap(csi_data, cmap='viridis', cbar=True, cbar_kws={'label': 'CSI Value'})
plt.title('CSI Subcarrier Values Heatmap')
plt.xlabel('Subcarrier Index')
plt.ylabel('Sample Index (Time)')
plt.show()
