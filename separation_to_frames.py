import pandas as pd
import numpy as np

# Step 1: Load the data (adjust delimiter as needed)
file_path = r'C:\Users\bennyv\Documents_lab\PTV analysis\20240916_experiment1\first segmentation try\10 images\smoothed_trajectories - Copy'  # Replace with your file path
df = pd.read_csv(file_path, delim_whitespace=True, header=None)  # Adjust delimiter if needed

# Step 2: Verify the data
print("DataFrame Preview:\n", df.head())
print("\nDataFrame Shape:", df.shape)

# Step 3: Extract columns 2, 3, 4 and the last column
# Assuming zero-based indexing (i.e., column 1 is index 0)
data_columns = df.iloc[:, 1:4]  # Columns 2, 3, and 4
group_column = df.iloc[:, -1]   # Last column (for grouping)

# Step 4: Group data and create separate matrices
grouped_data = {}

# Iterate through unique values in the last column
for value in group_column.unique():
    # Filter rows where the last column matches the value
    filtered_data = data_columns[group_column == value].to_numpy()

    # Store the matrix in a dictionary
    grouped_data[f'frame_{int(value)}'] = filtered_data

    # Display the created matrix
    print(f"\nMatrix for frame_{int(value)}:\n", filtered_data)

# Step 5: Access matrices
# Example: Access frame_2
frame_2 = grouped_data.get('frame_2')
print("\nFrame 2 Matrix:\n", frame_2)
