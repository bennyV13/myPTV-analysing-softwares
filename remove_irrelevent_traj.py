import pandas as pd
import os

def remove_irrelevent(file_path):
    """
    Filters data based on the following rules:
    1. Remove rows starting with -1.
    2. Remove rows where columns 5 to 10 are all zeros.
    3. Remove points outside the ROI: X [0:100], Y [0:100], Z [-50:0].

    Parameters:
    - file_path (str): Path to the input data file.
    - output_folder (str): Optional. Folder to save the filtered file.

    Returns:
    - filtered_data (DataFrame): The filtered DataFrame.
    """

    # Load the data into a DataFrame
    data = pd.read_csv(file_path, sep='\t', header=None)

    # Step 1: Filter out rows starting with -1
    filtered_data = data[data[0] != -1]

    # Step 2: Filter out rows where columns 5 to 10 are all zeros
    mask = (filtered_data.iloc[:, 4:10] != 0).any(axis=1)
    filtered_data = filtered_data[mask]

    # Step 3: Apply ROI Filter for X [0:100], Y [0:100], Z [-50:0]
    # Assuming columns 1, 2, 3 are X, Y, Z (0-based indexing)
    x_col = 1
    y_col = 2
    z_col = 3

    roi_mask = (
        (filtered_data[x_col] >= 0) & (filtered_data[x_col] <= 100) &
        (filtered_data[y_col] >= 0) & (filtered_data[y_col] <= 100) &
        (filtered_data[z_col] >= -50) & (filtered_data[z_col] <= 0)
    )
    filtered_data = filtered_data[roi_mask]
    return filtered_data
   
    # Step 4: Save the filtered data to a new file
    if output_folder is None:
        output_folder = os.path.dirname(file_path)

    output_file_path = os.path.join(output_folder, 'filtered_trajectories')
    filtered_data.to_csv(output_file_path, sep='\t', header=False, index=False)

    print(f"Filtered data saved to: {output_file_path}")

    