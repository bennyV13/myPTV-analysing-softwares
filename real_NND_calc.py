from scipy.spatial import KDTree
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import embed
from scipy.stats import gaussian_kde
from scipy.stats import kstest, norm
from CSR_NND_calc import rand_data
from remove_irrelevent_traj import remove_irrelevent


# Load the file
# Step 1: Load the data (adjust delimiter as needed)
file_path = r'20250203_analysis\rec17\smoothed_trajectories'  # Replace with your file path
df=remove_irrelevent(file_path)

# Step 2: Verify the data
# print("Data preview:")
# print(df.head())
# print("\nData description:")
# print(df.describe())

# Step 3: Extract columns 2, 3, 4 and the last column
# Assuming zero-based indexing (i.e., column 1 is index 0)
data_columns = df.iloc[:, 1:4]  # Columns 2, 3, and 4
group_column = df.iloc[:, -1]   # Last column (for grouping)

# Step 4: Group data and create separate matrices
grouped_data = {}
i=0
p_value={}
nni=[]
z_obs=[]
z_rand=[]
# Iterate through unique values in the last column
for value in group_column.unique():
    # Filter rows where the last column matches the value
    filtered_data = data_columns[group_column == value].to_numpy()

    # Store the matrix in a dictionary
    grouped_data[f'frame_{int(value)}'] = filtered_data

    f = grouped_data.get(f'frame_{int(value)}')
    x_size=np.max(f[:,0])-np.min(f[:,0])
    y_size=np.max(f[:,1])-np.min(f[:,1])
    z_size=np.max(f[:,2])-np.min(f[:,2])
    size_of_roi=x_size*y_size*z_size
    
    #calculate distances for random data
    particles_count=f.shape[0]
    rand_dist=rand_data(particles_count,x_size,y_size,z_size)
    rand_dist1=rand_data(particles_count,x_size,y_size,z_size)

    # Build KDTree
    kdtree = KDTree(f)

   
  
    # Query for the nearest neighbor for each particle
    # k=2 because the closest point is the particle itself
    distances, indices = kdtree.query(f , k=2)

    # Extract the nearest neighbor distances (skip the first which is distance to self)
    data = distances[:, 1]
    nearest_indices = indices[:, 1]
    obs_mean=np.mean(data)
    obs_std=np.std(data)
    rand_mean=np.mean(rand_dist)
    rand_mean1=np.mean(rand_dist1)
    rand_std=np.std(rand_dist)
    rand_std1=np.std(rand_dist1)
    expected_mean=0.55396/(particles_count/size_of_roi)**(1/3)
    nni.append(obs_mean/expected_mean)
    z_obs.append((obs_mean-rand_mean)/(rand_std)*np.sqrt(particles_count))
    z_rand.append((rand_mean1-rand_mean)/(rand_std)*np.sqrt(particles_count))
    i+=1
    # Calculate PDF
    kde = gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), 1000)
    pdf = kde(x_vals)
    #statistic, p_value[i] = kstest(data, 'norm',args=(mean,std))


    # # Plot PDF
    # x_vals=np.linspace(min(data), max(data), 1000)
    # plt.plot(x_vals, pdf, label='PDF')
    # #plt.fill_between(x_vals, pdf, alpha=0.3)
    # plt.title('PDF using SciPy gaussian_kde')
    # plt.xlabel('Nearest distances [mm]')
    # plt.ylabel('Density')
    # plt.show()
    
# Plot NNI
cheat=i
x_nni = np.arange(cheat)  # or np.linspace(0, i, i), either is fine
# plt.figure()
# plt.plot(x_nni, [nni[k] for k in range(cheat)], label='NNI')
# plt.xlabel('Frame number')
# plt.ylabel('NNI')

# Plot z_obs
plt.figure()
plt.plot(x_nni, [z_obs[k] for k in range(cheat)], label='Z score')
plt.xlabel('Frame number')
plt.ylabel('Z score')
mean_z=np.mean(z_obs)
std_z=np.std(z_obs)
plt.axhline(y=mean_z+std_z, color='g', linestyle='--', linewidth=1, label=f'Standard deviation = {std_z:.2f}')
plt.axhline(y=mean_z-std_z, color='g', linestyle='--', linewidth=1)
plt.axhline(y=mean_z, color='r', linestyle='--', linewidth=1, label=f'Mean = {mean_z:.2f}')
plt.legend()

# Plot z_rand
plt.figure()
plt.plot(x_nni, [z_rand[k] for k in range(cheat)], label='Z score')
plt.xlabel('Frame number')
plt.ylabel('Z score')
mean_z=np.mean(z_rand)
std_z=np.std(z_rand)
plt.axhline(y=mean_z+std_z, color='g', linestyle='--', linewidth=1, label=f'Standard deviation = {std_z:.2f}')
plt.axhline(y=mean_z-std_z, color='g', linestyle='--', linewidth=1)
plt.axhline(y=mean_z, color='r', linestyle='--', linewidth=1, label=f'Mean = {mean_z:.2f}')


plt.legend()
plt.show()
