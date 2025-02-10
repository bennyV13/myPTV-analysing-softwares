#matrix_spiter

import numpy as np
from scipy.interpolate import griddata
import os
#import matplotlib.pyplot as plt
#from matplotlib import cm


def find_3D_matrix(t0):
    # Filter data for the given time
    t0_data = data[np.isclose(times, t0)]
    if t0_data.size == 0:
        print("No data found for the given time.")
        return

    valid_particles = t0_data[:, 0] != -1
    t0_particles = t0_data[valid_particles]

    # Extract columns
    p, x, y, z, u, v, w, a_x, a_y, a_z = t0_particles[:, :10].T

    # Define the grid domain
    x_min, x_max = 0, 100
    y_min, y_max = 0, 100
    z_min, z_max = -50, 0

    # Generate a structured 3D grid
    #grid_size = 70
    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)
    z_grid = np.linspace(z_min, z_max, 50)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid)

    # Interpolate velocity components onto the grid
    points = np.column_stack((x, y, z))
    U = griddata(points, u, (X, Y, Z), method='linear', fill_value=0)
    V = griddata(points, v, (X, Y, Z), method='linear', fill_value=0)
    W = griddata(points, w, (X, Y, Z), method='linear', fill_value=0)
    
    


    # Save matrices to files
    directory = "C:/Users/pele/Desktop/עיבוד תוצאות/matrix"
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"U_time_{t0}.npz")
    np.savez(file_path, U=U, V=V, W=W)
    
###### end of function ######

# Load the data (fixing the file path issue)
data = np.loadtxt(r"20240116_experiment2\voxel_20\smoothed_trajectories")

# Validate shape
print("Data shape:", data.shape)

times = data[:, -1]
t0 = int(np.min(times))  # Convert to an integer if needed
t_max = int(np.max(times))  # Convert to an integer if needed

# Use range to iterate from t0 to t_max
for i in range(t0, t_max):  # Include t_max in the range
   find_3D_matrix(i)


#with open("averaging.py") as f:
#    exec(f.read())

    

    
 
    








#velocity_magnitude_0 = ((U**2 + V**2 + W**2)**0.5)

#velocity_magnitude = velocity_magnitude_0/velocity_magnitude_0.max()

#norm = plt.Normalize(velocity_magnitude.min(), velocity_magnitude.max())
#colors = cm.plasma(norm(velocity_magnitude))  # Use a colormap like 'plasma'

# Plot the 3D vector field
#fig = plt.figure(figsize=(12, 8))
#ax = fig.add_subplot(111, projection='3d')

# Flatten the grids and data for quiver plot
#ax.quiver(
    #X.flatten(), Y.flatten(), Z.flatten(),
    #U.flatten(), V.flatten(), W.flatten(),
    #length=0.1, normalize=True, colors=colors.reshape(-1, 4)  # RGBA colors
    #)

# Add a color bar
#mappable = cm.ScalarMappable(norm=norm, cmap='plasma')
#mappable.set_array(velocity_magnitude)
#cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
#cbar.set_label('Velocity Magnitude')

#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#ax.set_title('3D Vector Field with Color Indicating Magnitude')

#plt.savefig(f'vector_fild_time_{t0}.png')