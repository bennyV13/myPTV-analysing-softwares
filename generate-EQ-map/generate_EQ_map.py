import os
import numpy as np
from PIL import Image
import rawpy  # For DNG files
import tifffile as tiff  # For handling TIFF files
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Apply Gaussian filter with sigma=2 (adjustable)

folder_path=r"C:\Users\bennyv\OneDrive - post.bgu.ac.il\מסמכים\University\Master\מחקר\REASEARCH\PTV analysis\analysing-softwares\generate-EQ-map\EQ"
sigma=2
def main(folder_path, sigma):
    EQ_map = Generate_EQ(folder_path, sigma)

    # Get all valid image files (ignoring hidden/system files)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', 'tiff', 'dng', 'png', 'jpg', 'jpeg'))]

    for file in image_files:
        file_path = os.path.join(folder_path, file)
        file_name, file_ext = os.path.splitext(file)  # Get name and extension

        # Check if the image is a DNG
        is_dng = file_ext.lower() == ".dng"

        if is_dng:  # Process DNG files correctly
            with rawpy.imread(file_path) as raw:
                img = raw.postprocess(output_bps=16)  # Convert RAW to 16-bit
                img = np.array(Image.fromarray(img).convert('L')).astype(np.float32)  # Convert to grayscale
        else:  # Process TIFF, PNG, JPG, etc.
            img = np.array(Image.open(file_path).convert('L')).astype(np.float32)  # Convert to grayscale float32

        # Apply Equalization
        processed = Im_EQ(img, EQ_map)

        # Create new filename with "EQ" appended
        output_filename = f"{file_name}_EQ.tif"
        output_path = os.path.join(folder_path, output_filename)

        # Save the processed image as 32-bit TIFF
        tiff.imwrite(output_path, processed.astype(np.float32), dtype=np.float32)

    return


def Generate_EQ(folder_path,sigma):
    mip=generate_MIP(folder_path)
    EQ_map = gaussian_filter(mip, sigma).astype(np.float32)
    return EQ_map

def Im_EQ(image, EQ_map):
    epsilon = 1e-6  # Prevent division by zero
    return image / (EQ_map + epsilon)

def generate_MIP(folder_path):

    z_stack = create_stack(folder_path)  # Get stacked images & format info

    # Compute Maximum Intensity Projection (MIP)
    mip = np.max(z_stack, axis=0)  # Max projection along Z-axis

    return mip

def create_stack(folder_path, max_images=100):
    """
    Reads images from a folder and stacks them into a NumPy array, ensuring a 
    maximum of 100 images with equal spacing if there are more images.
    
    Parameters:
    - folder_path (str): Path to the folder containing images.
    - max_images (int): Maximum number of images to include in the stack.
    
    Returns:
    - z_stack (numpy.ndarray): A stacked NumPy array of selected images.
    """

    # Get all valid image files (ignoring hidden/system files)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('tif', 'tiff', 'dng', 'png', 'jpg', 'jpeg'))] # change to ends with .

    if not image_files:
        raise ValueError("No image files found in the folder")

    # Check if images are DNG
    first_file = image_files[0].lower()
    is_dng = first_file.endswith('.dng')

    # If more than `max_images`, select `max_images` with equal spacing
    num_images = len(image_files)
    if num_images > max_images:
        indices = np.linspace(0, num_images - 1, max_images, dtype=int)  # Generate evenly spaced indices
        image_files = [image_files[i] for i in indices]  # Select only those images

    # Initialize list to store image arrays
    image_stack = []

    # Load images into stack
    for file in image_files:
        file_path = os.path.join(folder_path, file)

        if is_dng:  # Process DNG files
            with rawpy.imread(file_path) as raw:
                img = np.array(Image.open(file_path).convert('L')).astype(np.float32)  # Convert to grayscale (L-mode), and 32 bit
        else:  # Process TIFF, PNG, JPG, etc.
            img = np.array(Image.open(file_path).convert('L')).astype(np.float32)  # Convert to 32-bit float

        image_stack.append(img)

    # Convert list to a NumPy 3D array (Z-stack)
    z_stack = np.stack(image_stack, axis=0)  # Shape: (Z, H, W) or (Z, H, W, C)

    return z_stack

"""
Save:

import tifffile as tiff
import numpy as np
# Assume mip is your grayscale image in NumPy format
tiff.imwrite("output.tif", mip.astype(np.float32), dtype=np.float32)


Plot:

import matplotlib.pyplot as plt
plt.style.use('dark_background')  # Enable dark mode
plt.imshow(mip, cmap='gray')  # Ensure grayscale output
plt.title("Maximum Intensity Projection (MIP)", color='white')  # Title in white
plt.axis("off")  # Hide axes
plt.show()

"""