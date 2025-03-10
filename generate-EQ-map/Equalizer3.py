import os
import sys
import numpy as np
from PIL import Image
import rawpy  # For DNG files
import tifffile as tiff  # For handling TIFF files
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from IPython import embed  # Optional: for interactive debugging if needed
import math

def Generate_EQ(folder_path, sigma):
    mip = generate_MIP(folder_path)
    return gaussian_filter(mip, sigma).astype(np.float32)

def Im_EQ(image, EQ_map):
    epsilon = 1e-6  # Prevent division by zero
    image=image / (EQ_map + epsilon)
    max=np.max(image)
    ref=254.0/max
    image=np.ceil(image*ref)
    return image

def generate_MIP(folder_path):
    z_stack = create_stack(folder_path)
    return np.max(z_stack, axis=0)

def create_stack(folder_path, max_images=100):
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('tif', 'tiff', 'dng', 'png', 'jpg', 'jpeg'))]

    if not image_files:
        raise ValueError("No image files found in the folder.")

    num_images = len(image_files)
    if num_images > max_images:
        indices = np.linspace(0, num_images - 1, max_images, dtype=int)
        image_files = [image_files[i] for i in indices]

    image_stack = []
    for file in image_files:
        file_path = os.path.join(folder_path, file)
        is_dng = file.lower().endswith('.dng')
        if is_dng:  # Process DNG files
            with rawpy.imread(file_path) as raw:
                img = raw.postprocess(output_bps=16)
            img = img.astype(np.float32)
            img = img[:, :, 0]  # Take only the first channel (Red)
        else:  # Process TIFF, PNG, JPG, etc.
            img = np.array(Image.open(file_path).convert('L')).astype(np.float32)
        image_stack.append(img)

    return np.stack(image_stack, axis=0)

def main(folder_path,output_path, sigma):
    # Generate the EQ_map
    EQ_map = Generate_EQ(folder_path, sigma)
    
    # Loop until the user confirms the EQ_map or chooses to exit
    while True:
        EQ_map = Generate_EQ(folder_path, sigma)
        
        # Display the EQ_map for review
        plt.figure()
        plt.imshow(EQ_map, cmap='gray')
        plt.title(f'EQ_map Preview (sigma = {sigma})')
        plt.axis('off')
        plt.show()

        # Ask the user if they want to continue processing with this EQ_map,
        # choose a new sigma, or exit the process.
        user_input = input("Enter 'y' to continue, 'n' to choose a new sigma, or 'exit' to terminate: ").strip().lower()
        if user_input in ['y', 'yes']:
            break  # Continue with the confirmed EQ_map
        elif user_input in ['exit', 'quit', 'q']:
            print("Terminating process as per user request.")
            sys.exit(0)
        elif user_input in ['n', 'new']:
            new_sigma = input("Enter new sigma value: ").strip()
            try:
                sigma = float(new_sigma)
            except ValueError:
                print("Invalid sigma value. Please enter a numeric value.")
    
    # Get all valid image files from the folder
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.tif', 'tiff', 'dng', 'png', 'jpg', 'jpeg'))]

    if not image_files:
        print("No image files found.")
        return

    for file in image_files:
        file_path = os.path.join(folder_path, file)
        file_name, file_ext = os.path.splitext(file)
        is_dng = file_ext.lower() == ".dng"

        if is_dng:  # Process DNG files correctly
            with rawpy.imread(file_path) as raw:
                img = raw.postprocess(output_bps=16)
            img = img.astype(np.float32)
            img = img[:, :, 0]  # Take only the first channel (Red)
        else:  # Process TIFF, PNG, JPG, etc.
            img = np.array(Image.open(file_path).convert('L')).astype(np.float32)

        # Apply Equalization
        processed = Im_EQ(img, EQ_map)

        # Create new filename with "EQ" appended and save in the output folder
        output_filename = f"{file_name}_EQ.tif"
        full_output_path = os.path.join(output_path, output_filename)
        tiff.imwrite(full_output_path, processed.astype(np.uint8))

folder_path = r"c:\Users\bennyv\Documents_lab\Research-stuff\for-generate-EQ-map\eq-test-300-rec21-1\before"
output_path = r"c:\Users\bennyv\Documents_lab\Research-stuff\for-generate-EQ-map\eq-test-300-rec21-1\after2"
sigma = 20
main(folder_path,output_path, sigma)
