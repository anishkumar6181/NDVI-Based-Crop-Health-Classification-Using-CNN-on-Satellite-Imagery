import os
import tifffile
import numpy as np

# Determine base directory (two levels up from this file's directory)
base_dir = os.path.dirname(os.path.dirname(__file__))

# File paths
filePathTiff = os.path.join(base_dir, r'data\raw\train\Health\hyper (1).tif')
filePathNpy = os.path.join(base_dir, r'data\processed\Health\hyper (1).npy')

# Read and inspect TIFF image
img = tifffile.imread(filePathTiff)
print('Tiff Image shape:', img.shape) # output - (64,64,125)

# Load the array and check its shape - should be (64, 64) for NDVI
data = np.load(filePathNpy)
print(f"Array shape: {data.shape}")  # Dimensions (e.g., (64, 64) for NDVI)
print(f"Array dtype: {data.dtype}")  # Data type (e.g., float32)
print(f"Array size: {data.size}")    # Total elements (e.g., 64*64 = 4096)