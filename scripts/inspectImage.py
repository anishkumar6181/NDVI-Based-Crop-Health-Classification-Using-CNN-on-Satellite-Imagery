import tifffile
filePathTiff = 'data/beyond-visible-spectrum-ai-for-agriculture-2024/train/Health/hyper (1).tif'
img = tifffile.imread(filePathTiff)
print('Tiff Image shape:', img.shape) # output - (64,64,125)


import numpy as np
filePathNpy = 'data/processed/Health/hyper (1).npy'
# Load the array and check its shape - should be (64, 64) for NDVI
data = np.load(filePathNpy)
print(f"Array shape: {data.shape}")  # Dimensions (e.g., (64, 64) for NDVI)
print(f"Array dtype: {data.dtype}")  # Data type (e.g., float32)
print(f"Array size: {data.size}")    # Total elements (e.g., 64*64 = 4096)