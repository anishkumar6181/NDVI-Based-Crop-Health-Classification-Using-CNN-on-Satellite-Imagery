import os
import numpy as np
import tifffile         # For reading TIFF files
import cv2              # OpenCV for image resizing
from tqdm import tqdm   # Progress bar
import shutil           # File operations

class HyperspectralImageProcessor:
    def __init__(self, input_dir, output_dir, img_size):
        # configuration settings
        self.__INPUT_DIR = input_dir
        self.__OUTPUT_DIR = output_dir
        self.__IMG_SIZE = img_size  # Resize images to sz*sz pixels

        # Hyperspectral camera settings (for band index calculation)
        self.__START_WAVELENGTH = 490  # First band wavelength (nm)
        self.__RESOLUTION = 4  # Wavelength step between bands (nm)

        # Wavelengths for NDVI calculation (in nanometers)
        self.__RED_WAVELENGTH = 670  # Red light band
        self.__NIR_WAVELENGTH = 850  # Near-Infrared band
 
    # HELPER FUNCTIONS
    def __wavelengthToBandIndex(self,wavelength_nm:float) -> int:
        """
            Description : Convert a wavelength in nanometers to the corresponding band index.
            Parameters  : wavelength_nm (float) - Wavelength in nanometers
            Example     : If the camera starts at 490nm with 4nm resolution:
                          - 670nm → band index = (670-490)/4 = 45
            Returns     : int - Band index corresponding to the wavelength
            Author      : Alok Ranjan
        """
        return int((wavelength_nm - self.__START_WAVELENGTH) / self.__RESOLUTION)
    
    def __calculateNDVI(self,hyperspectral_image):
        """
            Description : Compute NDVI from a hyperspectral image.
                            NDVI = (NIR - Red) / (NIR + Red)
            Returns     : 2D NDVI array with values in range [-1, 1]
            Author      : Alok Ranjan
        """
        # Get band indices for red and near-infrared
        red_band = self.__wavelengthToBandIndex(self.__RED_WAVELENGTH)
        nir_band = self.__wavelengthToBandIndex(self.__NIR_WAVELENGTH)

        # Sanity check: prevent index out of bounds
        total_bands = hyperspectral_image.shape[0]
        if red_band >= total_bands or nir_band >= total_bands:
            raise ValueError(f"Band index out of range: red={red_band}, nir={nir_band}, available={total_bands}")
        
        # Extract red and NIR bands (convert to float for calculations)
        red = hyperspectral_image[red_band].astype(np.float32)
        nir = hyperspectral_image[nir_band].astype(np.float32)
        
        # Calculate NDVI (add small value to denominator to avoid division by zero)
        denominator = (nir + red + 1e-8)
        ndvi = (nir - red) / denominator
        
        # Handle NaN and inf - Clean up invalid values (NaN, infinities)
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return ndvi

    # main processing function
    def processAllImages(self):
        print("\n=== Hyperspectral Image Processor ===")
        print(f"Input Directory: {os.path.abspath(self.__INPUT_DIR)}")
        print(f"Output Directory: {os.path.abspath(self.__OUTPUT_DIR)}")
        print(f"Target Image Size: {self.__IMG_SIZE}x{self.__IMG_SIZE} pixels")
        print(f"Using Bands: Red={self.__RED_WAVELENGTH}nm, NIR={self.__NIR_WAVELENGTH}nm\n")

        # Check if input directory exists
        if not os.path.exists(self.__INPUT_DIR):
            print(f"Error: Input directory not found at {self.__INPUT_DIR}")
            return

        # Clear old output directory if it exists
        if os.path.exists(self.__OUTPUT_DIR):
            shutil.rmtree(self.__OUTPUT_DIR)
            print("Cleared existing output directory.")

        # Process each class (Health, Rust, Other)
        class_names = ['Health', 'Rust', 'Other']
        for class_name in class_names:
            input_class_dir = os.path.join(self.__INPUT_DIR, class_name)
            output_class_dir = os.path.join(self.__OUTPUT_DIR, class_name)
            
            # Create output directory for this class
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Get all TIFF files in this class directory
            tiff_files = [f for f in os.listdir(input_class_dir) if f.endswith('.tif')]
            
            print(f"\nProcessing {len(tiff_files)} images in class '{class_name}'...")
            
            # Process each image with progress bar
            for filename in tqdm(tiff_files, desc=class_name):
                input_path = os.path.join(input_class_dir, filename)
                
                try:
                    # Read TIFF file (shape: [height, width, bands])
                    img = tifffile.imread(input_path)
                    
                    # Move bands to first dimension [bands, height, width]
                    img_bands_first = img.transpose(2, 0, 1)
                    
                    # Calculate NDVI
                    ndvi = self.__calculateNDVI(img_bands_first)
                    
                    # Resize NDVI image
                    ndvi_resized = cv2.resize(
                        ndvi, 
                        (self.__IMG_SIZE, self.__IMG_SIZE), 
                        interpolation=cv2.INTER_AREA
                    )
                    
                    # Save as numpy array
                    output_filename = filename.replace('.tif', '.npy')
                    output_path = os.path.join(output_class_dir, output_filename)
                    np.save(output_path, ndvi_resized)
                    
                except Exception as e:
                    print(f"\nError processing {filename}: {str(e)}")
                    continue

        print("\n=== Processing Complete ===")
        print(f"Saved results to: {os.path.abspath(self.__OUTPUT_DIR)}")


# Run the main processing function if this script is executed directly
input_dir = 'data/raw/train'
output_dir = 'data/processed'
img_size = 64
image_processor = HyperspectralImageProcessor(input_dir, output_dir, img_size)
def processAllImages():
    image_processor.processAllImages()
if __name__ == "__main__":
    processAllImages()
else:
    print("Module loaded. Use processAllImages() to start processing.")