import numpy as np
import os
import glob
import scipy.io
from scipy.ndimage import zoom

def resize_volume(volume, target_shape, order=3):
    """Resize the 3D volume to target shape."""
    factors = [t_dim / o_dim for t_dim, o_dim in zip(target_shape, volume.shape)]
    try:
        resized_volume = zoom(volume, factors, order=order)
    except Exception as e:
        print(f"An error occurred while resizing: {e}")
        return None
    return resized_volume

def process_mat_files(input_folder, output_folder, data_key='images', target_shape=(64, 128, 64), order=3):
    """Process .mat files in the input folder and save resized 3D volumes as .npy files."""
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List all .mat files in the input folder
    for filename in glob.glob(os.path.join(input_folder, "*.mat")):
        file_base_name = os.path.basename(filename)
        print(f"Processing {file_base_name}...")
        
        try:
            # Load the .mat file
            mat = scipy.io.loadmat(filename)
        except Exception as e:
            print(f"An error occurred while reading {file_base_name}: {e}")
            continue
        
        # Fetch the 3D array from the .mat file
        if data_key in mat:
            volume = mat[data_key]
            
            # Resize the volume
            resized_volume = resize_volume(volume, target_shape, order)
            if resized_volume is None:
                continue
            
            # Save the resized 3D volume as .npy file
            output_file_path = os.path.join(output_folder, file_base_name.replace('.mat', '.npy'))
            np.save(output_file_path, resized_volume)
            
            print(f"Saved resized volume for {file_base_name}.")
        else:
            print(f"Could not find the key '{data_key}' in {file_base_name}. Skipping.")

# Paths to input and output folders
input_folder = '/your/path/Normal'  # Update with your .mat files folder
output_folder = '/your/path/Resized/Data/'  # Update with your desired output folder for .npy files
  
# Execute the processing
process_mat_files(input_folder, output_folder)
