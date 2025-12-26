import os
import numpy as np
import h5py
import pandas as pd

# -------------------------------------------------
# Mapping Setup (same as your working notebook)
# -------------------------------------------------
# Load CSV files for mapping
nc12_df = pd.read_csv(r"C:\Users\ChloeAtherton\Capstone\nc12.csv")
nc13_df = pd.read_csv(r"C:\Users\ChloeAtherton\Capstone\nc13.csv")
nc16_df = pd.read_csv(r"C:\Users\ChloeAtherton\Capstone\nc16.csv")

# Merge CSVs and create a unified mapping
merged_classes = pd.concat([nc12_df, nc13_df, nc16_df], ignore_index=True)
merged_classes.drop(columns=['class'], inplace=True)
merged_classes.drop_duplicates(inplace=True)
merged_classes = merged_classes.reset_index(drop=True)
merged_classes['gt label'] = merged_classes.index

# Build per-dataset mapping dictionaries
nc12_mapping = pd.merge(nc12_df, merged_classes, on='label', how='left')
nc13_mapping = pd.merge(nc13_df, merged_classes, on='label', how='left')
nc16_mapping = pd.merge(nc16_df, merged_classes, on='label', how='left')

nc12_mapping_dict = dict(zip(nc12_mapping['class'], nc12_mapping['gt label']))
nc13_mapping_dict = dict(zip(nc13_mapping['class'], nc13_mapping['gt label']))
nc16_mapping_dict = dict(zip(nc16_mapping['class'], nc16_mapping['gt label']))

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def load_hdf5_mat_file(file_path):
    """
    Load a .mat file using h5py and return the HSI and GT data.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            hsi_data = np.array(f['HSI'])  # Expected shape: (270, H, W)
            gt_data = np.array(f['GT'])    # Expected shape: (H, W)
        print(f"Loaded file: {file_path}")
        print(f"HSI shape: {hsi_data.shape}, GT shape: {gt_data.shape}")
        return hsi_data, gt_data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None, None

def ScaleData(X, min_value, max_value):
    """
    Scale the data X between 0 and 1.
    """
    X = X.astype(np.float32)
    X = (X - min_value) / (max_value - min_value)
    return X

def process_full_image(hsi_data, gt_data, mapping_dict):
    """
    Process the full image:
      - Scale the HSI data between 0 and 1.
      - Remap GT labels using the provided mapping dictionary.
      - Adjust GT: set original 0 to 255 and subtract 1 from non-background.
      - Pad the image so that its spatial dimensions are multiples of 8.
    """
    # Normalize HSI data between 0 and 1
    min_val = np.min(hsi_data)
    max_val = np.max(hsi_data)
    hsi_data = ScaleData(hsi_data, min_val, max_val)
    
    # Remap GT labels using the mapping dictionary via pandas replace
    gt_data = pd.DataFrame(gt_data).replace(mapping_dict).to_numpy()
    
    # Adjust GT: set 0 values to 255 (background) and subtract 1 from non-background labels
    gt_data[gt_data == 0] = 255
    gt_data[gt_data != 255] -= 1
    
    # Pad the image so that height and width are multiples of 8
    h = gt_data.shape[0]
    w = gt_data.shape[1]
    multiple_required = 8
    pad_h = multiple_required * ((h // multiple_required) + (0 if h % multiple_required == 0 else 1)) - h
    pad_w = multiple_required * ((w // multiple_required) + (0 if w % multiple_required == 0 else 1)) - w
    
    hsi_data = np.pad(hsi_data, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
    gt_data = np.pad(gt_data, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=255)
    
    print(f"After processing: HSI shape: {hsi_data.shape}, GT shape: {gt_data.shape}")
    return hsi_data, gt_data

def save_processed_full_image(input_file, output_file):
    """
    Load a .mat file, process it, and save the processed full image as an NPZ file.
    """
    hsi_data, gt_data = load_hdf5_mat_file(input_file)
    if hsi_data is None or gt_data is None:
        print("Failed to load", input_file)
        return
    
    # Choose mapping dictionary based on the filename
    if "NC12" in input_file:
        mapping_dict = nc12_mapping_dict
    elif "NC13" in input_file:
        mapping_dict = nc13_mapping_dict
    elif "NC16" in input_file:
        mapping_dict = nc16_mapping_dict
    else:
        mapping_dict = None
    
    hsi_data, gt_data = process_full_image(hsi_data, gt_data, mapping_dict)
    np.savez_compressed(output_file, HSI=hsi_data, GT=gt_data)
    print("Processed full image saved to", output_file)

# -------------------------------------------------
# Main Offline NPZ Generation Script
# -------------------------------------------------
if __name__ == "__main__":
    # Folder containing your .mat files
    mat_folder = r"C:\Users\ChloeAtherton\Capstone\data\nc16"
    # Output folder for NPZ files
    output_folder = r"C:\Users\ChloeAtherton\Capstone\data"

    for file_name in os.listdir(mat_folder):
        if file_name.endswith('.mat'):
            input_file = os.path.join(mat_folder, file_name)
            output_file = os.path.join(output_folder, file_name.replace('.mat', '_processed_NEW.npz'))
            save_processed_full_image(input_file, output_file)
