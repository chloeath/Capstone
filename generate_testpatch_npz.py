import os
import numpy as np
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------
# Mapping Setup (from resunet_final.py)
# ---------------------------
nc12_df = pd.read_csv(r"C:\Users\ChloeAtherton\Capstone\nc12.csv")
nc13_df = pd.read_csv(r"C:\Users\ChloeAtherton\Capstone\nc13.csv")
nc16_df = pd.read_csv(r"C:\Users\ChloeAtherton\Capstone\nc16.csv")

merged_classes = pd.concat([nc12_df, nc13_df, nc16_df], ignore_index=True)
merged_classes.drop(columns=['class'], inplace=True)
merged_classes.drop_duplicates(inplace=True)
merged_classes = merged_classes.reset_index(drop=True)
merged_classes['gt label'] = merged_classes.index

nc12_mapping = pd.merge(nc12_df, merged_classes, on='label', how='left')
nc13_mapping = pd.merge(nc13_df, merged_classes, on='label', how='left')
nc16_mapping = pd.merge(nc16_df, merged_classes, on='label', how='left')

nc12_mapping_dict = dict(zip(nc12_mapping['class'], nc12_mapping['gt label']))
nc13_mapping_dict = dict(zip(nc13_mapping['class'], nc13_mapping['gt label']))
nc16_mapping_dict = dict(zip(nc16_mapping['class'], nc16_mapping['gt label']))

# ---------------------------
# Helper Functions
# ---------------------------
def load_hdf5_mat_file(file_path):
    """Load a .mat file using h5py and return HSI and GT data."""
    try:
        with h5py.File(file_path, 'r') as f:
            hsi_data = np.array(f['HSI'])  # Expected shape: (270, H, W)
            gt_data = np.array(f['GT'])    # Expected shape: (H, W)
        print(f"Loaded {file_path} with HSI shape {hsi_data.shape} and GT shape {gt_data.shape}")
        return hsi_data, gt_data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def ScaleData(X, min_value, max_value):
    """Scale data X between 0 and 1."""
    X = X.astype(np.float32)
    return (X - min_value) / (max_value - min_value)

def split_into_tiles(hsi_data, gt_data, tile_size=32, overlapping=True):
    """
    Process a full HSI image and its GT:
      - Scale HSI data.
      - Adjust GT: set 0 values to 255 (background) and subtract 1 from non-background.
      - Pad HSI and GT so dimensions are multiples of tile_size.
      - Extract overlapping patches.
    """
    # Scale HSI
    hsi_data = ScaleData(hsi_data, np.min(hsi_data), np.max(hsi_data))
    
    # Adjust GT: set zeros to 255 and subtract 1 from non-background pixels
    gt_data[gt_data == 0] = 255
    gt_data[gt_data != 255] -= 1

    # Pad so that height and width are multiples of tile_size
    H = hsi_data.shape[1]
    W = hsi_data.shape[2]
    pad_H = tile_size * ((H // tile_size) + (0 if H % tile_size == 0 else 1)) - H
    pad_W = tile_size * ((W // tile_size) + (0 if W % tile_size == 0 else 1)) - W
    
    hsi_data = np.pad(hsi_data, ((0, 0), (0, pad_H), (0, pad_W)), mode='constant')
    gt_data = np.pad(gt_data, ((0, pad_H), (0, pad_W)), mode='constant', constant_values=255)
    
    # Extract overlapping patches using a sliding window with 50% overlap
    hsi_tiles = []
    gt_tiles = []
    stride = tile_size // 2
    padded_H = hsi_data.shape[1]
    padded_W = hsi_data.shape[2]
    for i in range(0, padded_H - tile_size + 1, stride):
        for j in range(0, padded_W - tile_size + 1, stride):
            hsi_patch = hsi_data[:, i:i+tile_size, j:j+tile_size]
            gt_patch = gt_data[i:i+tile_size, j:j+tile_size]
            # Only add patch if there's at least one non-background pixel
            if np.any(gt_patch != 255):
                hsi_tiles.append(hsi_patch)
                gt_tiles.append(gt_patch)
    hsi_tiles = np.array(hsi_tiles)
    gt_tiles = np.array(gt_tiles)
    return hsi_tiles, gt_tiles

def process_multiple_hsi_files(folder_path, tile_size=32):
    """
    Process all .mat files in folder_path.
    For each file, load HSI and GT, apply the correct GT mapping,
    and split the image into overlapping patches.
    Returns combined arrays of HSI patches and GT patches.
    """
    combined_hsi_tiles = []
    combined_gt_tiles = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mat'):
            file_path = os.path.join(folder_path, file_name)
            hsi_data, gt_data = load_hdf5_mat_file(file_path)
            if hsi_data is None or gt_data is None:
                continue
            # Apply the correct mapping based on filename
            if file_name == "NC12.mat":
                gt_data = pd.DataFrame(gt_data).replace(nc12_mapping_dict).to_numpy()
            elif file_name == "NC13.mat":
                gt_data = pd.DataFrame(gt_data).replace(nc13_mapping_dict).to_numpy()
            elif file_name == "NC16.mat":
                gt_data = pd.DataFrame(gt_data).replace(nc16_mapping_dict).to_numpy()
            # Split into overlapping patches
            hsi_tiles, gt_tiles = split_into_tiles(hsi_data, gt_data, tile_size=tile_size, overlapping=True)
            combined_hsi_tiles.extend(hsi_tiles)
            combined_gt_tiles.extend(gt_tiles)
    combined_hsi_tiles = np.array(combined_hsi_tiles)
    combined_gt_tiles = np.array(combined_gt_tiles)
    print(f"Total number of HSI tiles: {len(combined_hsi_tiles)}")
    print(f"Total number of GT tiles: {len(combined_gt_tiles)}")
    return combined_hsi_tiles, combined_gt_tiles

# ---------------------------
# Main Script: Cache 10 Test Patches
# ---------------------------
if __name__ == "__main__":
    # Parameters
    folder_path = r"C:\Users\ChloeAtherton\Capstone\data\mat_data"  # Folder with .mat files
    tile_size = 32
    num_patches = 10  # Number of test patches to save
    output_file = r"C:\Users\ChloeAtherton\Capstone\data\cached_test_patches.npz"
    
    # Process the .mat files and extract overlapping patches with correct mapping
    combined_hsi_tiles, combined_gt_tiles = process_multiple_hsi_files(folder_path, tile_size=tile_size)
    print(f"Extracted {len(combined_hsi_tiles)} total patches from the folder.")
    
    # Split into train and test sets using a fixed random seed (42)
    _, hsi_test, _, gt_test = train_test_split(
        combined_hsi_tiles, combined_gt_tiles, test_size=0.2, random_state=42
    )
    print(f"Test set contains {len(hsi_test)} patches before selection.")
    
    # Select only the first num_patches patches
    if len(hsi_test) < num_patches:
        print("Warning: Fewer test patches available than desired.")
        num_patches = len(hsi_test)
    
    hsi_test = hsi_test[:num_patches]
    gt_test = gt_test[:num_patches]
    
    # Save the selected test patches to an NPZ file
    np.savez(output_file, HSI=hsi_test, GT=gt_test)
    print("Cached test patches saved to", output_file)
