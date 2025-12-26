# %%
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pandas as pd

# ---------------------------
# Mapping Setup (from resunet_final.py)
# ---------------------------
# Load CSVs (adjust paths as needed)
nc12_df = pd.read_csv(r"C:\Users\ChloeAtherton\Capstone\nc12.csv")
nc13_df = pd.read_csv(r"C:\Users\ChloeAtherton\Capstone\nc13.csv")
nc16_df = pd.read_csv(r"C:\Users\ChloeAtherton\Capstone\nc16.csv")

# Merge and generate new GT labels
merged_classes = pd.concat([nc12_df, nc13_df, nc16_df], ignore_index=True)
merged_classes.drop(columns=['class'], inplace=True)
merged_classes.drop_duplicates(inplace=True)
merged_classes = merged_classes.reset_index(drop=True)
merged_classes['gt label'] = merged_classes.index

# Create mapping dictionaries for each dataset.
nc12_mapping_dict = dict(zip(
    pd.merge(nc12_df, merged_classes, on='label', how='left')['class'],
    pd.merge(nc12_df, merged_classes, on='label', how='left')['gt label']
))
nc13_mapping_dict = dict(zip(
    pd.merge(nc13_df, merged_classes, on='label', how='left')['class'],
    pd.merge(nc13_df, merged_classes, on='label', how='left')['gt label']
))
nc16_mapping_dict = dict(zip(
    pd.merge(nc16_df, merged_classes, on='label', how='left')['class'],
    pd.merge(nc16_df, merged_classes, on='label', how='left')['gt label']
))

# ---------------------------
# Helper Functions
# ---------------------------
def load_hdf5_mat_file(file_path):
    import h5py
    try:
        data = h5py.File(file_path, 'r')
        hsi_data = np.array(data['HSI'])  # shape: (270, H, W)
        gt_data = np.array(data['GT'])    # shape: (H, W)
        print(f"Loaded file: {file_path}")
        print(f"HSI shape: {hsi_data.shape}, GT shape: {gt_data.shape}")
        return hsi_data, gt_data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def ScaleData(X, min_value, max_value):
    X = X.astype(np.float32)
    X -= min_value
    X /= (max_value - min_value)
    return X

def extract_overlapping_patches(image, patch_size, stride):
    from einops import rearrange
    import torch
    image = torch.tensor(image)
    if len(image.shape) > 2:
        patches = image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
        patches = rearrange(patches, "c h w p1 p2 -> (h w) c p1 p2")
    else:
        patches = image.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
        patches = rearrange(patches, "h w p1 p2 -> (h w) p1 p2")
    return patches.numpy()

def split_into_tiles(hsi_data, gt_data, tile_size=32, overlapping=False):
    """
    Split HSI and GT data into tiles.
    Assumes GT has already been remapped.
    """
    hsi_tiles = []
    gt_tiles = []
    
    # Scale HSI data between 0 and 1.
    hsi_data = ScaleData(hsi_data, np.min(hsi_data), np.max(hsi_data))
    
    # For consistency with your original pipeline, set background to 255 and subtract 1 from non-background.
    # (This step is done after remapping; adjust if necessary.)
    gt_data[gt_data == 0] = 255
    gt_data[gt_data != 255] -= 1

    h = hsi_data.shape[1]
    w = hsi_data.shape[2]
    pad_h = tile_size * ((h // tile_size) + 1) - h
    pad_w = tile_size * ((w // tile_size) + 1) - w

    hsi_data = np.pad(hsi_data, ((0,0), (0, pad_h), (0, pad_w)), mode='constant')
    gt_data = np.pad(gt_data, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=255)
    
    if overlapping:
        stride = 16
        hsi_tiles = extract_overlapping_patches(hsi_data, tile_size, stride)
        gt_tiles = extract_overlapping_patches(gt_data, tile_size, stride)
        filt = np.sum(gt_tiles < 100, axis=(1,2))
        a = filt > ((tile_size**2) * 0.3)
        hsi_tiles = hsi_tiles[a]
        gt_tiles = gt_tiles[a]
    else:
        rows, cols = gt_data.shape
        for i in range(0, rows, tile_size):
            for j in range(0, cols, tile_size):
                hsi_tile = hsi_data[:, i:i+tile_size, j:j+tile_size]
                gt_tile = gt_data[i:i+tile_size, j:j+tile_size]
                if hsi_tile.shape[1:] == (tile_size, tile_size) and gt_tile.shape == (tile_size, tile_size) and np.any(gt_tile != 255):
                    hsi_tiles.append(hsi_tile)
                    gt_tiles.append(gt_tile)
    return np.array(hsi_tiles), np.array(gt_tiles)

def process_multiple_hsi_files(folder_path, tile_size=32):
    """
    Process all .mat files in folder_path, applying remapping for each file based on its name,
    then split the data into tiles.
    """
    combined_hsi_tiles = []
    combined_gt_tiles = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mat'):
            file_path = os.path.join(folder_path, file_name)
            hsi_data, gt_data = load_hdf5_mat_file(file_path)
            if hsi_data is None or gt_data is None:
                continue
            # Apply remapping using the appropriate mapping dictionary.
            if file_name == "NC12.mat":
                gt_data = pd.DataFrame(gt_data).replace(nc12_mapping_dict).to_numpy()
            elif file_name == "NC13.mat":
                gt_data = pd.DataFrame(gt_data).replace(nc13_mapping_dict).to_numpy()
            elif file_name == "NC16.mat":
                gt_data = pd.DataFrame(gt_data).replace(nc16_mapping_dict).to_numpy()
            # Split into tiles (using overlapping mode as before).
            hsi_tiles, gt_tiles = split_into_tiles(hsi_data, gt_data, tile_size=tile_size, overlapping=True)
            combined_hsi_tiles.extend(hsi_tiles)
            combined_gt_tiles.extend(gt_tiles)
    combined_hsi_tiles = np.array(combined_hsi_tiles)
    combined_gt_tiles = np.array(combined_gt_tiles)
    print(f"Total number of HSI tiles: {len(combined_hsi_tiles)}")
    print(f"Total number of GT tiles: {len(combined_gt_tiles)}")
    return combined_hsi_tiles, combined_gt_tiles

# ---------------------------
# Main Offline Script
# ---------------------------
folder_path = r"C:\Users\ChloeAtherton\Capstone\data\mat_data"  # Folder containing your .mat files
tile_size = 32
num_patches = 10  # Number of test patches to save
output_file = r"C:\Users\ChloeAtherton\Capstone\data\cached_test_patches.npz"

# Process the .mat files to extract overlapping patches (with GT remapping)
combined_hsi_tiles, combined_gt_tiles = process_multiple_hsi_files(folder_path, tile_size=tile_size)
print(f"Extracted {len(combined_hsi_tiles)} total patches from the folder.")

# Split the combined patches into train and test sets (we only keep the test patches)
_, hsi_test, _, gt_test = train_test_split(
    combined_hsi_tiles, combined_gt_tiles, test_size=0.2, random_state=42
)
print(f"Test set contains {len(hsi_test)} patches before selection.")

# Select only the first num_patches patches.
if len(hsi_test) < num_patches:
    print("Warning: Fewer test patches available than desired.")
    num_patches = len(hsi_test)

hsi_test = hsi_test[:num_patches]
gt_test = gt_test[:num_patches]

# Save the selected test patches to an .npz file.
np.savez(output_file, HSI=hsi_test, GT=gt_test)
print("Cached test patches saved to", output_file)


# %%
import numpy as np
import h5py
import os
import pandas as pd

# ---------------------------------------------------
# Mapping Setup (from resunet_final.py)
# ---------------------------------------------------
nc12_df = pd.read_csv(r"C:\Users\ChloeAtherton\Capstone\nc12.csv")
nc13_df = pd.read_csv(r"C:\Users\ChloeAtherton\Capstone\nc13.csv")
nc16_df = pd.read_csv(r"C:\Users\ChloeAtherton\Capstone\nc16.csv")

# Merge the CSVs and create a unified mapping with new numeric GT labels.
merged_classes = pd.concat([nc12_df, nc13_df, nc16_df], ignore_index=True)
merged_classes.drop(columns=['class'], inplace=True)
merged_classes.drop_duplicates(inplace=True)
merged_classes = merged_classes.reset_index(drop=True)
merged_classes['gt label'] = merged_classes.index

# Create mapping dictionaries for each dataset.
nc12_mapping_dict = dict(zip(
    pd.merge(nc12_df, merged_classes, on='label', how='left')['class'],
    pd.merge(nc12_df, merged_classes, on='label', how='left')['gt label']
))
nc13_mapping_dict = dict(zip(
    pd.merge(nc13_df, merged_classes, on='label', how='left')['class'],
    pd.merge(nc13_df, merged_classes, on='label', how='left')['gt label']
))
nc16_mapping_dict = dict(zip(
    pd.merge(nc16_df, merged_classes, on='label', how='left')['class'],
    pd.merge(nc16_df, merged_classes, on='label', how='left')['gt label']
))

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def load_hdf5_mat_file(file_path):
    """Load full image data from a MATLAB v7.3 file using h5py."""
    try:
        with h5py.File(file_path, 'r') as f:
            hsi_data = np.array(f['HSI'])  # shape: (270, H, W)
            gt_data = np.array(f['GT'])    # shape: (H, W)
        print(f"Loaded file: {file_path}")
        print(f"HSI shape: {hsi_data.shape}, GT shape: {gt_data.shape}")
        return hsi_data, gt_data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def ScaleData(X, min_value, max_value):
    """Scale data between 0 and 1."""
    X = X.astype(np.float32)
    X -= min_value
    X /= (max_value - min_value)
    return X

def process_full_image(hsi_data, gt_data, mapping_dict):
    """
    Scale and pad the full image, applying the GT remapping.
    
    Parameters:
      hsi_data: raw hyperspectral image array.
      gt_data: raw ground truth array.
      mapping_dict: a dictionary mapping original GT values to new labels.
      
    Returns:
      Processed hsi_data and gt_data.
    """
    # Scale HSI data between 0 and 1.
    min_val = np.min(hsi_data)
    max_val = np.max(hsi_data)
    hsi_data = (hsi_data - min_val) / (max_val - min_val)
    
    # Remap ground truth using the provided mapping.
    # This replaces each original GT value with its new label.
    gt_data = pd.DataFrame(gt_data).replace(mapping_dict).to_numpy()
    
    # Pad image so dimensions are multiples of 8.
    multiple_required = 8
    h = hsi_data.shape[1]
    w = hsi_data.shape[2]
    pad_h = multiple_required * ((h // multiple_required) + 1) - h
    pad_w = multiple_required * ((w // multiple_required) + 1) - w
    hsi_data = np.pad(hsi_data, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
    gt_data = np.pad(gt_data, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=255)
    return hsi_data, gt_data

def save_processed_full_image(input_file, output_file):
    """
    Load a .mat file, process the full image (scaling, GT remapping, padding),
    and save it as a compressed npz file.
    The appropriate GT mapping is selected based on the input file name.
    """
    hsi_data, gt_data = load_hdf5_mat_file(input_file)
    if hsi_data is None or gt_data is None:
        print("Failed to load input file.")
        return
    
    # Choose the appropriate mapping dictionary based on the file name.
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



# %%


# %%
# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------
input_file = r"C:\Users\ChloeAtherton\Capstone\data\mat_data\NC16.mat"  # Adjust as needed.
output_file = r"C:\Users\ChloeAtherton\Capstone\data\NC16_processed.npz"
save_processed_full_image(input_file, output_file)



