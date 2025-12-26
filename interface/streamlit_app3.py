import streamlit as st
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import os

# -------------------------------------------------
# MAPPING SETUP
# -------------------------------------------------
nc12_df = pd.read_csv(r"C:\Users\ChloeAtherton\Capstone\nc12.csv")
nc13_df = pd.read_csv(r"C:\Users\ChloeAtherton\Capstone\nc13.csv")
nc16_df = pd.read_csv(r"C:\Users\ChloeAtherton\Capstone\nc16.csv")

# 1) Merge the CSVs
merged_classes = pd.concat([nc12_df, nc13_df, nc16_df], ignore_index=True)

# 2) Drop 'class' column, then drop duplicates *by label only*
merged_classes.drop(columns=['class'], inplace=True, errors='ignore')
merged_classes.drop_duplicates(subset=['label'], inplace=True)
merged_classes.reset_index(drop=True, inplace=True)

# 3) Force it down to 19 rows if there's more than 19
'''
if len(merged_classes) > 19:
    merged_classes = merged_classes.iloc[:19].copy()
merged_classes.reset_index(drop=True, inplace=True)'
'''

# 4) Assign new numeric labels 0..18
merged_classes['gt label'] = merged_classes.index
number_classes = len(merged_classes)
print("Final merged_classes:\n", merged_classes)
print("number_classes =", number_classes)

# 5) Rebuild the mapping dictionaries.
#    We re-merge only on ['label','gt label'] to avoid reintroducing duplicates.
nc12_mapping = pd.merge(
    nc12_df[['label','class']],
    merged_classes[['label','gt label']],
    on='label', how='left'
)
nc13_mapping = pd.merge(
    nc13_df[['label','class']],
    merged_classes[['label','gt label']],
    on='label', how='left'
)
nc16_mapping = pd.merge(
    nc16_df[['label','class']],
    merged_classes[['label','gt label']],
    on='label', how='left'
)

# For each dataset, map old 'class' -> new 'gt label'
nc12_mapping_dict = dict(zip(nc12_mapping['class'], nc12_mapping['gt label']))
nc13_mapping_dict = dict(zip(nc13_mapping['class'], nc13_mapping['gt label']))
nc16_mapping_dict = dict(zip(nc16_mapping['class'], nc16_mapping['gt label']))
# -------------------------------------------------
# Utility Functions for Remapping
# -------------------------------------------------
def remap_gt(gt, mapping_dict):
    """
    Remap ground truth using the provided mapping dictionary.
    The ignore label (255) is preserved.
    """
    gt_remapped = np.copy(gt)
    for old_val, new_val in mapping_dict.items():
        # Only update pixels that are not the ignore label
        gt_remapped[(gt == old_val) & (gt != 255)] = new_val
    return gt_remapped

# -------------------------------------------------
# Data Loading Functions
# -------------------------------------------------
def load_processed_full_image(npz_file, mapping_dict):
    """
    Load a preprocessed full image from an .npz file and remap ground truth.
    The .npz file should have keys:
      - 'HSI': full image tensor (shape: 270 x H x W)
      - 'GT':  ground truth (shape: H x W)
    """
    data = np.load(npz_file, allow_pickle=True)
    hsi_data = data['HSI']
    gt_data = data['GT']
    gt_data = remap_gt(gt_data, mapping_dict)
    return hsi_data, gt_data

def load_cached_test_patches(npz_file, mapping_dict):
    """
    Load cached test patches from an .npz file and remap ground truth.
    The .npz file should have keys:
      - 'HSI': array of shape (N, 270, 32, 32)
      - 'GT':  array of shape (N, 32, 32)
    """
    data = np.load(npz_file, allow_pickle=True)
    hsi_test = data['HSI']
    gt_test = data['GT']
    gt_test = remap_gt(gt_test, mapping_dict)
    return hsi_test, gt_test

# -------------------------------------------------
# Model Definition and Loading
# -------------------------------------------------
# Import your model (assumes that model.py defines ResUNet)
from model import ResUNet

def load_model():
    save_file = r"C:\Users\ChloeAtherton\Capstone\weights_resunet_new_100.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Now use the new number of classes (from merged mapping)
    model = ResUNet(in_channels=270, num_classes=number_classes).to(device)
    model.load_state_dict(torch.load(save_file, map_location=device))
    st.success("Model loaded!")
    time.sleep(1)
    return model

# A simple dataset for single full-image inference.
class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, image, mask):
        self.image = torch.tensor(image, dtype=torch.float32)  # (270, H, W)
        self.mask = torch.tensor(mask, dtype=torch.long)       # (H, W)
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return self.image, self.mask

# -------------------------------------------------
# Inference Functions
# -------------------------------------------------
def run_inference_on_processed_image(npz_file, model, mapping_dict):
    """
    Run inference on a full image loaded from a processed .npz file.
    """
    device = next(model.parameters()).device
    hsi_data, gt_data = load_processed_full_image(npz_file, mapping_dict)
    dataset = SingleImageDataset(hsi_data, gt_data)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            preds[masks == 255] = 255
    compute_time = time.time() - start_time

    valid_mask = masks != 255
    pixel_accuracy = (preds[valid_mask] == masks[valid_mask]).float().mean().item() if valid_mask.sum() > 0 else 0.0

    image_2d = images[0].mean(dim=0).cpu().numpy()
    ground_truth = masks[0].cpu().numpy()
    pred_mask = preds[0].cpu().numpy()
    
    return compute_time, pixel_accuracy, image_2d, ground_truth, pred_mask

def run_inference_on_cached_test_patch(npz_file, model, patch_index, mapping_dict):
    """
    Run inference on a single test patch from cached test patches.
    """
    device = next(model.parameters()).device
    hsi_test, gt_test = load_cached_test_patches(npz_file, mapping_dict)
    if patch_index >= len(hsi_test):
        raise ValueError(f"Patch index {patch_index} out of range (max index {len(hsi_test)-1}).")
    patch_hsi = hsi_test[patch_index]  # shape: (270, 32, 32)
    patch_gt = gt_test[patch_index]    # shape: (32, 32)
    
    patch_hsi = torch.tensor(patch_hsi, dtype=torch.float32).unsqueeze(0).to(device)
    patch_gt_tensor = torch.tensor(patch_gt, dtype=torch.long).unsqueeze(0).to(device)
    
    model.eval()
    start_time = time.time()
    with torch.no_grad():
         outputs = model(patch_hsi)
         preds = torch.argmax(outputs, dim=1)  # shape: (1, 32, 32)
         preds = preds.squeeze(0)  # Now shape is (32, 32)
         preds[patch_gt_tensor.squeeze(0) == 255] = 255
    compute_time = time.time() - start_time

    valid_mask = patch_gt_tensor.squeeze(0) != 255
    pixel_accuracy = (preds[valid_mask] == patch_gt_tensor.squeeze(0)[valid_mask]).float().mean().item() if valid_mask.sum() > 0 else 0.0

    image_2d = patch_hsi[0].mean(dim=0).cpu().numpy()
    ground_truth = patch_gt_tensor.squeeze(0).cpu().numpy()
    pred_mask = preds.cpu().numpy()

    return compute_time, pixel_accuracy, image_2d, ground_truth, pred_mask

def plot_results(image, ground_truth, pred_mask):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Input Image (2D Projection)")
    axs[0].axis("off")
    axs[1].imshow(ground_truth, cmap='nipy_spectral', vmin=0, vmax=number_classes-1)
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")
    axs[2].imshow(pred_mask, cmap='nipy_spectral', vmin=0, vmax=number_classes-1)
    axs[2].set_title("Predicted Mask")
    axs[2].axis("off")
    st.pyplot(fig)

# -------------------------------------------------
# Streamlit App Interface
# -------------------------------------------------
def main():
    st.title("Object Based Classification of Drone Hyperspectral Imagery for Wetland Mapping")
    st.subheader("Capstone Project by Chloe, Maddy, Caroline and Will")
    st.markdown("""
    ### Choose an Inference Mode
    - **Full Image**: Run inference on a preprocessed full image (from a .npz file).
    - **Test Patch**: Run inference on a 32x32 patch (from cached test patches).
    """)
    
    mode = st.radio("Select Inference Mode", ["Full Image", "Test Patch"])
    
    # For full image mode, include the mapping dictionary with each file.
    full_image_files = {
        "Image 1: NC12": {
            "npz": r"C:\Users\ChloeAtherton\Capstone\data\NC12_processed.npz",
            "mapping": nc12_mapping_dict
        },
        "Image 2: NC13": {
            "npz": r"C:\Users\ChloeAtherton\Capstone\data\NC13_processed.npz",
            "mapping": nc13_mapping_dict
        },
        "Image 3: NC16": {
            "npz": r"C:\Users\ChloeAtherton\Capstone\data\NC16_processed.npz",
            "mapping": nc16_mapping_dict
        }
    }
    
    # For test patches, allow the user to select the mapping.
    cached_test_file = r"C:\Users\ChloeAtherton\Capstone\data\cached_test_patches.npz"
    mapping_files = {
        "NC12 Mapping": nc12_mapping_dict,
        "NC13 Mapping": nc13_mapping_dict,
        "NC16 Mapping": nc16_mapping_dict
    }
    print(number_classes)
    
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
             
    if mode == "Full Image":
         selected_label = st.selectbox("Select a Full Image", list(full_image_files.keys()))
         file_info = full_image_files[selected_label]
         if st.button("Run Inference on Full Image"):
              with st.spinner("Running inference on full image..."):
                  try:
                      compute_time, accuracy, image, gt, pred = run_inference_on_processed_image(file_info["npz"], model, file_info["mapping"])
                  except Exception as e:
                      st.error(f"Error during inference: {e}")
                      return
              st.success("Inference completed!")
              st.write(f"**Compute Time:** {compute_time:.2f} seconds")
              st.write(f"**Pixel Accuracy:** {accuracy:.2f}")
              plot_results(image, gt, pred)
    else:
         patch_index = st.selectbox("Select Test Patch Index", list(range(10)), format_func=lambda i: f"Test Patch {i+1}")
         selected_mapping_key = st.selectbox("Select Mapping for Test Patches", list(mapping_files.keys()))
         mapping_dict = mapping_files[selected_mapping_key]
         if st.button("Run Inference on Test Patch"):
              with st.spinner("Running inference on test patch..."):
                  try:
                      compute_time, accuracy, image, gt, pred = run_inference_on_cached_test_patch(cached_test_file, model, patch_index, mapping_dict)
                  except Exception as e:
                      st.error(f"Error during inference: {e}")
                      return
              st.success("Inference completed!")
              st.write(f"**Compute Time:** {compute_time:.2f} seconds")
              st.write(f"**Pixel Accuracy:** {accuracy:.2f}")
              plot_results(image, gt, pred)
    
if __name__ == "__main__":
    main()
