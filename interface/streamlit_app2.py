import streamlit as st
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import ResUNet
from dataloader import SingleImageDataset  # used for full image inference
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import os

# -----------------------
# Model and Utility Functions
# -----------------------

def load_model():
    save_file = r'C:\Users\ChloeAtherton\Capstone\models\weights_resunet_test.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUNet(in_channels=270, num_classes=17).to(device)
    model.load_state_dict(torch.load(save_file))
    st.success("Model loaded!")
    time.sleep(1)
    return model


def load_processed_full_image(npz_file):
    """
    Load a preprocessed full image from an .npz file.
    The file should contain keys:
      - 'HSI': full image tensor (shape: 270 x H x W)
      - 'GT':  ground truth (shape: H x W)
    """
    data = np.load(npz_file, allow_pickle=True)
    hsi_data = data['HSI']
    gt_data = data['GT']
    return hsi_data, gt_data

def load_cached_test_patches(npz_file):
    """
    Load precomputed test patches from an .npz file.
    The file should contain keys:
      - 'HSI': an array of shape (N, 270, 32, 32)
      - 'GT':  an array of shape (N, 32, 32)
    """
    data = np.load(npz_file, allow_pickle=True)
    hsi_test = data['HSI']
    gt_test = data['GT']
    return hsi_test, gt_test

def run_inference_on_processed_image(npz_file, model):
    """
    Run inference on a full image loaded from a cached npz file.
    Since the image is preprocessed offline (scaled and padded), we can load it directly.
    """
    device = next(model.parameters()).device
    hsi_data, gt_data = load_processed_full_image(npz_file)
    
    # Create a dataset and DataLoader for the full image.
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

def run_inference_on_cached_test_patch(npz_file, model, patch_index):
    """
    Run inference on a single test patch loaded from a cached test patches file.
    """
    device = next(model.parameters()).device
    hsi_test, gt_test = load_cached_test_patches(npz_file)
    if patch_index >= len(hsi_test):
        raise ValueError(f"Patch index {patch_index} out of range (max index {len(hsi_test)-1}).")
    patch_hsi = hsi_test[patch_index]  # shape: (270, 32, 32)
    patch_gt = gt_test[patch_index]    # shape: (32, 32)
    
    patch_hsi = torch.tensor(patch_hsi, dtype=torch.float32).unsqueeze(0).to(device)
    patch_gt = torch.tensor(patch_gt, dtype=torch.long).unsqueeze(0).to(device)
    
    model.eval()
    start_time = time.time()
    with torch.no_grad():
         outputs = model(patch_hsi)
         preds = torch.argmax(outputs, dim=1)
         preds[patch_gt == 255] = 255
    compute_time = time.time() - start_time

    valid_mask = patch_gt != 255
    pixel_accuracy = (preds[valid_mask] == patch_gt[valid_mask]).float().mean().item() if valid_mask.sum() > 0 else 0.0

    image_2d = patch_hsi[0].mean(dim=0).cpu().numpy()
    ground_truth = patch_gt[0].cpu().numpy()
    pred_mask = preds[0].cpu().numpy()

    return compute_time, pixel_accuracy, image_2d, ground_truth, pred_mask

def plot_results(image, ground_truth, pred_mask):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Input Image (2D Projection)")
    axs[0].axis("off")
    axs[1].imshow(ground_truth, cmap='nipy_spectral', vmin=0, vmax=16)
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")
    axs[2].imshow(pred_mask, cmap='nipy_spectral', vmin=0, vmax=16)
    axs[2].set_title("Predicted Mask")
    axs[2].axis("off")
    st.pyplot(fig)

# -----------------------
# Streamlit Interface
# -----------------------

def main():
    st.title("Object Based Classification of Drone Hyperspectral Imagery for Wetland Mapping")
    st.subheader("Capstone Project by Chloe, Maddy, Caroline and Will")
    st.markdown("""
    ### Choose an Inference Mode
    - **Full Image**: Run inference on a preprocessed full image loaded from a cached npz file.
    - **Test Patch**: Run inference on a 32x32 patch loaded from a cached test patches npz file.
    """)
    
    mode = st.radio("Select Inference Mode", ["Full Image", "Test Patch"])
    
    # File paths for cached processed full images.
    full_image_files = {
        "Image 1: NC12": r"C:\Users\ChloeAtherton\Capstone\data\NC12_processed.npz",
        "Image 2: NC13": r"C:\Users\ChloeAtherton\Capstone\data\NC13_processed.npz",
        "Image 3: NC16": r"C:\Users\ChloeAtherton\Capstone\data\NC16_processed.npz"
    }
    
    # Cached test patches file.
    cached_test_file = r"C:\Users\ChloeAtherton\Capstone\data\cached_test_patches.npz"
    
    try:
        model = load_model()
    except Exception as e:
             st.error(f"Error loading model: {e}")
             return
             
    if mode == "Full Image":
         selected_label = st.selectbox("Select a Full Image", list(full_image_files.keys()))
         selected_filepath = full_image_files[selected_label]
         if st.button("Run Inference on Full Image"):
              with st.spinner("Running inference on full image..."):
                  try:
                      compute_time, accuracy, image, ground_truth, pred_mask = run_inference_on_processed_image(selected_filepath, model)
                  except Exception as e:
                      st.error(f"Error during inference: {e}")
                      return
              st.success("Inference completed!")
              st.write(f"**Compute Time:** {compute_time:.2f} seconds")
              st.write(f"**Pixel Accuracy:** {accuracy:.2f}")
              plot_results(image, ground_truth, pred_mask)
    else:
         patch_index = st.selectbox("Select Test Patch Index", list(range(10)), format_func=lambda i: f"Test Patch {i+1}")
         if st.button("Run Inference on Test Patch"):
              with st.spinner("Running inference on test patch..."):
                  try:
                      compute_time, accuracy, image, ground_truth, pred_mask = run_inference_on_cached_test_patch(cached_test_file, model, patch_index)
                  except Exception as e:
                      st.error(f"Error during inference: {e}")
                      return
              st.success("Inference completed!")
              st.write(f"**Compute Time:** {compute_time:.2f} seconds")
              st.write(f"**Pixel Accuracy:** {accuracy:.2f}")
              plot_results(image, ground_truth, pred_mask)
    
if __name__ == "__main__":
    main()
