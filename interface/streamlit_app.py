import streamlit as st
import time
import tempfile
import scipy.io as sio
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import ResUNet
from dataloader import SingleImageDataset
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import h5py
# -----------------------
# Replace these placeholder functions with your actual functions from your notebook.
# For example, if you have already defined:
#    - load_model() that loads your UNet model with pretrained weights
#    - run_inference() that processes the .mat file input and returns predictions
#    - compute_pixel_accuracy() that computes the pixel accuracy between the prediction and ground truth
# then import and use those functions here.
# -----------------------

def load_model():
    # Replace with your actual model loading code.
    # For example:
    # model = UNetModel()
    # model.load_state_dict(torch.load("pretrained_weights.pth", map_location=torch.device('cpu')))
    save_file = r'C:\Users\ChloeAtherton\Capstone\models\weights_resunet_test.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    model = ResUNet(in_channels=270, num_classes=17).to(device)
    model.load_state_dict(torch.load(save_file))
    # return model
    st.success("Model load complete!")
    time.sleep(1)
      # Dummy model, replace with your actual model.
    return model


def load_hdf5_mat_file(file_path):
    """
    Load a .mat file and extract hyperspectral image and ground truth data.

    Args:
        file_path (str): Path to the .mat file.

    Returns:
        np.ndarray, np.ndarray: HSI and GT data.
    """
    try:
        data = h5py.File(file_path, 'r')
        hsi_data = np.array(data['HSI'])  # Hyperspectral image data
        gt_data = np.array(data['GT'])   # Ground truth data
        print(f"Loaded file: {file_path}")
        print(f"HSI shape: {hsi_data.shape}, GT shape: {gt_data.shape}")
        return hsi_data, gt_data
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return None, None

def ScaleData(X,min_value,max_value):
  #This function scales the data 'X' between 0 and 1
  min_value=np.float16(min_value)
  max_value=np.float16(max_value)
  X -= min_value
  X /= (max_value - min_value)
  return X


# Run inference on a single image from the uploaded .mat file.
def run_inference_on_image(mat_filepath, model):
    """
    Loads the .mat file provided by the user, preprocesses the data, runs inference on the single image,
    and returns the compute time, pixel accuracy, and data for visualization.

    Assumes the .mat file has keys 'HSI' for the hyperspectral image and 'GT' for the ground truth.
    """
    # Use the device from the model
    device = next(model.parameters()).device

    # Load the .mat file (expects keys 'HSI' and 'GT')
    hsi_data, gt_data = load_hdf5_mat_file(mat_filepath)
    if hsi_data is None or gt_data is None:
        raise ValueError("Error loading .mat file.")

    # Scale the hyperspectral data between 0 and 1
    hsi_data = ScaleData(hsi_data, np.min(hsi_data), np.max(hsi_data))

    # Process ground truth: set zeros as background (255) and subtract 1 from non-background labels.
    gt_data[gt_data == 0] = 255
    gt_data[gt_data != 255] -= 1

    # Pad data so that the spatial dimensions are multiples of a given value (e.g., 8)
    multiple_required = 8
    h = hsi_data.shape[1]
    w = hsi_data.shape[2]
    pad_h = multiple_required * (h // multiple_required + 1) - h
    pad_w = multiple_required * (w // multiple_required + 1) - w
    hsi_data = np.pad(hsi_data, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
    gt_data = np.pad(gt_data, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=255)

    # Create a dataset and dataloader for the single image.
    dataset = SingleImageDataset(hsi_data, gt_data)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Run inference and measure compute time.
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)  # outputs shape: (1, num_classes, H, W)
            preds = torch.argmax(outputs, dim=1)
            # Set pixels to background where ground truth is background.
            preds[masks == 255] = 255
    compute_time = time.time() - start_time

    # Compute pixel accuracy ignoring background pixels (where GT == 255)
    valid_mask = masks != 255
    if valid_mask.sum() > 0:
        pixel_accuracy = (preds[valid_mask] == masks[valid_mask]).float().mean().item()
    else:
        pixel_accuracy = 0.0

    # For visualization:
    # - Create a 2D projection of the input image by averaging across the spectral channels.
    # - Retrieve the ground truth and prediction from the tensors.
    image_2d = images[0].mean(dim=0).cpu().numpy()
    ground_truth = masks[0].cpu().numpy()
    pred_mask = preds[0].cpu().numpy()

    return compute_time, pixel_accuracy, image_2d, ground_truth, pred_mask

# Function to display results in a plot.
def plot_results(image, ground_truth, pred_mask):
    """
    Create a plot comparing the input image, the ground truth, and the predicted mask.
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Input Image (2D Projection)')
    axs[0].axis('off')
    
    axs[1].imshow(ground_truth, cmap='nipy_spectral', vmin=0, vmax=16)
    axs[1].set_title('Ground Truth')
    axs[1].axis('off')
    
    axs[2].imshow(pred_mask, cmap='nipy_spectral', vmin=0, vmax=16)
    axs[2].set_title('Predicted Mask')
    axs[2].axis('off')
    
    st.pyplot(fig)

# -----------------------
# Streamlit Interface
# -----------------------

# Streamlit Interface
def main():
    st.title("Segmentation Inference for Hyperspectral Data for Wetlands Mapping")
    st.subheader("Capstone Project by Chloe, Maddy, Caroline and Will")
    st.markdown(
        """
        Select a **.mat** file containing your hyperspectral image and ground truth data from the dropdown menu to run inference.
        The interface will display the compute time, pixel accuracy, and a comparison plot.
        """
    )
    
    # Define a dictionary of file labels to file paths.
    file_paths = {
        "Image 1: NC12": r"C:\Users\ChloeAtherton\Capstone\data\NC12.mat",
        "Image 2: NC13": r"C:\Users\ChloeAtherton\Capstone\data\NC13.mat",
        "Image 3: NC16": r"C:\Users\ChloeAtherton\Capstone\data\NC16.mat"
    }
    
    # Dropdown menu to select one of the file paths.
    selected_label = st.selectbox("Select a .mat file", list(file_paths.keys()))
    selected_filepath = file_paths[selected_label]
    
    # Load the pretrained model.
    with st.spinner("Loading model..."):
        try:
            model = load_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

    # Button to run inference.
    if st.button("Run Inference"):
        with st.spinner("Running inference..."):
            try:
                compute_time, accuracy, image, ground_truth, pred_mask = run_inference_on_image(selected_filepath, model)
            except Exception as e:
                st.error(f"Error during inference: {e}")
                return
            
            st.success("Inference completed!")
            st.write(f"**Compute Time:** {compute_time:.2f} seconds")
            st.write(f"**Pixel Accuracy:** {accuracy:.2f}")
            # Display the results plot.
            plot_results(image, ground_truth, pred_mask)

if __name__ == "__main__":
    main()
