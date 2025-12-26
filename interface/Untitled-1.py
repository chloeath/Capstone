# %%
# map gt values to their appropriate label

import pandas as pd
nc12_df = pd.read_csv(r"C:\Users\ChloeAtherton\Capstone\nc12.csv")
nc13_df = pd.read_csv(r"C:\Users\ChloeAtherton\Capstone\nc13.csv")
nc16_df = pd.read_csv(r"C:\Users\ChloeAtherton\Capstone\nc12.csv")
merged_classes = pd.concat([nc12_df, nc13_df, nc16_df], ignore_index=True)
merged_classes.drop(columns=['class'], inplace=True)
merged_classes.drop_duplicates(inplace=True)
merged_classes=merged_classes.reset_index(drop=True)
merged_classes['gt label'] = merged_classes.index
merged_classes

nc12_mapping = pd.merge(nc12_df, merged_classes, on='label', how='left')
nc13_mapping = pd.merge(nc13_df, merged_classes, on='label', how='left')
nc16_mapping = pd.merge(nc16_df, merged_classes, on='label', how='left')

nc12_mapping_dict = dict(zip(nc12_mapping['class'], nc12_mapping['gt label']))
nc13_mapping_dict = dict(zip(nc13_mapping['class'], nc13_mapping['gt label']))
nc16_mapping_dict = dict(zip(nc16_mapping['class'], nc16_mapping['gt label']))

number_classes = len(merged_classes)


# %%
merged_classes

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim


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



# %%
import os

# %%
def ScaleData(X,min_value,max_value):
  #This function scales the data 'X' between 0 and 1
  min_value=np.float16(min_value)
  max_value=np.float16(max_value)
  X -= min_value
  X /= (max_value - min_value)
  return X

# %%
from einops import rearrange

def extract_overlapping_patches(image, patch_size, stride):
    """
    Extract overlapping patches from an image.

    Args:
        image (torch.Tensor): Input image of shape [C, H, W].
        patch_size (int): Size of each patch (height and width).
        stride (int): Stride for extracting patches (controls overlap).

    Returns:
        patches (torch.Tensor): Extracted patches of shape [num_patches, C, patch_size, patch_size].
    """
    image = torch.tensor(image)
    # Unfold the image into patches
    if len(image.shape)>2:
        patches = image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)

        # Reshape to [num_patches, C, patch_size, patch_size]
        patches = rearrange(patches,"c h w p1 p2 -> (h w) c p1 p2")
        # patches = patches.contiguous().view(-1, image.size(0), patch_size, patch_size)
    else:

        patches = image.unfold(0, patch_size, stride).unfold(1, patch_size, stride)

        # Reshape to [num_patches, C, patch_size, patch_size]
        patches = rearrange(patches," h w p1 p2 -> (h w) p1 p2")
        # patches = patches.contiguous().view(-1, patch_size, patch_size)

    return patches.numpy()

# %%
def split_into_tiles(hsi_data, gt_data, tile_size=32, overlapping=False):
    """
    Split HSI and GT data into smaller tiles of size (tile_size x tile_size).

    Args:
        hsi_data (np.ndarray): Hyperspectral image data.
        gt_data (np.ndarray): Ground truth data.
        tile_size (int): Size of the tiles (default: 32x32).

    Returns:
        list, list: HSI tiles and corresponding GT tiles.
    """
    hsi_tiles = []
    gt_tiles = []

    # scale hsi data between 0 and 1
    hsi_data = ScaleData(hsi_data, np.min(hsi_data), np.max(hsi_data))

    gt_data[gt_data == 0] = 255
    gt_data[gt_data != 255] -= 1

    # pad data to allow all image data to be included
    h = hsi_data.shape[1]
    w = hsi_data.shape[2]
    pad_h = tile_size * (h//tile_size + 1) - h  # Padding needed for height
    pad_w = tile_size * (w//tile_size + 1) - w  # Padding needed for width

    hsi_data = np.pad(hsi_data, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
    gt_data = np.pad(gt_data, ((0, pad_h), (0, pad_w)), mode='constant',constant_values=255)

    if overlapping == True:
      stride=16
      hsi_tiles = extract_overlapping_patches(hsi_data, tile_size, stride)
      gt_tiles = extract_overlapping_patches(gt_data, tile_size, stride)

      filt=np.sum(gt_tiles<100,axis=(1,2))
      a = filt>((tile_size**2)*0.3) # filter data to include only patches with >30% labelled (non-background) data

      hsi_tiles = hsi_tiles[a]
      gt_tiles = gt_tiles[a]

    else:
      rows, cols = gt_data.shape
      for i in range(0, rows, tile_size):
          for j in range(0, cols, tile_size):
              hsi_tile = hsi_data[:, i:i + tile_size, j:j + tile_size]
              gt_tile = gt_data[i:i + tile_size, j:j + tile_size]

              # Check if the tile matches the expected size
              if hsi_tile.shape[1:] == (tile_size, tile_size) and gt_tile.shape == (tile_size, tile_size) and np.any(gt_tile != 255):
                  hsi_tiles.append(hsi_tile)
                  gt_tiles.append(gt_tile)

    return hsi_tiles, gt_tiles


# %%
def process_multiple_hsi_files(folder_path, tile_size=32):
    """
    Process multiple .mat files in a folder to create a combined dataset.

    Args:
        folder_path (str): Path to the folder containing .mat files.
        tile_size (int): Size of the tiles (default: 32x32).

    Returns:
        np.ndarray, np.ndarray: Combined HSI tiles and GT tiles.
    """
    combined_hsi_tiles = []
    combined_gt_tiles = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mat'):
            file_path = os.path.join(folder_path, file_name)

            # Load the file
            hsi_data, gt_data = load_hdf5_mat_file(file_path)
            if hsi_data is None or gt_data is None:
                continue
            if file_name == "NC12.mat":
                gt_data = pd.DataFrame(gt_data).replace(nc12_mapping_dict).to_numpy()
            elif file_name == "NC13.mat":
                gt_data = pd.DataFrame(gt_data).replace(nc13_mapping_dict).to_numpy()
            elif file_name == "NC16.mat":
                gt_data = pd.DataFrame(gt_data).replace(nc16_mapping_dict).to_numpy()

            # Split into tiles
            hsi_tiles, gt_tiles = split_into_tiles(hsi_data, gt_data, tile_size, overlapping=True)
            combined_hsi_tiles.extend(hsi_tiles)
            combined_gt_tiles.extend(gt_tiles)

    # Convert to NumPy arrays
    combined_hsi_tiles = np.array(combined_hsi_tiles)
    combined_gt_tiles = np.array(combined_gt_tiles)

    print(f"Total number of HSI tiles: {len(combined_hsi_tiles)}")
    print(f"Total number of GT tiles: {len(combined_gt_tiles)}")

    return combined_hsi_tiles, combined_gt_tiles

# %%
folder_path = r"C:\Users\ChloeAtherton\Capstone\data\mat_data"  # Path to the folder containing .mat files
tile_size = 32

# Process all files in the folder
hsi_tiles, gt_tiles = process_multiple_hsi_files(folder_path, tile_size)

# Verify the combined dataset
print(f"Final dataset shape - HSI tiles: {hsi_tiles.shape}, GT tiles: {gt_tiles.shape}")

# %%
from sklearn.model_selection import train_test_split

# Split into train and test sets (80% train, 20% test)
hsi_train, hsi_test, gt_train, gt_test = train_test_split(
    hsi_tiles, gt_tiles, test_size=0.2, random_state=42
)

print(f"Train set: {len(hsi_train)} samples")
print(f"Test set: {len(hsi_test)} samples")


# %%
import torch
import torch.nn as nn

class ResUNet(nn.Module):
    def __init__(self, in_channels=270, num_classes=24):
        super(ResUNet, self).__init__()
        self.num_classes = num_classes

        # Residual block
        def res_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # Encoder with residual blocks
        self.enc1 = res_block(in_channels, 64)
        self.enc2 = res_block(64, 128)
        self.enc3 = res_block(128, 256)
        self.pool = nn.MaxPool2d(2)

        # 1x1 convolution for residual matching in encoders
        self.res1 = nn.Conv2d(in_channels, 64, kernel_size=1, bias=False)
        self.res2 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.res3 = nn.Conv2d(128, 256, kernel_size=1, bias=False)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = res_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = res_block(128, 64)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder with residual connections
        enc1 = self.enc1(x) + self.res1(x)  # Residual connection on input
        enc2 = self.enc2(self.pool(enc1)) + self.res2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2)) + self.res3(self.pool(enc2))

        # Decoder
        dec3 = self.dec3(torch.cat([self.up3(enc3), enc2], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc1], dim=1))

        return self.final(dec2)


# %%
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import random

class HyperspectralDataset(Dataset):
    def __init__(self, images, masks, hflip=None, vflip=None):
        self.images = images  # Hyperspectral tiles of shape (270, 32, 32)
        self.masks = masks    # Ground truth tiles of shape (32, 32)
        self.hflip = hflip
        self.vflip = vflip

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)  # (270, 32, 32)
        mask = torch.tensor(self.masks[idx], dtype=torch.long)      # (32, 32)
        if self.hflip:
            if random.random() < 0.5:
                image = v2.functional.horizontal_flip(image)
                mask = v2.functional.horizontal_flip(mask)
        if self.vflip:
            if random.random() < 0.5:
                image = v2.functional.vertical_flip(image)
                mask = v2.functional.vertical_flip(mask)
        return image, mask

# Split into training and testing
train_dataset = HyperspectralDataset(hsi_train, gt_train)
test_dataset = HyperspectralDataset(hsi_test, gt_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# %%
import torch
print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

unique_classes = torch.unique(torch.tensor(gt_tiles))
print(f"Unique classes in masks: {unique_classes}")
print(f"Number of classes: {len(unique_classes)}")

for images, masks in train_loader:
    print(f"Input shape: {images.shape}")
    print(f"Ground truth shape: {masks.shape}")
    break


# %%
# TRAINING SCRIPT

import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to the correct device
model = ResUNet(in_channels=270, num_classes=number_classes).to(device)
print(f"Model is on device: {next(model.parameters()).device}")

save_file = r"C:\Users\ChloeAtherton\Capstone\models\retrained_wednesday.pth"
plot_file = r"C:\Users\ChloeAtherton\Capstone\models\retrain_plot.png"

# Ensure criterion is defined
criterion = nn.CrossEntropyLoss(ignore_index = 255)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_and_validate_model(model, train_loader, test_loader, criterion, optimizer, save_file, plot_file, num_epochs=10):
    model.train()  # Set the model to training mode

    # Store losses for plotting
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        running_train_loss = 0.0
        model.train()  # Set the model to training mode

        # Training loop
        for images, masks in train_loader:
            # Check the model's output shape

            # Move data to the appropriate device
            images, masks = images.to(device), masks.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)  # Output: (batch_size, num_classes, H, W)

            # Compute loss
            loss = criterion(outputs, masks)
            running_train_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Calculate average training loss for this epoch
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loop
        running_val_loss = 0.0
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():

            for images, masks in test_loader:
                # Move data to the appropriate device
                images, masks = images.to(device), masks.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                running_val_loss += loss.item()

        # Calculate average validation loss for this epoch
        avg_val_loss = running_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()

    if save_file != None:
        torch.save(model.state_dict(), save_file)
    if plot_file != None:
        plt.savefig(plot_file)

    return train_losses, val_losses



train_losses, val_losses = train_and_validate_model(model, train_loader, test_loader, criterion, optimizer, save_file, plot_file, num_epochs=100)



# %%
def evaluate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0
    iou_scores = []
    dice_scores = []

    with torch.no_grad():  # Disable gradient computation
        for images, masks in dataloader:
            # Move data to the appropriate device
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)  # (batch_size, num_classes, height, width)
            preds = torch.argmax(outputs, dim=1)  # Predicted class for each pixel

            # Compute loss
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            # Ignore pixels with 255 as GT
            mask_no_background = masks != 255
            predictions = preds[mask_no_background]
            targets = masks[mask_no_background]

            # Compute pixel-wise accuracy
            total_correct += (predictions == targets).sum().item()
            total_pixels += targets.numel()

            for cls in range(model.num_classes):
                # Calculate intersection and union for the class
                intersection = ((predictions == cls) & (targets == cls)).sum().item()
                union = ((predictions == cls) | (targets == cls)).sum().item()
                dice_denominator = (predictions == cls).sum().item() + (targets == cls).sum().item()
                if union == 0:
                    iou_scores.append(float('nan'))  # No samples for this class
                    dice_scores.append(float('nan'))
                else:
                    iou_scores.append(intersection / union)
                    dice_scores.append(2 * intersection / dice_denominator)

    # Compute average metrics
    print('Total Pixels', total_pixels)
    pixel_accuracy = total_correct / total_pixels

    valid_ious = [iou for iou in iou_scores if not torch.isnan(torch.tensor(iou))]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else float('nan')

    valid_dice = [dice for dice in dice_scores if not torch.isnan(torch.tensor(dice))]
    mean_dice = sum(valid_dice) / len(valid_dice) if valid_dice else float('nan')

    print(f"Validation Loss: {total_loss / len(dataloader):.4f}")
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Dice Score: {mean_dice:.4f}")

    return pixel_accuracy, mean_iou, mean_dice


# %%
# Evaluate the model on the test set
import torch
save_file = r"C:\Users\ChloeAtherton\Capstone\models\retrained_wednesday.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResUNet(in_channels=270, num_classes=number_classes).to(device)
model.load_state_dict(torch.load(save_file))
criterion = nn.CrossEntropyLoss(ignore_index=255)
pixel_accuracy, mean_iou, mean_dice = evaluate_model(model, test_loader, criterion)


# %%
class SingleImageDataset(Dataset):
    def __init__(self, image, mask):
        self.image = torch.tensor(image, dtype=torch.float32)  # (270, H, W)
        self.mask = torch.tensor(mask, dtype=torch.long)       # (H, W)

    def __len__(self):
        return 1  # Only one sample

    def __getitem__(self, idx):
        return self.image, self.mask

# %%
def get_single_data_loader(file_path=None):
  hsi_data_one, gt_data_one = load_hdf5_mat_file(file_path)

  hsi_data_one = ScaleData(hsi_data_one, np.min(hsi_data_one), np.max(hsi_data_one))

  if file_path == r"C:\Users\ChloeAtherton\Capstone\data\mat_data\NC12.mat":
    gt_data_one = pd.DataFrame(gt_data_one).replace(nc12_mapping_dict).to_numpy()
  elif file_path == r"C:\Users\ChloeAtherton\Capstone\data\mat_data\NC13.mat":
    gt_data_one = pd.DataFrame(gt_data_one).replace(nc13_mapping_dict).to_numpy()
  elif file_path == R"C:\Users\ChloeAtherton\Capstone\data\mat_data\NC12.mat":
    gt_data_one = pd.DataFrame(gt_data_one).replace(nc16_mapping_dict).to_numpy()

  gt_data_one[gt_data_one == 0] = 255
  gt_data_one[gt_data_one != 255] -= 1

  # pad data to allow all image data to be included
  multiple_required = 8
  h = hsi_data_one.shape[1]
  w = hsi_data_one.shape[2]
  pad_h = multiple_required * (h//multiple_required + 1) - h  # Padding needed for height
  pad_w = multiple_required * (w//multiple_required + 1) - w  # Padding needed for width

  hsi_data_one = np.pad(hsi_data_one, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
  gt_data_one = np.pad(gt_data_one, ((0, pad_h), (0, pad_w)), mode='constant',constant_values=255)

  from torch.utils.data import TensorDataset

  # Example image tensor

  data = SingleImageDataset(hsi_data_one, gt_data_one)
  loader = DataLoader(data, batch_size=1, shuffle=False)
  return loader

# %%
# try and run the model on a whole dataset?
# try running one file

save_file = r"C:\Users\ChloeAtherton\Capstone\models\retrained_wednesday.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResUNet(in_channels=270, num_classes=number_classes).to(device)
model.load_state_dict(torch.load(save_file))
criterion = nn.CrossEntropyLoss(ignore_index=255)

nc12_loader = get_single_data_loader(r"C:\Users\ChloeAtherton\Capstone\data\mat_data\NC12.mat")
nc13_loader = get_single_data_loader(r"C:\Users\ChloeAtherton\Capstone\data\mat_data\NC13.mat")
nc16_loader = get_single_data_loader(r"C:\Users\ChloeAtherton\Capstone\data\mat_data\NC16.mat")

# pixel_accuracy_one, mean_iou_one, mean_dice_one = evaluate_model(model, loader, criterion)



# %%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_predictions_2d(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    num_cls = len(merged_classes)
    cmap = plt.cm.get_cmap('nipy_spectral', num_cls)
    
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
        preds[masks == 255] = 255

        for i in range(min(10, len(images))):
            plt.figure(figsize=(12, 4))
            # Collapse hyperspectral input to 2D by averaging across channels.
            input_2d = images[i].mean(dim=0).cpu()
            plt.subplot(1, 3, 1)
            plt.title("Input (2D Projection)")
            plt.imshow(input_2d)
            
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth")
            plt.imshow(masks[i].cpu(), cmap=cmap, vmin=0, vmax=num_cls-1)
            
            plt.subplot(1, 3, 3)
            plt.title("Prediction")
            plt.imshow(preds[i].cpu(), cmap=cmap, vmin=0, vmax=num_cls-1)
            
            # Build legend handles based on the actual number of classes.
            handles_pred = []
            for j in range(num_cls):
                # Look up the label for class j.
                df_row = merged_classes.loc[merged_classes['gt label'] == j, 'label']
                label = df_row.values[0] if not df_row.empty else f"Class {j}"
                handles_pred.append(mpatches.Patch(color=cmap(j), label=label))
            plt.legend(handles=handles_pred, title="Classes", loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
            
            plt.show()


# %%
visualize_predictions_2d(model, nc12_loader)
visualize_predictions_2d(model, nc13_loader)
visualize_predictions_2d(model, nc16_loader)
visualize_predictions_2d(model, test_loader)


