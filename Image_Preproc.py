# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 08:10:57 2024

@author: Haliz369
"""
import numpy as np

def ExtractPatches(img,GT=None, windowSize=3):
# This function divied the input image into patches making it ready for training your algorithm
# img: your image
# GT= your ground truth image. If you do not have any GT, just leave it
# windowSize: the size of each patch

  if GT is None:
        GT=np.ones_like(img[0,:,:]) # if the GT is not define, we create a dummy GT. In this case, you should ignore the output 'labels'

  margin = int((windowSize - 1) / 2) # margin to be added into your image
  img=np.pad(img, pad_width=((0,0),(margin,margin),(margin,margin)),mode='edge') # padding the input image according to the margin

  GT=np.pad(GT, pad_width=((margin,margin),(margin,margin)) ) # padding GT
  pos_in_image=np.asarray(np.where(GT!=0)).T # find the labled data in ground truth gt
  labels= GT[pos_in_image[:,0],pos_in_image[:,1]] #label of the samples

  ExtractedPatches=[]
  for i in pos_in_image:
    b=img[:,i[0]-margin:i[0]+margin+1, i[1]-margin:i[1]+margin+1] # extract patches
    ExtractedPatches.append(b)

  return np.asanyarray(ExtractedPatches), labels, pos_in_image
#######################################################################

def ExtractPatches2(img, GT=None, windowSize=3):
    # This function divides the input image into patches making it ready for training your algorithm
    # img: your image
    # GT= your ground truth image. If you do not have any GT, just leave it
    # windowSize: the size of each patch

    if GT is None:
        GT = np.ones_like(img[0, :, :])  # if the GT is not defined, we create a dummy GT. In this case, you should ignore the output 'labels'

    margin = int((windowSize - 1) / 2)  # margin to be added into your image
    img = np.pad(img, pad_width=((0, 0), (margin, margin), (margin, margin)), mode='edge')  # padding the input image according to the margin

    GT = np.pad(GT, pad_width=((margin, margin), (margin, margin)))  # padding GT
    pos_in_image = np.asarray(np.where(GT != 0)).T  # find the labeled data in ground truth gt
    labels = GT[pos_in_image[:, 0], pos_in_image[:, 1]]  # label of the samples

    def generate_patches():
        for i in pos_in_image:
            b = img[:, i[0] - margin:i[0] + margin + 1, i[1] - margin:i[1] + margin + 1]  # extract patches
            yield b

    return generate_patches(), labels, pos_in_image

def ScaleData(X,min_value,max_value):
  #This function sclae the data 'X' between 0 and 1
  min_value=np.float16(min_value)
  max_value=np.float16(max_value)
  X -= min_value
  X /= (max_value - min_value)
  return X



from sklearn.model_selection import StratifiedShuffleSplit

def replace_percent_stratified(GT, percent):
    percent=100-percent
    unique_labels = np.unique(GT)
    num_classes = len(unique_labels)
    total_pixels = GT.size
    selected_pixels = int(total_pixels * (percent / 100))

    # Create an array of class labels for the StratifiedShuffleSplit
    labels = GT.reshape(-1)  # Flatten GT array
    indices = np.arange(len(labels))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=selected_pixels, random_state=42)
    selected_indices = next(sss.split(indices, labels))[1]

    new_GT = GT.copy()
    new_GT.flat[selected_indices] = 0

    return new_GT

# from sklearn.model_selection import StratifiedShuffleSplit
# import numpy as np

def replace_percent_stratified2(GT, percent):
    unique_labels, class_counts = np.unique(GT, return_counts=True)
    total_pixels = GT.size
    selected_pixels = int(total_pixels * (percent / 100))
    
    # Create an array of class labels for the StratifiedShuffleSplit
    labels = GT.reshape(-1)  # Flatten GT array
    indices = np.arange(len(labels))
    
    # Initialize a dictionary to store the indices for each class
    class_indices = {label: [] for label in unique_labels}
    for i, label in enumerate(labels):
        class_indices[label].append(indices[i])
    
    # Sample from each class proportionally
    selected_indices = []
    for label, count in zip(unique_labels, class_counts):
        num_samples = int(count * (percent / 100))
        sss = StratifiedShuffleSplit(n_splits=1, test_size=count - num_samples, random_state=42)
        _, sampled_indices = next(sss.split(np.zeros(count), np.zeros(count)))
        selected_indices.extend(class_indices[label][i] for i in sampled_indices)

    # Create the new ground truth array
    new_GT = GT.copy()
    new_GT.flat[selected_indices] = 0

    return new_GT


def replace_percent_stratified3(GT, percent):
    unique_labels, class_counts = np.unique(GT, return_counts=True)
    total_pixels = GT.size
    selected_pixels = int(total_pixels * (percent / 100))
    
    # Create an array of class labels for the StratifiedShuffleSplit
    labels = GT.reshape(-1)  # Flatten GT array
    indices = np.arange(len(labels))
    
    # Initialize a dictionary to store the indices for each class
    class_indices = {label: [] for label in unique_labels}
    for i, label in enumerate(labels):
        class_indices[label].append(indices[i])
    
    # Initialize a list to store selected class labels
    selected_labels = []
    
    # Sample from each class proportionally, ensuring each class has at least two samples
    selected_indices = []
    for label, count in zip(unique_labels, class_counts):
        if count >= 2:  # Ensure at least two samples per class
            selected_labels.append(label)
            num_samples = int(count * (percent / 100))
            sss = StratifiedShuffleSplit(n_splits=1, test_size=count - num_samples, random_state=42)
            _, sampled_indices = next(sss.split(np.zeros(count), np.zeros(count)))
            selected_indices.extend(class_indices[label][i] for i in sampled_indices)
    
    # Create the new ground truth array
    new_GT = GT.copy()
    new_GT.flat[selected_indices] = 0

    return new_GT

import torch

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