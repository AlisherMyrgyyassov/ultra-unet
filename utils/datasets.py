
import os
import torch
import pandas as pd
import torch.nn.functional as F
import json

import random

def split_indices(seed = None, total_count=35, split_ratio=(28, 7)):
    """
    Splits a range of indices into two non-overlapping groups based on a given ratio.

    Args:
    - seed (int): Random seed for reproducibility.
    - total_count (int): Total number of indices.
    - split_ratio (tuple): Tuple representing the split ratio (first_group_size, second_group_size).

    Returns:
    - tuple: Two lists containing the indices for each group.
    """
    # Set the random seed for reproducibility
    if seed != None: random.seed(seed)

    # Generate a list of indices from 0 to total_count-1
    indices = list(range(total_count))

    # Shuffle the indices randomly
    random.shuffle(indices)

    # Split indices according to the split_ratio
    first_group_size, second_group_size = split_ratio
    if sum(split_ratio) != total_count:
        raise ValueError("The sum of the split ratios must equal the total count of indices")

    first_group = indices[:first_group_size]
    second_group = indices[first_group_size:first_group_size + second_group_size]

    return first_group, second_group


def extract_coordinates(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Initialize an empty list to store all coordinates
    all_coordinates = []

    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        region_shape_attributes = row['region_shape_attributes']
              
        # Load the JSON data
        data = json.loads(region_shape_attributes)
        
        # Extract the coordinates
        all_points_x = data.get('all_points_x', [])
        all_points_y = data.get('all_points_y', [])
        
        # Combine x and y coordinates into a list of [x, y] pairs
        coordinates = [[x, y] for x, y in zip(all_points_x, all_points_y)]
        
        # Append the coordinates to the main list
        all_coordinates.append(coordinates)
    
    # Return the compiled list of lists
    return all_coordinates



from torch.utils.data import Dataset
import torch
import numpy as np
import json
from utils.processing import *
from utils.postprocessing import *


import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF

class AnnotationDataset(Dataset):
    def __init__(self, annotations_folder):
        """
        Initialize the dataset with the path to the folder containing annotation files.
        
        Parameters:
        annotations_folder (str): Path to the folder containing annotation files.
        """
        self.annotations_folder = annotations_folder
        self.annotation_files = sorted(
            [f for f in os.listdir(annotations_folder) if f.endswith('_annot.json')],
            key=lambda x: int(x.split('_')[0])
        )
    
    def __len__(self):
        """
        Return the number of annotation files in the dataset.
        
        Returns:
        int: Number of annotation files.
        """
        return len(self.annotation_files)
    
    def __getitem__(self, idx):
        """
        Get the annotation data for the given index.
        
        Parameters:
        idx (int): Index of the annotation file to retrieve.
        
        Returns:
        list: Annotation data from the JSON file.
        """
        if idx >= len(self) or idx < 0:
            raise IndexError("Index out of range")
        
        annotation_file = self.annotation_files[idx]
        annotation_path = os.path.join(self.annotations_folder, annotation_file)
        
        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)
        
        return annotation_data

class NoisyTransform:
    def __init__(self):
        self.horizontal_flip_prob = 0.5
        self.rotation_degrees = 15
        self.gaussian_noise_mean = 0
        self.gaussian_noise_std = 0.03
        self.shadow_prob = 0.99

    def add_gaussian_noise(self, images):
        noise = torch.randn(images.size()) * self.gaussian_noise_std + self.gaussian_noise_mean
        noisy_images = images + noise
        return torch.clamp(noisy_images, 0, 1)  # Ensure the pixel values are within [0, 1]

    def add_shadow(self, images):
        if random.random() < self.shadow_prob:
            shadow_value = random.uniform(0.2, 0.5)
            start_row = random.randint(0, images.shape[1] // 2)
            end_row = random.randint(images.shape[1] // 2, images.shape[1])
            images[:, start_row:end_row, :] *= shadow_value
        return images

    def __call__(self, images, heatmaps):
        # Random Horizontal Flip
        if random.random() < self.horizontal_flip_prob:
            images = F.hflip(images)
            heatmaps = F.hflip(heatmaps)

        # Random Rotation
        # angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        # images = F.rotate(images, angle)
        # heatmaps = F.rotate(heatmaps, angle)

        # Add Gaussian Noise
        images = self.add_gaussian_noise(images)

        # Add Shadowing
        images = self.add_shadow(images)

        return images, heatmaps
    
class SmallTransform:
    def __init__(self):
        self.horizontal_flip_prob = 0.5

    def __call__(self, images, heatmaps):
        if random.random() < self.horizontal_flip_prob:
            images = F.hflip(images)
            heatmaps = F.hflip(heatmaps)

        return images, heatmaps

class Fast_Dataset(Dataset):
    # It is a dataset based on already saved miniatures processed using the main dataset class

    def __init__(self, num_samples, data_dir, type_heatmap = "gauss", transform = None, param = 3, resized_dims = None):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            data_dir (str): Directory where the .npy files and .json files are stored.
            type_heatmap (str): Type of heatmap to generate. ("circles" or "gauss")
            transform (callable, optional): Optional transform to be applied on a sample.
            param (int): Sigma or radius value for heatmap/circles
            resized_dims(tuple): (height, weight) tuple
        """
        self.num_samples = num_samples
        self.data_dir = data_dir
        self.type_heatmap = type_heatmap
        self.transform = transform
        self.param = param
        self.resized_dims = resized_dims


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load images from .npy files
        images = np.load(f"{self.data_dir}/{idx}_images.npy")
        # Load annots
        with open(f"{self.data_dir}/{idx}_annot.json", 'r') as f:
            annotations = json.load(f)

        original_height, original_width = images.shape[1],images.shape[2] # height=439, width=659

        if self.resized_dims is not None: 
            height, width = self.resized_dims[0], self.resized_dims[1] # If resize
            images = resize_images(images=images, target_shape=(height, width))
            annotations = resize_coordinates_list(annotations, (original_width, original_height), (width, height))
        else:
            height, width = original_height, original_width # If no resize

        images = torch.from_numpy(images).float()        

        heatmaps_list = []
        for annot in annotations: 
            heatmap = annotations_to_heatmap(annot, type = self.type_heatmap, param=self.param, 
                                             height=height, width=width)
            heatmaps_list.append(heatmap)

        heatmaps_tensor = torch.tensor(heatmaps_list, dtype=torch.float32)

        # Transformations
        if self.transform: images, heatmaps_tensor = self.transform(images, heatmaps_tensor)

        return images, heatmaps_tensor
    
    def get_annotations(self, idx):
        # Not resized
        with open(f"{self.data_dir}/{idx}_annot.json", 'r') as f:
            annotations = json.load(f)

        images = np.load(f"{self.data_dir}/{idx}_images.npy")

        original_height, original_width = images.shape[1],images.shape[2] # height=439, width=659

        if self.resized_dims is not None: 
            height, width = self.resized_dims[0], self.resized_dims[1] # If resize
            images = resize_images(images=images, target_shape=(height, width))
            annotations = resize_coordinates_list(annotations, (original_width, original_height), (width, height))

        return annotations
    
class Window_Dataset(Dataset):
    def __init__(self, dataset, window_size):
        self.dataset = dataset
        self.window_size = window_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        images, heatmaps = self.dataset[idx]
        images = gaussian_sliding_window(images, window_size=self.window_size)
        heatmaps = heatmaps_for_sliding_window(heatmaps, window_size=self.window_size)
        return images, heatmaps

    def get_annotations(self, idx):
        return self.dataset.get_annotations(idx)
    