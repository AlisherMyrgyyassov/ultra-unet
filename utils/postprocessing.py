import torch
import numpy as np
from skimage import measure

def largest_connected_component(images, threshold=0.7):
    """
    Keeps only the largest connected component in either a batch of 2D images or a single 2D image.
    Args:
    - images (numpy array): Input batch of images (Batch Size, W, H) or a single image (W, H)
    - threshold (float): Pixels with intensities below this are considered background.

    Returns:
    - numpy array: Processed batch of images or a single processed image.
    """
    if isinstance(images, torch.Tensor): images = images.numpy() #convert to numpy if not

    # Check if input is a single image by its dimension
    if images.ndim == 2:
        images = images[np.newaxis, ...]  # Add a batch dimension if it's a single image

    # Initialize processed batch with the same shape and type as input
    processed_batch = np.zeros_like(images)

    for i in range(images.shape[0]):
        # Apply threshold
        binary_image = images[i] > threshold

        # Label connected components
        labels = measure.label(binary_image, connectivity=2)
        if labels.max() == 0:
            continue  # No components found

        # Find the largest component
        largest_component = np.argmax(np.bincount(labels.flat)[1:]) + 1

        # Create a mask of the largest component
        processed_batch[i] = (labels == largest_component).astype(float)

    # If it was a single image input, remove the batch dimension before returning
    if processed_batch.shape[0] == 1:
        return processed_batch[0]
    else:
        return processed_batch

import numpy as np
import networkx as nx
from skimage import measure, morphology, util
from skimage.graph import route_through_array

def clean_skeleton(image):
    """
    Cleans a skeletonized image by keeping the longest continuous path, 
    removing small branches that are connected to the main path.
    
    Args:
    - image (numpy.ndarray): Input 2D skeletonized image (W, H)

    Returns:
    - numpy.ndarray: Processed image.
    """
    # Skeleton must be binary, ensure it is
    skeleton = image > 0.5

    # Convert skeleton to graph
    G = nx.Graph()
    for r, c in np.argwhere(skeleton):
        # Add edges to all 8 possible neighbors
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr != 0 or dc != 0:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < skeleton.shape[0] and 0 <= cc < skeleton.shape[1] and skeleton[rr, cc]:
                        G.add_edge((r, c), (rr, cc))

    # Find all paths from each junction or endpoint to another
    endpoints = [node for node, degree in G.degree if degree == 1]
    max_path = []
    max_length = 0

    # Compute the longest path in the graph
    for start in endpoints:
        lengths = nx.single_source_dijkstra_path_length(G, start)
        farthest_node = max(lengths, key=lengths.get)
        path_length = lengths[farthest_node]
        if path_length > max_length:
            max_path = nx.shortest_path(G, start, farthest_node)
            max_length = path_length

    # Create an image from the longest path
    cleaned_image = np.zeros_like(image)
    for r, c in max_path:
        cleaned_image[r, c] = 1

    return cleaned_image

def convert_msd_to_mm(original_shape, new_shape, msd_pixel_score):
    """
    Convert MSD score from pixels to mm considering the original and resized image shapes.
    
    Parameters:
    original_shape (tuple): Original shape of the image (height, width).
    new_shape (tuple): Resized shape of the image (height, width).
    msd_pixel_score (float): MSD score in pixels.
    
    Returns:
    float: MSD score in mm.
    """
    reference_pixel_length = 439
    reference_mm_length = 90  # in mm

    pixel_to_mm_factor = reference_mm_length / reference_pixel_length

    height_scale = original_shape[0] / new_shape[0]
    width_scale = original_shape[1] / new_shape[1]

    average_scale = (height_scale + width_scale) / 2

    adjusted_msd_pixel_score = msd_pixel_score * average_scale

    msd_mm_score = adjusted_msd_pixel_score * pixel_to_mm_factor
    
    return msd_mm_score
