import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm

def process_video_frames(video_path, size = (120, 120), sampling_rate = 10,
                         crop_percentages = (0.21, 0.18, 0.16, 0.23)):
    """
    Convert video to separate numpy frames

    Parameters:
    -video_path (str): path to the file
    -size (tuple): resize arrays to (width, height). If None, then no resize done.
    -sampling_rate (int): sampling rate of the output frames
    -crop_percentages (tuple): top_pct, bottom_pct, left_pct, right_pct

    Returns:
    -list: list of numpy frames
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    frames = []
    frame_count = 0 # To control the sampling rate

    while True:
        ret, frame = cap.read()
        if not ret: break  # no frames left to read

        if frame_count % sampling_rate == 0:
            # dims
            height, width = frame.shape[:2]
            top_pct, bottom_pct, left_pct, right_pct = crop_percentages # Define crop percentages
            top = int(height * top_pct)  # Remove top %
            bottom = int (height - height * bottom_pct) # Remove bottom %
            left = int(width * left_pct)  # Remove left %
            right = int(width - width * right_pct)  # Remove right %

            frame = frame[top:bottom, left:right] # Crop
            if size != None: frame = cv2.resize(frame, size) # Resize

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Grayscale
            frames.append(frame)
        
        frame_count += 1
    cap.release()

    return frames

def get_video_dimensions(video_path, crop_percentages=None):
    """
    Get the dimensions (width and height) of a video, optionally returning cropped dimensions.

    Parameters:
    - video_path: Path to the video file.
    - crop_percentages: Optional tuple of (top_pct, bottom_pct, left_pct, right_pct) for cropping.

    Returns:
    - A tuple containing (width, height) of the video frames, optionally cropped.
    """

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if crop_percentages:
        top_pct, bottom_pct, left_pct, right_pct = crop_percentages
        
        top = int(height * top_pct)  
        bottom = int(height - height * bottom_pct) 
        left = int(width * left_pct)  
        right = int(width - width * right_pct)  

        new_width = right - left
        new_height = bottom - top
    else:
        new_width, new_height = width, height

    cap.release()

    return new_width, new_height

def isolate_largest_object(frame):
    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)

    # Find contours from the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found")
        return None

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest contour
    mask = np.zeros_like(frame)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to the original frame
    isolated_image = cv2.bitwise_and(frame, frame, mask=mask)

    return isolated_image

def save_images_to_folder(images, folder_path):
    for i, image in enumerate(images):
        file_path = os.path.join(folder_path, f'{i + 1}.jpg')
        cv2.imwrite(file_path, image)


def process_images(image_list):
    processed_images = []

    for img in image_list:
        # equalized_img = cv2.equalizeHist(img)
        # img_float = equalized_img.astype(np.float32) / 255

        img_float = img/255.0

        processed_images.append(img_float)

    return processed_images

def animate_one(images, points_array, filename = "animation.gif", save_gif=False):
    """
    Creates an animation of images with points plotted on top,
    optionally saves as a GIF.

    Args:
    - images (list of numpy.ndarray): A list of 2D arrays, each representing an image.
    - points_array (list of numpy.ndarray): A list of arrays, where each array contains points (as [x, y] coordinates) to plot on the corresponding image.
    - msd_values (list of float): A list of Mean Squared Displacement values to display below each image.
    - save_gif (bool): If True, saves the animation as a GIF file.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        # Clear the current axes
        ax.clear()
        # Display the image
        ax.imshow(images[frame], cmap='gray')
        # Plot the points for this frame
        ax.scatter(points_array[frame][:, 0], points_array[frame][:, 1], color='red', s=10)
        # Turn off axis labels
        ax.axis('off')
        return ax  # Return both the axes and the text element as a tuple

    ani = FuncAnimation(fig, update, frames=len(images), repeat=False)

    if save_gif:
        # Save the animation as a GIF
        ani.save(filename, writer='pillow', fps=3)

    plt.show()


def animate_two(images, points_array, fps=3, filename="animation.gif", save_option=None):
    """
    Creates an animation of images with points plotted on top,
    optionally saves as a GIF or video.

    Args:
    - images (list of numpy.ndarray): A list of 2D arrays, each representing an image.
    - points_array (list of numpy.ndarray): A list of arrays, where each array contains points (as [x, y] coordinates) to plot on the corresponding image.
    - fps (int): Frames per second for the output animation.
    - filename (str): The filename to save the animation.
    - save_option (str): "gif" to save as GIF, "video" to save as video.
    """
    if isinstance(images[0], list):
        images = [np.array(image) for image in images]
    if isinstance(points_array[0], list):
        points_array = [np.array(points) for points in points_array]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Initialize a tqdm progress bar
    progress_bar = tqdm(total=len(images), desc="Animating frames")

    def update(frame):
        ax1.clear()
        ax2.clear()
        
        ax1.imshow(images[frame], cmap='gray')
        ax1.axis('off')
        
        if points_array[frame].size > 0:
            ax2.imshow(images[frame], cmap='gray')
            ax2.scatter(points_array[frame][:, 0], points_array[frame][:, 1], color='red', s=10)
        else:
            ax2.imshow(images[frame], cmap='gray')
    
        ax2.axis('off')
        
        # Update the progress bar
        progress_bar.update(1)

        return ax1, ax2  # Return both axes

    ani = FuncAnimation(fig, update, frames=len(images), repeat=False)

    if save_option == "gif":
        ani.save(filename, writer='pillow', fps=fps)

    if save_option == "video":
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(filename, writer=writer)

    # plt.show()

    # Close the progress bar after completion
    progress_bar.close()
