# Tongue Contour Segmentation with UltraUNet
This project involves the segmentation of tongue contours from ultrasound video data using a UltraUNet model. The processed data can be output as a video, images, and/or as coordinate data in JSON format.

# Project Overview
The main script, get_contour.py, processes video or NumPy-array input to segment tongue contours with a pre-trained 2D UltraUNet model (1 input channel, 1 output). The script includes options for cropping the video, outputting a processed video, and saving coordinate data in the json format. The model is robust and effective across different datasets with varying imaging quality, conditions, and noise levels. 

# Installation
Ensure you have Python 3.6+ and the necessary libraries installed. 

>Note: Ensure you have access to a CUDA-compatible GPU for optimal performance.
**The GitHub Repo contains the .pth file of UltraUNet trained on all available data**

# Usage
To run the main processing script, use the following command:

```
cd [path to your folder]
python get_contour.py --video_path [path to video]
```
The progress bar will show the approximate time needed to finish segmentation.
Typically, a full segmentation task takes about 3 minutes on a 2-minute video.

## Command-Line Arguments
```
--video_path: (Required) Path to the video file.
--crop_percentages: Crop percentages for top, bottom, left, right of the video. Default is (0.21, 0.18, 0.16, 0.23).
--output_video: Flag to output the processed video.
--no_output_video: Use this flag to disable video output.
--output_coords: Flag to output coordinates in JSON. Enabled by default.
--no_output_coords: Use this flag to disable the coordinate output.
--output_images:  Flag to output images.
--no_output_images: Use this flag to disable the image output
--histogram_matching: Flag to enable histogram matching (This usually improves the model's performance)
--no_histogram_matching: Flag to disable histogram matching
```

The output will depend on your selected flags. If the output video is enabled, then a video with a tongue contour on top of it will be processed and generated. It may take a while, though.
You may also set up a custom crop percentage depending on your task and the area you would like to process through the model.

The pipeline also automatically extracts only the largest connected area from your GUI video, ignoring other interface options except for the main window.


# Example
Below are examples showcasing the script's functionality.
Image sample output taken from the output folder:

<p align="center">
  <img src="https://github.com/user-attachments/assets/9952142a-96b2-4b7c-bf3b-908cb955b988"/>
</p>


Single frame from a video output:
<p align="center">
  <img src="https://github.com/user-attachments/assets/ee77b632-345f-416a-b4aa-265671a552dc"/>
</p>

# Citation
If you would like to use this work or would like to know more about model validation results, methodology, and training pipeline, please refer to this paper:
https://doi.org/10.48550/arXiv.2509.23225


