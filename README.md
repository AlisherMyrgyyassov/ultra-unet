# Tongue Contour Segmentation
This project involves the segmentation of tongue contours from ultrasound video data using a U-Net model. The processed data can be output as a video or as coordinate data in JSON format.

# Project Overview
The main script, get_contour.py, processes video input to segment tongue contours with a pre-trained 2D U-Net model (1 input channel, 1 output). The script includes options for cropping the video, outputting a processed video, and saving coordinate data.

# Installation
Ensure you have Python 3.6+ and the necessary libraries installed. You can set up the environment by running:
```
pip install -r requirements.txt
```
>Note: Ensure you have access to a CUDA-compatible GPU for optimal performance.
**To run the model, you need to request the latest checkpoint directly from me. Contact me to get the file.**

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
--output_video: Flag to output the processed video. Enabled by default.
--no_output_video: Use this flag to disable video output.
--output_coords: Flag to output coordinates in JSON. Enabled by default.
--no_output_coords: Use this flag to disable coordinate output.
```

The output will depend on your selected flags. If the output video is enabled, then a video with a tongue contour on top of it will be processed and generated. It may take a while, though.
You may also set up a custom crop percentage depending on your task and the area you would like to process through the model.

Example
Below are examples showcasing the script's functionality.

![image](https://github.com/user-attachments/assets/ee77b632-345f-416a-b4aa-265671a552dc)

You may also have a look at the sample outputs added to this repo.
