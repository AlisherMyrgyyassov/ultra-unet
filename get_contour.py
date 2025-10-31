from utils.video import *
from utils.processing import *
from utils.postprocessing import *
from networks.ultra_unet import UltraUNet
import torch
from tqdm import tqdm
import argparse

"""
Important info:
If numpy file is selected, then the format should be int8 of size (N, height, width)
Bruce: crop_percentages = (0.25, 0.05, 0.17, 0.16)
Mandarin: crop_percentages = (0.21, 0.18, 0.16, 0.23)
"""


def main():
    parser = argparse.ArgumentParser(description="Process video path and options for cropping and output.")

    parser.add_argument('--data_path', type=str, required=True, help='Path to the video or numpy file')
    parser.add_argument('--crop_percentages', type=float, nargs=4, default=(0.21, 0.18, 0.16, 0.23),
                        help='Crop percentages for top, bottom, left, right (default: 0.21, 0.18, 0.16, 0.23)')
    parser.add_argument('--output_video', action='store_true', default=False,
                        help='Flag to output video (default: False)')
    parser.add_argument('--no_output_video', action='store_false', dest='output_video',
                        help='Flag to disable video output')
    parser.add_argument('--output_coords', action='store_true', default=True,
                        help='Flag to output coordinates (default: True)')
    parser.add_argument('--no_output_coords', action='store_false', dest='output_coords',
                        help='Flag to disable coordinates output')
    parser.add_argument('--output_images', action='store_true', default=False,
                        help='Flag to output images (default: False)')
    parser.add_argument('--no_output_images', action='store_false', dest='output_images',
                        help='Flag to disable images output')
    parser.add_argument('--histogram_matching', action='store_true', default=True,
                        help='Flag to enable histogram matching (default: True)')
    parser.add_argument('--no_histogram_matching', action='store_false', dest='histogram_matching',
                        help='Flag to disable histogram matching')
    parser.add_argument('--save_input_array', action='store_true', default=False,
                        help='Flag to output original array (default: False)')

    args = parser.parse_args()

    data_path = args.data_path
    crop_percentages = tuple(args.crop_percentages)
    output_video = args.output_video
    output_coords = args.output_coords
    output_images = args.output_images
    save_input_array = args.save_input_array
    histogram_matching = args.histogram_matching

    reference_data_path = r"mandarin-hist.pkl"
    import pickle
    from utils.processing import match_histogram_single
    with open(reference_data_path, 'rb') as file:
        reference_data = pickle.load(file)
    reference_cdf = reference_data['cdf']

    # To add later:
    output_gif = False
    batch_size = 10

    # Data accepting
    if data_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        original_width, original_height = get_video_dimensions(data_path, crop_percentages=crop_percentages)

        original_frames = process_video_frames(data_path, size = None, sampling_rate=1, 
                                                crop_percentages=crop_percentages)

        if original_frames is not None: print(f"There are {len(original_frames)} frames processed")

        resized_frames = resize_images(np.array(original_frames), (224,224), save_file=None)
        isolated_frames = [isolate_largest_object(frame) for frame in resized_frames]
        if histogram_matching:
            isolated_frames = [match_histogram_single(frame, reference_cdf) for frame in isolated_frames]
        print("Video file accepted")

    elif data_path.endswith('.npy'):
        original_frames = np.load(data_path)
        original_width, original_height = original_frames.shape[2], original_frames.shape[1]

        resized_frames = resize_images(np.array(original_frames), (224,224), save_file=None)
        isolated_frames = [isolate_largest_object(frame) for frame in resized_frames]
        if histogram_matching:
            isolated_frames = [match_histogram_single(frame, reference_cdf) for frame in isolated_frames]
        print("Numpy array file accepted")

    else:
        print("Unsupported file format.")


    model = UltraUNet(1,1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    checkpoint = torch.load('ultra_unet-full-train.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    animate_skeletons = []

    def split_into_batches(data, batch_size):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    
    video_set = split_into_batches(isolated_frames, batch_size)


    with torch.no_grad():
        for images in tqdm(video_set, desc="Processing Images"):
            images = torch.tensor(np.array(images)).unsqueeze(1).float().cuda() / 255.0
            outputs = torch.sigmoid(model(images)).cpu().detach().numpy()

            for slice_num in range(outputs.shape[0]):
                out_heatmap = outputs[slice_num][0]
                out_heatmap = largest_connected_component(out_heatmap)
                    
                skelet = perform_skeletonization(out_heatmap, threshold=0.6)
                skelet = clean_skeleton(skelet)

                output_coord = skeleton_to_coordinates(skelet)
                animate_skeletons.append(output_coord)

    animate_skeletons = resize_coordinates_list(animate_skeletons, (224,224), (original_width, original_height))

    if output_video: 
        print("Video generation has started...")
        animate_two(original_frames, animate_skeletons, fps=30, filename="output.mp4", save_option="video")
        print("Video generation finished")
    if output_coords: 
        print("JSON file generation has started...")
        with open("contours.json", "w") as file:
            json.dump(animate_skeletons, file)
        print("JSON file has been created")
    if output_images:
        print("Images generation has started...")
        save_images_with_points(original_frames, animate_skeletons, output_dir="output")
        print("Images has been created")
    if save_input_array:
        print("Saving input array...")
        np.save("input_array.npy", isolated_frames)
        print("Input array has been created")

if __name__ == "__main__":
    main()