from utils.video import *
from utils.processing import *
from utils.postprocessing import *
from unet import UNet
import torch
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser(description="Process video path and options for cropping and output.")

    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file')
    parser.add_argument('--crop_percentages', type=float, nargs=4, default=(0.21, 0.18, 0.16, 0.23),
                        help='Crop percentages for top, bottom, left, right (default: 0.21, 0.18, 0.16, 0.23)')
    parser.add_argument('--output_video', action='store_true', default=True,
                        help='Flag to output video (default: True)')
    parser.add_argument('--no_output_video', action='store_false', dest='output_video',
                        help='Flag to disable video output')
    parser.add_argument('--output_coords', action='store_true', default=True,
                        help='Flag to output coordinates (default: True)')
    parser.add_argument('--no_output_coords', action='store_false', dest='output_coords',
                        help='Flag to disable coordinates output')

    args = parser.parse_args()

    video_path = args.video_path
    crop_percentages = tuple(args.crop_percentages)
    output_video = args.output_video
    output_coords = args.output_coords


    # To add later:
    output_gif = False


    original_width, original_height = get_video_dimensions(video_path, crop_percentages=crop_percentages)

    original_frames = process_video_frames(video_path, size = None, sampling_rate=1, 
                                            crop_percentages=crop_percentages)

    if original_frames is not None: print(f"There are {len(original_frames)} frames processed")

    resized_frames = resize_images(np.array(original_frames), (512,512), save_file=None)
    isolated_frames = [isolate_largest_object(frame) for frame in resized_frames]

    model = UNet(1,1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    checkpoint = torch.load('model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    animate_images = []
    animate_skeletons = []


    def split_into_batches(data, batch_size):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    batch_size = 10
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

    animate_skeletons = resize_coordinates_list(animate_skeletons, (512,512), (original_width, original_height))

    if output_video: 
        print("Video generation has started...")
        animate_two(original_frames, animate_skeletons, fps=30, filename="output.mp4", save_option="video")
        print("Video generation finished")
    if output_coords: 
        print("JSON file generation has started...")
        with open("output.json", "w") as file:
            json.dump(animate_skeletons, file)
        print("JSON file has been created")

if __name__ == "__main__":
    main()