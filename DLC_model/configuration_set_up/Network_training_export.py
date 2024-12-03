import deeplabcut
import os

path_config_file = "/home/hice1/rwang753/scratch/week8/20_videos_0603-RW-2024-10-10/config.yaml"

deeplabcut.create_training_dataset(path_config_file, net_type='resnet_50', augmenter_type='imgaug')

deeplabcut.train_network(path_config_file, shuffle=2, displayiters=100,saveiters=100)

deeplabcut.evaluate_network(path_config_file,plotting=True)

new_video_path = "/home/hice1/rwang753/scratch/week9/trained_20videos_0603_RW_2024-10-10/20_videos_0603-RW-2024-10-10/videos/5_27_2024"
deeplabcut.analyze_videos(path_config_file,new_video_path, videotype='.MP4',save_as_csv=True)
video_files = [f for f in os.listdir(new_video_path) if f.endswith(('.MP4'))]
for video_file in video_files:
    # Construct the full path to the video file
    video_path = os.path.join(new_video_path, video_file)
    deeplabcut.create_labeled_video(path_config_file,video_path)
    deeplabcut.plot_trajectories(path_config_file,video_path)
