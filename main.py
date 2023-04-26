import torch
from ultralytics import YOLO
import video_processing
import argparse


# Working on torch 2.0.0+cu117
# Build cuda_11.7.r11.7/compiler.31294372_0
# Requirements to run the model on GPU
# CUDA Toolkit 11.7: https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux
# cuDNN 11.x https://developer.nvidia.com/rdp/cudnn-download
# Python 3.10.7

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = [YOLO(model_path) for model_path in args.models]

    video_file_path = args.video

    class_labels = ['v1', 'v2', 'v3']

    video_processing.process(video_file_path, models, class_labels, record_angles=True, record_angle_freq=10)


def parse_args():
    parser = argparse.ArgumentParser(description='Process video with specified models.')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['models/2023-03-25-yolov8m-100epochs-augmented-last.pt',
                                 'models/2023-03-25-yolov8m-239epochs-augmented-last.pt'],
                        help='Paths to the model files.')
    parser.add_argument('--video', type=str, default='videos/my_video-2.mkv',
                        help='Path to the video file.')

    return parser.parse_args()


# python main.py --models path/to/model1.pt path/to/model2.pt path/to/model3.pt --video path/to/your/video.mkv
if __name__ == '__main__':
    args = parse_args()
    main(args)
