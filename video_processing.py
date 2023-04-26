import cv2
import time
import pupil_apriltags as apriltag

from scripts.angle_recording import get_angles, init_angle_vars, plot_angle_histogram
from scripts.april_tag_utils import get_predicted_angle
from scripts.bounding_box_utils import get_highest_confidence_boxes, draw_result_on_frame, draw_fps_on_frame
from scripts.kalman_filter_utils import init_kalman_filter


def process(
        video_file_path,
        models,
        class_labels,
        record_angles=False,
        record_angle_freq=10
):
    """Summary line.

        Extended description of function.

        Args:
            arg1 (int): Description of arg1
            arg2 (str): Description of arg2

        Returns:
            bool: Description of return value
            :param record_angle_freq:
            :param record_angles:
            :param class_labels:
            :param models:
            :param video_file_path:

        """
    cap = cv2.VideoCapture(video_file_path)

    models = models

    # Get the video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    # Define the desired width and height
    desired_width = 640
    desired_height = int((desired_width / width) * height)

    # Calculate the aspect ratio of the original and scaled frames
    aspect_ratio_x = width / desired_width
    aspect_ratio_y = height / desired_height

    # Create a video writer object
    out = cv2.VideoWriter('output.mp4', codec, fps, (width, height))

    # Initialize the AprilTag detector
    detector = apriltag.Detector()

    # Initialize the Kalman filter
    init_kalman_filter(video_fps=fps)

    # Keep track of frames
    n_frames = 0

    init_angle_vars()

    # Loop through each frame of the video
    while cap.isOpened():

        # Read the frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        n_frames += 1

        # Scale the frame to the desired size
        scaled_frame = cv2.resize(frame, (desired_width, desired_height))

        # Start time to measure model running time
        start_time = time.time()

        confidence_boxes = get_highest_confidence_boxes(
            models,
            scaled_frame,
            aspect_ratio_x,
            aspect_ratio_y,
            class_labels
        )

        # End time for model inference
        fps_model = 1 / (time.time() - start_time)

        # Get the predicted angle of the found confidence_boxes
        predicted_angles = get_predicted_angle(frame, confidence_boxes, detector)

        draw_result_on_frame(frame, confidence_boxes, predicted_angles)

        draw_fps_on_frame(frame, fps, fps_model)

        if n_frames % record_angle_freq == 0 and record_angles and 'v1' in predicted_angles:
            cv2.destroyAllWindows()
            get_angles(frame, predicted_angles)
        else:
            cv2.imshow('Frame', frame)

        # Save result to output.mp4 video file
        out.write(frame)
        key = cv2.waitKey(1) & 0xFF

        # Press 'q' to exit the loop
        if key == ord('q'):
            break

        # Press 'p' to pause/resume the loop
        if key == ord('p'):
            while True:
                key2 = cv2.waitKey(1) & 0xFF
                cv2.imshow('Frame', frame)
                if key2 == ord('p') or key2 == ord('q'):
                    break

    if record_angles:
        plot_angle_histogram()

