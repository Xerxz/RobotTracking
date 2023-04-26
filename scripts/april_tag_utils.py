import cv2
from scipy.spatial.transform import Rotation

import scripts.kalman_filter_utils
from scripts.kalman_filter_utils import *
from scripts.bounding_box_utils import *

# Create a dummy camera matrix with the focal lengths set to the image width and height
width, height = 1280, 720  # Replace these with the dimensions of your video frame
dummy_camera_matrix = np.array([[width, 0, width / 2], [0, height, height / 2], [0, 0, 1]])

# Create a dummy distortion coefficients array with all zeros
dummy_distortion_coeffs = np.zeros(5)

april_tag_labels = {6: 'v1', 4: 'v2', 5: 'v3'}
april_tag_labels_reverse = {'v1': 6, 'v2': 4, 'v3': 5}

keys = [0, 0, 0]
angles = [0, 0, 0]
last_filtered_angles = [0, 0, 0]
last_bboxes = [np.array([]), np.array([]), np.array([])]


def normalize_angle(angle):
    return angle + 360 if angle < 0 else angle


def detect_apriltags(frame, bboxes, detector):
    # Initialize a dictionary to store the detected AprilTag IDs and their corresponding bounding boxes
    detected_tags = {}

    # Iterate over the bounding boxes
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox

        # Calculate the margins for cropping
        margin_top = min(10, y1)
        margin_bottom = min(10, frame.shape[0] - y2)
        margin_left = min(10, x1)
        margin_right = min(10, frame.shape[1] - x2)

        # Crop the frame using the calculated margins
        cropped_frame = frame[y1 - margin_top:y2 + margin_bottom, x1 - margin_left:x2 + margin_right]

        # Check if the cropped frame is not empty
        if cropped_frame.size == 0:
            continue
        # Convert the cropped frame to grayscale
        cropped_frame_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to remove noise
        blur_size = (3, 3)  # You can adjust the size of the kernel depending on the level of noise
        blurred_img = cv2.GaussianBlur(cropped_frame_gray, blur_size, 0)

        # Apply a threshold to convert the image to black and white
        threshold_value = 200  # You can adjust this value depending on the image conditions
        _, thresholded_img = cv2.threshold(blurred_img, threshold_value, 255, cv2.THRESH_BINARY)

        tags = detector.detect(thresholded_img)

        # If an AprilTag is detected, store its ID, angles, and the bounding box in the dictionary
        if tags:
            tag_id = tags[0].tag_id

            # Compute the pose using solvePnP
            object_points = np.float32([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]])
            image_points = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            ret, rvec, tvec = cv2.solvePnP(object_points, image_points, dummy_camera_matrix, dummy_distortion_coeffs)

            # Convert the rotation vector to a rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # Calculate the Euler angles from the rotation matrix
            sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
            singular = sy < 1e-6

            if not singular:
                pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                yaw = np.arctan2(-rotation_matrix[2, 0], sy)
                roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                yaw = np.arctan2(-rotation_matrix[2, 0], sy)
                roll = 0

            # Get the homography matrix
            homography = tags[0].homography

            num, Rs, Ts, Ns = cv2.decomposeHomographyMat(homography, dummy_camera_matrix)
            r = Rotation.from_matrix(Rs)
            angles = r.as_euler("zyx", degrees=True)

            # Convert the Euler angles to degrees and normalize them
            pitch, yaw, roll = [normalize_angle(np.rad2deg(angle)) for angle in [pitch, yaw, roll]]

            detected_tags[tag_id] = {
                'bbox': (x1, y1, x2, y2),
                'pitch': normalize_angle(pitch),
                'yaw': normalize_angle(yaw),
                'roll': normalize_angle(roll),
                'angle': angles
            }

    return detected_tags


def get_predicted_angle(frame, all_boxes, detector):

    result = []

    for label_index_og, (conf, x1, y1, x2, y2) in all_boxes.items():

        april_id = detect_apriltags(frame, [(x1, y1, x2, y2)], detector)

        # If april_tag is detected
        if april_id:
            key, tag_info = next(iter(april_id.items()))
            keys[label_map[label_index_og]] = key
            bbox_info = tag_info['bbox']

            # Find the correct label based on the detected april_id
            correct_label = april_tag_labels[key]

            # Break if the correct label does not match the current label
            if correct_label != label_index_og:
                break

            # Find the label with the highest IoU score for the current bounding box
            label_index = label_map[label_index_og]

            angles[label_index] = (tag_info['angle'][2][0] + 90)  # Get the yaw angle

            xy1 = (bbox_info[0], bbox_info[1])
            xy2 = (bbox_info[2], bbox_info[3])
        else:
            label_index = label_map[label_index_og]
            key = april_tag_labels_reverse[label_index_og]
            xy1 = (x1, y1)
            xy2 = (x2, y2)

        # Update the Kalman filter with the new measurements (bounding box positions and angle)
        kf = scripts.kalman_filter_utils.kalman_filters[label_index]
        angle_difference = abs(angles[label_index] - last_filtered_angles[label_index])

        angle_update_threshold = 30  # Set an appropriate threshold value in degrees

        if angle_difference < angle_update_threshold:
            state_mean, state_covariance = kf.filter_update(
                kf.initial_state_mean, kf.initial_state_covariance, [angles[label_index]]
            )
            kf.initial_state_mean = state_mean
            kf.initial_state_covariance = state_covariance

            # Retrieve the estimated angles from the updated state of the Kalman filter
            angle = state_mean[0]
            angles[label_index] = angle
            last_filtered_angles[label_index] = angle

        label = april_tag_labels[key]
        angle = (360 - angles[label_index] % 360)
        result.append({label: angle})

    merged_dict = {key: value for d in result for key, value in d.items()}

    return merged_dict
