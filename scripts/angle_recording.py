import math
import cv2
import matplotlib
from matplotlib import pyplot as plt

global ground_truth_angles
global predicted_angles

matplotlib.use('TkAgg')  # or 'Qt5Agg'


def init_angle_vars():
    global ground_truth_angles
    global predicted_angles
    ground_truth_angles = []
    predicted_angles = []


def angle_from_points(p1, p2):
    dx = p1[0] - p2[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle + 360 if angle < 0 else angle


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append((x, y))


def get_angles(frame, predicted_angle):
    # Store the two points forming a vector
    ground_truth = []

    # Display the frame and collect ground truth angles with mouse clicks
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', mouse_callback, ground_truth)

    while len(ground_truth) < 2:
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # print(f"{ground_truth}")
    cv2.destroyAllWindows()

    gt_angle = angle_from_points(ground_truth[0], ground_truth[1])
    ground_truth_angles.append(gt_angle)
    predicted_angles.append((360 + predicted_angle['v1']) % 360)


def plot_angle_histogram():

    differences = [(gt - pred + 180) % 360 - 180 for gt, pred in zip(ground_truth_angles, predicted_angles)]
    filtered_differences = [diff for diff in differences if abs(diff) <= 50]

    min_diff = int(min(filtered_differences) // 10) * 10
    max_diff = int((max(filtered_differences) // 10) + 1) * 10

    plt.hist(differences, bins=range(min_diff, max_diff+10, 10), edgecolor='black', alpha=0.7)
    plt.xlabel('Angle Difference (Degrees)')
    plt.ylabel('Frequency')
    plt.title(f'{len(filtered_differences)}: Histogram of Angle Differences Between Ground Truth and Predicted')
    plt.show()

