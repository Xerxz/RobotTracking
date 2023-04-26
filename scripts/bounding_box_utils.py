import cv2
import numpy as np

label_map = {'v1': 0, 'v2': 1, 'v3': 2}
real_labels = ['v1', 'v2', 'v3']
april_tag_labels = {6: 'v1', 4: 'v2', 5: 'v3'}
april_tag_labels_reverse = {'v1': 6, 'v2': 4, 'v3': 5}


def get_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)

    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def get_highest_confidence_boxes(models, scaled_frame, aspect_ratio_x, aspect_ratio_y, real_labels):
    highest_confidence_boxes = {}

    for model in models:
        results = model(scaled_frame)

        for box in results[0]:
            xyxy = box.boxes.xyxy.cpu().detach().numpy()
            conf = box.boxes.conf
            cls = box.boxes.cls
            label = real_labels[int(cls)]

            x1, y1, x2, y2 = xyxy[0]

            # Adjust the bounding box coordinates according to the aspect ratio of the scaled frame
            x1 = int(x1 * aspect_ratio_x)
            y1 = int(y1 * aspect_ratio_y)
            x2 = int(x2 * aspect_ratio_x)
            y2 = int(y2 * aspect_ratio_y)

            if conf > 0.1:
                overlapping = False
                for existing_label, existing_box in highest_confidence_boxes.items():
                    if get_iou((x1, y1, x2, y2), existing_box[1:]) > 0.5:  # 0.5 is the IoU threshold for overlapping
                        overlapping = True
                        if conf > existing_box[0]:
                            highest_confidence_boxes[existing_label] = (conf, x1, y1, x2, y2)
                        break

                    # Update highest_confidence_boxes if the confidence is higher than the existing box for the label
                    # and not overlapping with any existing box.
                if not overlapping:
                    highest_confidence_boxes[label] = (conf, x1, y1, x2, y2)

    return highest_confidence_boxes


def draw_result_on_frame(frame, confidence_boxes, angles):

    for label, (conf, x1, y1, x2, y2) in confidence_boxes.items():
        conf = conf.item()

        if not angles or label not in angles:
            break

        angle = angles[label]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f"{label} {conf * 100:.0f}% {(360 - angle % 360):.0f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #if frame_id == 1:
        #    offset_x = x1
        #    offset_y = y1
        #elif frame_id == 2:
        #    offset_x = x1 + int(
        #        np.interp(x1, np.array([0, frame.shape[0] // 2, frame.shape[1]]), np.array([30, 0, -10])))
        #    offset_y = y1 + frame.shape[0] - 135
        cv2.putText(frame, f"{x1, y1}",
                    (x1, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Calculate the center of the bounding box
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Calculate the endpoint of the line
        end_x = int(center_x + 50 * np.sin(np.deg2rad(angle+90)))
        end_y = int(center_y - 50 * -np.cos(np.deg2rad(angle+90)))

        # Draw the line
        cv2.line(frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 2)


def draw_fps_on_frame(frame, fps_video, fps_model):
    # Background rectangle for Video FPS
    bg_x, bg_y, bg_w, bg_h = 0, 0, 250, 80
    cv2.rectangle(frame, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), (0, 0, 0), -1)

    # Video FPS text
    cv2.putText(frame, f"Video FPS: {fps_video:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Model FPS text
    cv2.putText(frame, f"Model FPS: {fps_model:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
