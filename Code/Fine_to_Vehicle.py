import cv2
import numpy as np
import time
import mediapipe as mp
from ultralytics import YOLO
import re
from paddleocr import PaddleOCR
import os

# Initialize PaddleOCR for license plate text recognition
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load YOLOv10 model for person and license plate detection
yolo_weights_object = "./weights/yolov10x.pt"  # YOLOv10 weights for person detection
yolo_weights_license_plate = "./weights/license150.pt"  # YOLOv10 weights for license plate detection

yolo_model_object = YOLO(yolo_weights_object)
yolo_model_license_plate = YOLO(yolo_weights_license_plate)

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Function to format license plate textyl
def format_plate_text(text):
    """Formats the text to ensure it has a hyphen and replaces specific characters with a hyphen."""
    text = re.sub(r'[.,]', '-', text)
    if '-' not in text:
        mid = len(text) // 2
        formatted_text = text[:mid] + '-' + text[mid:]
        return formatted_text
    return text

# Function to detect persons using YOLOv10
def detect_person(frame):
    results = yolo_model_object(frame)[0]  # Get the first result
    person_detected = False
    boxes = []

    for result in results.boxes:  # Iterate through detected objects
        if int(result.cls[0]) == 0:  # Person class ID (0 is typically the class for 'person')
            x_min, y_min, x_max, y_max = map(int, result.xyxy[0])
            boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
            person_detected = True

    return person_detected, boxes

# Function to detect license plates using YOLOv10
def detect_license_plate(frame):
    results = yolo_model_license_plate(frame)
    plates = []

    for result in results:
        for box in result.boxes:
            if box.conf > 0.5 and int(box.cls) == 0:  # Assuming '0' is license plate class
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                plates.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

    return plates

# Function to analyze pose and check for peeing behavior
def analyze_pose_for_peeing(frame, person_box):
    x_min, y_min, w, h = map(int, person_box)
    cropped_frame = frame[y_min:y_min + h, x_min:x_min + w]  # Crop the detected person area

    frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        if is_standing(left_hip, right_hip, left_knee, right_knee) and hand_near_groin(left_wrist, right_wrist, left_hip, right_hip):
            return True

    return False

# Function to check if the person is standing
def is_standing(left_hip, right_hip, left_knee, right_knee):
    return left_hip.y < left_knee.y and right_hip.y < right_knee.y

# Function to check if hands are near the groin area
def hand_near_groin(left_wrist, right_wrist, left_hip, right_hip):
    groin_y = (left_hip.y + right_hip.y) / 2
    return abs(left_wrist.y - groin_y) < 0.1 or abs(right_wrist.y - groin_y) < 0.1

# Function to process video for both license plate and peeing detection
def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Variables for peeing detection logic
    frames_per_second = int(fps)
    consecutive_seconds_needed = 5
    frames_per_segment = frames_per_second
    required_frames_for_peeing = int(0.9 * frames_per_segment)  # 90% of the frames

    peeing_frames_count = 0
    consecutive_peeing_seconds = 0
    peeing_confirmed = False
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect persons
        person_detected, person_boxes = detect_person(frame)

        # Detect license plates
        plates = detect_license_plate(frame)

        # Store detected plates and their texts
        detected_plates = []

        for (x, y, w, h) in plates:
            plate_img = frame[y:y + h, x:x + w]
            result = ocr.ocr(plate_img, cls=True)

            if result and len(result) > 0 and result[0] is not None:
                text = result[0][0][1][0]
                confidence = result[0][0][1][1]

                # Format text
                formatted_text = format_plate_text(text)
                detected_plates.append((formatted_text, confidence, (x, y, w, h)))

        # Analyze peeing behavior and draw on the frame
        for box in person_boxes:
            if analyze_pose_for_peeing(frame, box):
                peeing_frames_count += 1

        # Every second, evaluate if peeing was detected in 90% of frames
        if frame_count % frames_per_segment == 0:
            if peeing_frames_count >= required_frames_for_peeing:
                consecutive_peeing_seconds += 1
                print(f"Peeing detected for {consecutive_peeing_seconds} consecutive second(s).")
            else:
                consecutive_peeing_seconds = 0  # Reset if not detected in 90% of frames

            peeing_frames_count = 0  # Reset for the next second

        # If peeing is detected for 5 consecutive seconds, confirm the behavior
        if consecutive_peeing_seconds >= consecutive_seconds_needed:
            peeing_confirmed = True

        # Draw on frame
        for box in person_boxes:
            if peeing_confirmed:
                cv2.putText(frame, "Peeing Confirmed", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                if analyze_pose_for_peeing(frame, box):
                    cv2.putText(frame, "Peeing Behavior Detected", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw all detected plates on the frame
        for formatted_text, confidence, (x, y, w, h) in detected_plates:
            # Draw bounding box for license plate
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Display the formatted text
            cv2.putText(frame, f"{formatted_text} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
process_video('./testvids/Peeing/Peeing 29s.mp4', 'output_video.mp4')
