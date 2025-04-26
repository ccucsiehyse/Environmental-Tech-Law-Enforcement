import cv2
import numpy as np
import time
import mediapipe as mp
from ultralytics import YOLO
import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Load YOLOv10 model for person detection using SAHI
yolo_weights_path = "./weights/yolov10x.pt"  # YOLOv10 weights for person detection
detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8', model_path=yolo_weights_path, confidence_threshold=0.4, device="cuda:0")

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Function to detect persons using YOLOv10 with SAHI
def detect_person_with_sahi(frame):
    # Get sliced prediction using SAHI
    result = get_sliced_prediction(frame, detection_model, slice_height=512, slice_width=512, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
    person_detected = False
    boxes = []

    for obj in result.object_prediction_list:
        if obj.category.id == 0:  # Person class ID (typically 0 for 'person')
            x_min, y_min, x_max, y_max = map(int, obj.bbox.to_voc_bbox())
            boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
            person_detected = True

    return person_detected, boxes

# Function to analyze pose and check for peeing behavior
def analyze_pose_for_peeing(frame, person_box):
    x_min, y_min, w, h = map(int, person_box)
    cropped_frame = frame[y_min:y_min + h, x_min:x_min + w]  # Crop the detected person area

    frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Draw pose landmarks on the cropped person frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

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

# Specify a single video file to process
video_file = './testvids/Peeing/peeing_28s.mp4'  # Replace this with the path to your video
output_folder = './outputs/Peeing6'
os.makedirs(output_folder, exist_ok=True)

# Check if the video file exists
if not os.path.exists(video_file):
    print(f"Video file '{video_file}' does not exist.")
    exit()

# Open the results text file
results_file_path = './outputs/Peeing6/result.txt'
results_file = open(results_file_path, 'w')
results_file.write(f"{'File Name':<30}{'Peeing Detected':<20}{'Total Frames':<15}{'Processing Time (s)':<20}\n")
results_file.flush()

# Process the specified video file
print(f"Processing video: {video_file}")

cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print(f"Error opening video file: {video_file}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_file_name = f"{os.path.splitext(os.path.basename(video_file))[0]}_output.mp4"
output_path = os.path.join(output_folder, output_file_name)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frames_per_second = int(fps)
consecutive_seconds_needed = 6
frames_per_segment = frames_per_second
required_frames_for_peeing = int(0.9 * frames_per_segment)

peeing_frames_count = 0
consecutive_peeing_seconds = 0
peeing_detected = False
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    person_detected, boxes = detect_person_with_sahi(frame)

    if person_detected:
        for box in boxes:
            if analyze_pose_for_peeing(frame, box):
                peeing_frames_count += 1

        for box in boxes:
            x_min, y_min, w, h = box
            cv2.rectangle(frame, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0), 2)
            cv2.putText(frame, "person", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if (cap.get(cv2.CAP_PROP_POS_FRAMES) % frames_per_second) == 0:
        if peeing_frames_count >= required_frames_for_peeing:
            consecutive_peeing_seconds += 1
        else:
            consecutive_peeing_seconds = 0

        peeing_frames_count = 0

    if consecutive_peeing_seconds >= consecutive_seconds_needed:
        peeing_detected = True
        break

    out.write(frame)

cap.release()
out.release()

total_time = time.time() - start_time
results_file.write(f"{os.path.basename(video_file):<30}{'Yes' if peeing_detected else 'No':<20}{frame_count:<15}{total_time:<20.2f}\n")
results_file.flush()

results_file.close()
print("Processing completed for the video.")
