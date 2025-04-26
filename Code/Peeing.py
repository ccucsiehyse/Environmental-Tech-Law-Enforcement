import cv2
import numpy as np
import time
import mediapipe as mp
from ultralytics import YOLO
import os

# Load YOLOv10 model for person detection
yolo_weights_path = "./weights/yolov10x.pt"  # YOLOv10 weights for person detection
yolo_model = YOLO(yolo_weights_path)

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Function to detect persons using YOLOv10
def detect_person(frame):
    results = yolo_model(frame)[0]  # Get the first result
    person_detected = False
    boxes = []

    for result in results.boxes:  # Iterate through detected objects
        if int(result.cls[0]) == 0:  # Person class ID (0 is typically the class for 'person')
            x_min, y_min, x_max, y_max = map(int, result.xyxy[0])
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

# Directories for input and output
input_folder = './testvids/Peeing'
output_folder = './outputs/Peeing6'

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Check if input folder exists
if not os.path.exists(input_folder):
    print(f"Input folder '{input_folder}' does not exist.")
    exit()

# Create or open the results text file
results_file_path = './outputs/Peeing6/result.txt'
results_file = open(results_file_path, 'w')

# Write header with formatted string
results_file.write(f"{'File Name':<30}{'Peeing Detected':<20}{'Total Frames':<15}{'Processing Time (s)':<20}\n")
print("Opened results file and wrote header.")
results_file.flush()

# Process each video file in the input folder
for video_file in os.listdir(input_folder):
    if video_file.endswith('.mp4') or video_file.endswith('.avi'):
        video_path = os.path.join(input_folder, video_file)
        print(f"Processing video: {video_file}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_file}")
            continue

        # Get video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video FPS: {fps}")
        print(f"Resolution: {frame_width}x{frame_height}")

        # Output file path and codec
        if video_file.endswith('.mp4'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_file_name = f"{os.path.splitext(video_file)[0]}_output.mp4"
        elif video_file.endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_file_name = f"{os.path.splitext(video_file)[0]}_output.avi"
        else:
            print(f"Unsupported video format for file: {video_file}")
            continue

        output_path = os.path.join(output_folder, output_file_name)
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frames_per_second = int(fps)
        consecutive_seconds_needed = 6
        frames_per_segment = frames_per_second
        required_frames_for_peeing = int(0.9 * frames_per_segment)

        # Initialize tracking variables
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

            # Detect person in the frame
            person_detected, boxes = detect_person(frame)

            if person_detected:
                for box in boxes:
                    # Apply Mediapipe Pose estimation on the detected person
                    if analyze_pose_for_peeing(frame, box):
                        peeing_frames_count += 1
                        print("Peeing Detected for this Frame")

                # Draw bounding boxes for detected persons and label them 'person'
                for box in boxes:
                    x_min, y_min, w, h = box
                    cv2.rectangle(frame, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0), 2)
                    cv2.putText(frame, "person", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Peeing detection logic
            if (cap.get(cv2.CAP_PROP_POS_FRAMES) % frames_per_second) == 0:
                if peeing_frames_count >= required_frames_for_peeing:
                    consecutive_peeing_seconds += 1
                    print(f"Peeing detected for {consecutive_peeing_seconds} consecutive second(s).")
                else:
                    consecutive_peeing_seconds = 0

                peeing_frames_count = 0

            if consecutive_peeing_seconds >= consecutive_seconds_needed:
                peeing_detected = True
                break

            # Write the processed frame to the output video
            out.write(frame)

        cap.release()
        out.release()

        end_time = time.time()
        total_time = end_time - start_time

        # Determine if peeing was detected
        peeing_detected_str = "Yes" if peeing_detected else "No"

        # Write results to the text file with formatted string
        results_file.write(f"{video_file:<30}{peeing_detected_str:<20}{frame_count:<15}{total_time:<20.2f}\n")
        results_file.flush()
        print(f"Results written for video: {video_file}")

        print(f"Total time taken to process video {video_file}: {total_time:.2f} seconds")
        print(f"Peeing activity was {'detected' if peeing_detected else 'not detected'} in the video {video_file}.")

# Close the results file
results_file.close()
print("Processing completed for all videos.")
