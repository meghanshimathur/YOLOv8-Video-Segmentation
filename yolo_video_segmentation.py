# --------------------------------------------------
# YOLOv8 Video Segmentation Project
# Author: Meghanshi Mathur
# Description:
# This script performs object detection on a video using YOLOv8.
# Steps:
# 1. Load YOLO model
# 2. Read input video
# 3. Detect objects frame-by-frame
# 4. Save annotated frames
# 5. Combine frames into output video
# --------------------------------------------------

model = YOLO("yolov8n.pt")   # n=nano, s=small, m=medium, l=large, x=x-large
# You can also use custom trained model later:
# model = YOLO("path/to/your_custom_model.pt")
print("Model loaded ✓")

!unzip archive.zip

!pip install -q ultralytics opencv-python

import cv2
from ultralytics import YOLO

# Load input video file
cap = cv2.VideoCapture("/content/walk.mp4")

# Process each frame and apply YOLO detection
if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

print("Starting detection... Press stop to end")

frame_id = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Finished reading video")
        break
        
# Save detected frames as images
    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    # Save frames instead of cv2.imshow
    cv2.imwrite(f"output_frame_{frame_id}.jpg", annotated_frame)
    frame_id += 1

cap.release()

# Convert saved frames back into video
print("Detection finished!")
print("Saved output frames as images.")

model = YOLO("yolov8n.pt")

import cv2
import glob

# Path of saved frames
frame_files = sorted(
    glob.glob("output_frame_*.jpg"),
    key=lambda x: int(x.split("_")[-1].split(".")[0])
)

if len(frame_files) == 0:
    print("❌ No frames found!")
    exit()

# Read first frame to get size
first_frame = cv2.imread(frame_files[0])
height, width, _ = first_frame.shape

# Output video file
output_video = "yolo_output_video.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 20   # you can change this

out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for file in frame_files:
    img = cv2.imread(file)
    out.write(img)

out.release()

print("✅ Video created successfully:", output_video)

