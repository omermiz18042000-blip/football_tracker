"""
STEP 2: Detect People in Video using YOLO
This version uses the second test video for comparison
"""

import cv2
from ultralytics import YOLO

# Load the pre-trained YOLO model
# YOLOv8s = small model (fast, good balance of speed and accuracy)
print("Loading YOLO model...")
model = YOLO('yolov8s.pt')  # Automatically downloads if not found
print("Model loaded!\n")

# Open the video file
video_path = 'data/test_video_2.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video Properties:")
print(f"  FPS: {fps}")
print(f"  Resolution: {width}x{height}")
print(f"  Total Frames: {total_frames}\n")

# Process only first 5 frames for testing
frames_to_process = 5
frame_count = 0

print(f"Processing first {frames_to_process} frames...\n")

while frame_count < frames_to_process:
    success, frame = cap.read()
    
    # If we can't read the frame, we've reached the end
    if not success:
        print("End of video reached")
        break
    
    frame_count += 1
    
    # Run YOLO detection on this frame
    # verbose=False means don't print detection info
    results = model(frame, verbose=False)
    
    # Extract detection boxes from results
    # Format: [x1, y1, x2, y2, confidence, class_id]
    boxes = results[0].boxes.data
    
    # Count people detected (class_id = 0 is person in COCO dataset)
    people_count = 0
    
    print(f"--- Frame {frame_count} ---")
    
    # Loop through each detection
    for detection in boxes:
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        class_id = int(class_id)
        
        # Only count people (class_id = 0)
        if class_id == 0:
            people_count += 1
            
            # Calculate size of detection box
            width_pixels = int(x2 - x1)
            height_pixels = int(y2 - y1)
            
            print(f"  Detection {people_count}: Confidence={confidence:.2f}, Size={width_pixels}x{height_pixels}px")
    
    print(f"Total people found: {people_count}\n")

# Clean up
cap.release()

print("Done! Analysis complete.")
print("\nKey observations:")
print("- Compare confidence scores and detection count with first video")
print("- Note any differences in player sizes")
print("- Ready to move to STEP 3: Player Tracking")
