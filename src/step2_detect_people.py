"""
PHASE 2: DETECT PEOPLE WITH YOLO

Purpose: Use AI to find people in video frames

Why STEP 2?
- STEP 1: We learned how to READ frames
- STEP 2: Now we PROCESS frames with AI
- Next steps: Classify teams, track players, save results

What is YOLO?
- Deep learning model trained on millions of images
- Can detect: people, cars, dogs, etc.
- Pre-trained = we don't train it, just use it
- Returns: Bounding boxes + confidence scores
"""

import cv2
from ultralytics import YOLO
import sys

print("="*70)
print("PHASE 2: DETECT PEOPLE WITH YOLO")
print("="*70)

# ============================================================
# STEP 1: Load the YOLO model
# ============================================================
# This is the AI model that detects people

print("\n[1] Loading YOLO model...")
# YOLO('yolov8s.pt') loads the "small" YOLO v8 model
# 's' = small (fast), 'm' = medium, 'l' = large (accurate but slow)
# .pt = PyTorch format (the model file)
# First time: downloads ~21MB from internet
# Next time: uses cached version

model = YOLO('yolov8s.pt')
print("✅ YOLO model loaded successfully!")

# ============================================================
# STEP 2: Open the video file
# ============================================================
# Same as STEP 1 - we need to read frames

VIDEO_PATH = 'data/test_video.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ ERROR: Could not open video at {VIDEO_PATH}")
    sys.exit(1)

print(f"✅ Video opened: {VIDEO_PATH}")

# Get video properties (same as before)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo info:")
print(f"  FPS: {fps}")
print(f"  Resolution: {frame_width} x {frame_height}")
print(f"  Total frames: {total_frames}")

# ============================================================
# STEP 3: Process ONLY FIRST 5 FRAMES (for testing)
# ============================================================
# Why only 5? To test quickly without processing entire video
# Later we'll process all frames

print(f"\n[2] Processing first 5 frames with YOLO...")
print("-" * 70)

frame_count = 0
max_frames_to_test = 5  # Only process 5 frames for learning

while cap.isOpened() and frame_count < max_frames_to_test:
    success, frame = cap.read()
    
    if not success:
        break
    
    frame_count += 1
    
    # ========================================================
    # STEP 4: Run YOLO detection on this frame
    # ========================================================
    # model(frame) = "Hey YOLO, find objects in this frame"
    # results = list of detections
    
    results = model(frame, verbose=False)
    # verbose=False = don't print extra debug info
    
    # ========================================================
    # STEP 5: Extract detection information
    # ========================================================
    # results[0] = detections for this frame
    # results[0].boxes = all bounding boxes found
    
    detections = results[0].boxes.data
    # detections = tensor with shape (N, 6) where N = number of objects
    # Each row: [x1, y1, x2, y2, confidence, class_id]
    #   x1, y1 = top-left corner
    #   x2, y2 = bottom-right corner
    #   confidence = 0.0-1.0 (how sure?)
    #   class_id = what type? (0=person, 1=car, etc.)
    
    print(f"\nFrame {frame_count}:")
    print(f"  Frame shape: {frame.shape}")
    print(f"  Objects found: {len(detections)}")
    
    # ========================================================
    # STEP 6: Loop through each detection
    # ========================================================
    # For each person found in the frame
    
    for detection_idx, detection in enumerate(detections):
        # Unpack the detection values
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        
        # Convert to integers (pixel coordinates must be whole numbers)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        confidence = float(confidence)
        class_id = int(class_id)
        
        # YOLO class IDs: 0=person, 1=bicycle, 2=car, etc.
        # We only care about class 0 (people)
        class_names = {0: "Person", 1: "Bicycle", 2: "Car"}
        class_name = class_names.get(class_id, f"Unknown({class_id})")
        
        print(f"\n  Detection {detection_idx + 1}:")
        print(f"    Type: {class_name}")
        print(f"    Confidence: {confidence:.2f} ({int(confidence*100)}%)")
        print(f"    Position: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        print(f"    Size: {x2-x1}x{y2-y1} pixels")
        
        # ====================================================
        # STEP 7: Draw bounding box on frame (visual)
        # ====================================================
        # cv2.rectangle() draws a box on the frame
        # This helps us SEE what YOLO detected
        
        if class_id == 0:  # Only draw if it's a person
            # Choose color based on confidence
            if confidence > 0.9:
                color = (0, 255, 0)  # Green = very confident
            elif confidence > 0.7:
                color = (0, 255, 255)  # Yellow = confident
            else:
                color = (0, 0, 255)  # Red = less confident
            
            # Draw rectangle: (frame, top_left, bottom_right, color, thickness)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw text showing confidence
            text = f"Person: {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# ============================================================
# STEP 8: Close the video
# ============================================================
# Always clean up resources!

cap.release()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Frames processed: {frame_count}")
print(f"YOLO detection complete ✅")
print("="*70)

# ============================================================
# KEY LEARNING POINTS FOR STEP 2
# ============================================================
# 1. YOLO is a pre-trained AI model
# 2. model(frame) runs detection on one frame
# 3. Returns boxes with: coordinates, confidence, class_id
# 4. class_id=0 means person
# 5. Confidence tells us how sure YOLO is
# 6. We can draw boxes to visualize detections
# 7. Next: Track these detections across frames!
