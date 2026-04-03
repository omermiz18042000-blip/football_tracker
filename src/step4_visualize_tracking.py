"""
STEP 4: Visualize Tracking - Draw Boxes & Save Video
======================================================

PROJECT APPROACH:
Now we take everything we've built (detection + tracking) and make it VISUAL.
This is the payoff moment - seeing your algorithm actually work on the video.

🎯 INTERVIEW TALKING POINT:
"Visualization is crucial in ML. You need to SEE what your model is doing to debug it.
When you see low confidence players or ID flickering, you know to:
  - Lower confidence threshold
  - Increase tracking distance
  - Or collect better training data"

WHAT THIS DOES:
1. Runs detection on each frame
2. Runs tracking to get persistent player IDs
3. DRAWS colored rectangles around each player
4. LABELS each with: Player ID + Confidence
5. SAVES annotated video to output/annotated_video.mp4

This is the visual proof that your pipeline works!
"""

import cv2
from ultralytics import YOLO
import math
import os

# ============================================================================
# CENTROID TRACKER (copied from STEP 3)
# ============================================================================

class CentroidTracker:
    def __init__(self, max_distance=50, max_frames_missing=5):
        self.next_id = 0
        self.tracked_players = {}
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing
    
    def calculate_centroid(self, x1, y1, x2, y2):
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2
        return (centroid_x, centroid_y)
    
    def euclidean_distance(self, point1, point2):
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        distance = math.sqrt(dx**2 + dy**2)
        return distance
    
    def update(self, detections):
        new_centroids = {}
        for i, det in enumerate(detections):
            x1, y1, x2, y2, confidence = det
            centroid = self.calculate_centroid(x1, y1, x2, y2)
            new_centroids[i] = centroid
        
        if len(self.tracked_players) == 0:
            for i in new_centroids:
                self.tracked_players[self.next_id] = {
                    'centroid': new_centroids[i],
                    'frames_missing': 0
                }
                self.next_id += 1
            return self.tracked_players
        
        matched_new_ids = set()
        
        for player_id in list(self.tracked_players.keys()):
            tracked_centroid = self.tracked_players[player_id]['centroid']
            closest_distance = float('inf')
            closest_new_id = None
            
            for new_id, new_centroid in new_centroids.items():
                if new_id in matched_new_ids:
                    continue
                
                distance = self.euclidean_distance(tracked_centroid, new_centroid)
                
                if distance < closest_distance:
                    closest_distance = distance
                    closest_new_id = new_id
            
            if closest_new_id is not None and closest_distance < self.max_distance:
                self.tracked_players[player_id]['centroid'] = new_centroids[closest_new_id]
                self.tracked_players[player_id]['frames_missing'] = 0
                matched_new_ids.add(closest_new_id)
            else:
                self.tracked_players[player_id]['frames_missing'] += 1
        
        for new_id in new_centroids:
            if new_id not in matched_new_ids:
                self.tracked_players[self.next_id] = {
                    'centroid': new_centroids[new_id],
                    'frames_missing': 0
                }
                self.next_id += 1
        
        for player_id in list(self.tracked_players.keys()):
            if self.tracked_players[player_id]['frames_missing'] > self.max_frames_missing:
                del self.tracked_players[player_id]
        
        return self.tracked_players


# ============================================================================
# MAIN VISUALIZATION SCRIPT
# ============================================================================

# Create output folder if it doesn't exist
os.makedirs('output', exist_ok=True)

# Load YOLO model
print("Loading YOLO model...")
model = YOLO('yolov8s.pt')
print("Model loaded!\n")

# Initialize tracker
tracker = CentroidTracker(max_distance=50, max_frames_missing=5)

# Open input video
video_path = 'data/test_video_2.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open {video_path}")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input Video: {width}x{height} @ {fps} FPS, {total_frames} frames")

# Create video writer for output
# FourCC code 'mp4v' = MPEG-4 video codec
output_path = os.path.abspath('output/annotated_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Output Video: {output_path}\n")

# Process entire video
frames_to_process = total_frames
frame_count = 0

print(f"Processing {frames_to_process} frames (this may take a minute)...\n")

while frame_count < frames_to_process:
    success, frame = cap.read()
    if not success:
        break
    
    frame_count += 1
    
    # Run YOLO detection
    results = model(frame, verbose=False)
    boxes = results[0].boxes.data
    
    # Convert detections
    detections = []
    detection_boxes = []  # Store original box data for drawing
    
    for box in boxes:
        x1, y1, x2, y2, confidence, class_id = box.tolist()
        if int(class_id) == 0:  # Person class
            detections.append((x1, y1, x2, y2, confidence))
            detection_boxes.append((int(x1), int(y1), int(x2), int(y2), confidence))
    
    # Update tracker
    tracked_players = tracker.update(detections)
    
    # 🎯 DRAWING ON FRAME: This is the visualization magic
    # ====================================================
    
    # For each detection, we need to know which player ID it belongs to
    # We'll match by finding closest centroid
    
    for i, (x1, y1, x2, y2, confidence) in enumerate(detection_boxes):
        detection_centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Find which tracked player this belongs to
        best_player_id = None
        best_distance = float('inf')
        
        for player_id, player_data in tracked_players.items():
            player_centroid = player_data['centroid']
            dist = math.sqrt(
                (detection_centroid[0] - player_centroid[0])**2 +
                (detection_centroid[1] - player_centroid[1])**2
            )
            if dist < best_distance:
                best_distance = dist
                best_player_id = player_id
        
        # COLOR CODING by confidence
        # WHY COLOR CODING?
        # 🎯 INTERVIEW TALKING POINT:
        # "In visualization, use colors to encode information.
        # High confidence (green) = high quality detection.
        # Low confidence (red) = model is uncertain.
        # This helps debug at a glance."
        
        if confidence > 0.7:
            color = (0, 255, 0)  # Green = high confidence
        elif confidence > 0.5:
            color = (0, 255, 255)  # Yellow = medium confidence
        else:
            color = (0, 0, 255)  # Red = low confidence
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw player ID and confidence
        label = f"P{best_player_id} ({confidence:.2f})"
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Write annotated frame to output video
    out.write(frame)
    
    # Print progress every 50 frames
    if frame_count % 50 == 0:
        print(f"  Processed {frame_count}/{frames_to_process} frames...")

# Clean up
cap.release()
out.release()

print(f"\n✅ Done! Video saved to: {output_path}")
print(f"\nOpening video...\n")

# Automatically open the video file with default player
import subprocess
import platform

try:
    if platform.system() == 'Windows':
        os.startfile(output_path)
    elif platform.system() == 'Darwin':  # macOS
        subprocess.Popen(['open', output_path])
    else:  # Linux and others
        subprocess.Popen(['xdg-open', output_path])
except Exception as e:
    print(f"Could not auto-open video: {e}")
    print(f"Open manually: {output_path}")
print("COLOR LEGEND:")
print("  🟢 Green  = High confidence (>0.7)")
print("  🟡 Yellow = Medium confidence (0.5-0.7)")
print("  🔴 Red    = Low confidence (<0.5)\n")
print("🎯 INTERVIEW TALKING POINT:")
print("'This visualization step is where you catch bugs:")
print("  - If IDs flicker: tracking threshold too loose")
print("  - If confidence too low: need better camera angle")
print("  - If boxes missing people: detection model needs tuning'")
