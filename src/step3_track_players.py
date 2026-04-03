"""
STEP 3: Track Players Across Frames
=======================================

PROJECT APPROACH:
We're building a modular pipeline: Read Frames → Detect → Track → Classify → Output
Each step builds confidence and understanding before moving to the next.
This teaches you to think about data flow and architectural decisions.

🎯 INTERVIEW TALKING POINT:
"I implemented centroid-based tracking because:
  1. It's simple enough to explain and debug (good for entry-level)
  2. Works well for ~22 players where interactions are sparse
  3. O(n²) complexity is acceptable: 22² = 484 comparisons per frame, not expensive
  4. If this were 1000s of objects, I'd use Deep SORT with neural network embeddings"

HOW IT WORKS:
1. In Frame N, detect all people (14-17 detections)
2. Calculate the CENTER POINT of each detection box (centroid)
3. In Frame N+1, detect all people again
4. For each detection in Frame N+1, find the CLOSEST detection from Frame N
5. If distance < threshold (50 pixels), it's probably the same person → assign same ID
6. Otherwise, it's a new person → assign new ID

WHY THIS WORKS FOR VIDEO:
- People barely move between consecutive frames (25-30 FPS is fast)
- In 1/30th of a second, a player moves only a few pixels
- So simple distance matching works great

🎯 INTERVIEW TALKING POINT:
"The centroid approach assumes smooth motion. If objects teleport (like in occlusion),
this breaks. For that, you'd need appearance features (Deep SORT) or Kalman filtering.
This is a trade-off discussion in interviews."
"""

import cv2
from ultralytics import YOLO
import math

# ============================================================================
# CENTROID TRACKER CLASS
# This is the heart of simple tracking
# ============================================================================

class CentroidTracker:
    """
    Simple centroid-based tracker.
    
    ALGORITHM FLOW:
    1. Keep a dictionary: currently_tracked_ids = { player_id: centroid_xy, ... }
    2. Each frame:
       a. Get new detections with their centroids
       b. Find which new detection is closest to each existing player
       c. If close enough (< threshold), it's the same player → update position
       d. If too far OR no match, it's a new player → assign new ID
       e. Delete players who haven't been seen in N frames
    
    🎯 INTERVIEW TALKING POINT:
    "This is O(n²) - for each new detection, we check all existing tracked players.
    For 22 players, that's fine. But at scale (1000s of objects), you'd use
    spatial indexing or nearest neighbor trees."
    """
    
    def __init__(self, max_distance=50, max_frames_missing=5):
        """
        max_distance: If a new detection is > 50 pixels away from a tracked player,
                      assume it's a different person
        max_frames_missing: If a player isn't detected for 5 frames, forget about them
        """
        self.next_id = 0
        self.tracked_players = {}  # { player_id: { 'centroid': (x, y), 'frames_missing': 0 } }
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing
    
    def calculate_centroid(self, x1, y1, x2, y2):
        """
        Given a bounding box (top-left and bottom-right corners),
        calculate the CENTER POINT.
        
        WHY CENTROID?
        - Simple: Just average x and y coordinates
        - Robust: Even if detection box moves slightly, centroid is stable
        - Fast: O(1) calculation
        
        For a box from (100, 50) to (120, 100):
        - Centroid = ((100+120)/2, (50+100)/2) = (110, 75)
        """
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2
        return (centroid_x, centroid_y)
    
    def euclidean_distance(self, point1, point2):
        """
        Calculate straight-line distance between two points.
        
        Formula: distance = sqrt((x2-x1)² + (y2-y1)²)
        
        WHY EUCLIDEAN?
        - Standard distance metric
        - Works in pixels: distance = pixels moved
        - Alternative: Manhattan distance (|x2-x1| + |y2-y1|) works too
        
        Example: point1=(100, 100), point2=(103, 104)
        distance = sqrt(3² + 4²) = sqrt(25) = 5 pixels
        """
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        distance = math.sqrt(dx**2 + dy**2)
        return distance
    
    def update(self, detections):
        """
        MAIN TRACKING FUNCTION - called once per frame
        
        Input: detections = [(x1, y1, x2, y2, confidence), ...]
        Output: updated player IDs
        
        THREE-STEP PROCESS:
        1. Calculate centroids for all new detections
        2. Match new detections to existing tracked players
        3. Handle unmatched detections (new players) and unmatched players (occlusion)
        """
        
        # STEP 1: Convert detection boxes to centroids
        # -----------------------------------------------
        new_centroids = {}  # { temp_id: (x, y), ... }
        for i, det in enumerate(detections):
            x1, y1, x2, y2, confidence = det
            centroid = self.calculate_centroid(x1, y1, x2, y2)
            new_centroids[i] = centroid
        
        # STEP 2: Match centroids to existing players
        # -----------------------------------------------
        
        # If no players tracked yet, assign new IDs to all detections
        if len(self.tracked_players) == 0:
            for i in new_centroids:
                self.tracked_players[self.next_id] = {
                    'centroid': new_centroids[i],
                    'frames_missing': 0
                }
                self.next_id += 1
            return self.tracked_players
        
        # Match logic: for each tracked player, find closest new detection
        matched_new_ids = set()  # Track which new detections we've matched
        
        for player_id in list(self.tracked_players.keys()):
            tracked_centroid = self.tracked_players[player_id]['centroid']
            
            # Find closest new detection
            closest_distance = float('inf')
            closest_new_id = None
            
            for new_id, new_centroid in new_centroids.items():
                # Skip if already matched to another player
                if new_id in matched_new_ids:
                    continue
                
                distance = self.euclidean_distance(tracked_centroid, new_centroid)
                
                # Keep track of the closest match
                if distance < closest_distance:
                    closest_distance = distance
                    closest_new_id = new_id
            
            # DECISION: Is this a match?
            # 🎯 INTERVIEW TALKING POINT:
            # "The 50-pixel threshold is tuned for our video characteristics.
            # At 25-30 FPS, players don't move more than 50 pixels between frames.
            # If it were slower video (10 FPS), we'd need a larger threshold."
            
            if closest_new_id is not None and closest_distance < self.max_distance:
                # YES - same player, update position
                self.tracked_players[player_id]['centroid'] = new_centroids[closest_new_id]
                self.tracked_players[player_id]['frames_missing'] = 0
                matched_new_ids.add(closest_new_id)
            else:
                # NO - this player wasn't detected this frame (occlusion or off-camera)
                self.tracked_players[player_id]['frames_missing'] += 1
        
        # STEP 3: Handle unmatched new detections
        # -----------------------------------------------
        for new_id in new_centroids:
            if new_id not in matched_new_ids:
                # This is a new player we haven't seen before
                self.tracked_players[self.next_id] = {
                    'centroid': new_centroids[new_id],
                    'frames_missing': 0
                }
                self.next_id += 1
        
        # Clean up: Remove players missing for too long
        # (They likely left the field or are permanently occluded)
        for player_id in list(self.tracked_players.keys()):
            if self.tracked_players[player_id]['frames_missing'] > self.max_frames_missing:
                del self.tracked_players[player_id]
        
        return self.tracked_players


# ============================================================================
# MAIN SCRIPT
# ============================================================================

# Load YOLO model
print("Loading YOLO model...")
model = YOLO('yolov8s.pt')
print("Model loaded!\n")

# Initialize centroid tracker
tracker = CentroidTracker(max_distance=50, max_frames_missing=5)

# Open video
video_path = 'data/test_video_2.mp4'  # Using the better video (0.84 confidence)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open {video_path}")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames\n")

# Process first 5 frames
frames_to_process = 5
frame_count = 0

print(f"Processing {frames_to_process} frames with tracking...\n")

while frame_count < frames_to_process:
    success, frame = cap.read()
    if not success:
        break
    
    frame_count += 1
    
    # Run YOLO detection
    results = model(frame, verbose=False)
    boxes = results[0].boxes.data
    
    # Convert detections to simple format: (x1, y1, x2, y2, confidence)
    detections = []
    for box in boxes:
        x1, y1, x2, y2, confidence, class_id = box.tolist()
        if int(class_id) == 0:  # Person class
            detections.append((x1, y1, x2, y2, confidence))
    
    # Update tracker with this frame's detections
    tracked_players = tracker.update(detections)
    
    # Report results
    print(f"--- Frame {frame_count} ---")
    print(f"Detections: {len(detections)} people found")
    print(f"Tracked Players: {len(tracked_players)} unique IDs\n")
    
    # Print each tracked player
    for player_id in sorted(tracked_players.keys()):
        centroid = tracked_players[player_id]['centroid']
        frames_missing = tracked_players[player_id]['frames_missing']
        print(f"  Player {player_id}: centroid=({centroid[0]:.0f}, {centroid[1]:.0f}), missing={frames_missing} frames")
    print()

cap.release()

print("Done! Tracking complete.\n")
print("KEY INSIGHTS:")
print("- Tracked Players = persistent IDs across frames")
print("- Detections = raw YOLO output (different per frame)")
print("- 'frames_missing': Counter for occlusion (useful for cleanup)\n")
print("🎯 INTERVIEW TALKING POINT:")
print("'This simple centroid tracker demonstrates the matching problem:")
print("  Given N old objects and M new objects, find best pairings.")
print("  At scale, this becomes assignment problems solved with")
print("  Hungarian algorithm or bipartite matching. For 22 players,")
print("  greedy nearest-neighbor is fast enough.'")
