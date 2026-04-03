"""
PHASE 1: VIDEO FRAME READER

Purpose: Read a video file and understand its structure

Why we do this FIRST:
- We need to understand what data we're working with
- Frame rate, resolution, total frames = crucial info
- This is the INPUT to all other processing

Think of it like: Inspecting ingredients before cooking
"""

import cv2  # OpenCV library for video/image processing
import sys  # System utilities

# ============================================================
# STEP 1: Define the path to our video
# ============================================================
# Videos should be in data/ folder (clean organization)

VIDEO_PATH = 'data/test_video.mp4'

print("="*60)
print("PHASE 1: VIDEO FRAME READER")
print("="*60)
print(f"\nAttempting to open: {VIDEO_PATH}")

# ============================================================
# STEP 2: Open the video file
# ============================================================
# cv2.VideoCapture() creates a "reader" object
# Think of it like opening a book to read it page-by-page

cap = cv2.VideoCapture(VIDEO_PATH)

# Check if video opened successfully
if not cap.isOpened():
    print(f"❌ ERROR: Could not open video at {VIDEO_PATH}")
    print("Make sure test_video.mp4 is in the 'data/' folder")
    sys.exit(1)  # Exit the program

print("✅ Video opened successfully!")

# ============================================================
# STEP 3: Extract video properties
# ============================================================
# These tell us about the video characteristics

fps = int(cap.get(cv2.CAP_PROP_FPS))
# FPS = Frames Per Second
# Why important? 
# - If we process slower than FPS, video will stutter
# - If we save output, we need this to match speed

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Resolution: width x height in pixels
# Why important?
# - Affects detection accuracy (small players hard to detect)
# - Affects processing speed (larger = slower)
# - Needed if we save output video

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Total frames in video
# Why important?
# - Tells us how much data to process
# - Can estimate processing time

duration_seconds = total_frames / fps
# Calculate duration
# Useful for showing progress

# ============================================================
# STEP 4: Display video information
# ============================================================

print(f"\nVIDEO INFORMATION:")
print(f"  FPS:            {fps} frames/second")
print(f"  Resolution:     {frame_width} x {frame_height} pixels")
print(f"  Total frames:   {total_frames}")
print(f"  Duration:       {duration_seconds:.2f} seconds")

# ============================================================
# STEP 5: Read frames one-by-one
# ============================================================
# This is the core of what we do: process each frame

frame_count = 0
successful_reads = 0
max_frames = 10 * fps  # Read frames for 10 seconds
print(f"\nReading frames...")

while True:
    # Read ONE frame from the video
    success, frame = cap.read()
    # success = True if we got a frame, False if video ended or error
    # frame = numpy array containing the pixel data
    
    if not success:
        # Video ended or couldn't read frame
        break
    
    frame_count += 1
    successful_reads += 1
    
    # Stop after 10 seconds
    if frame_count >= max_frames:
        print(f"  Reached 10 second limit. Stopping.")
        break
    
    # Print progress every 30 frames (not too spammy)
    if frame_count % 30 == 0:
        percentage = (frame_count / total_frames) * 100
        print(f"  Frame {frame_count}/{total_frames} ({percentage:.1f}%)")
        print(f"    - Frame shape: {frame.shape} (height, width, channels)")
        print(f"    - Data type: {frame.dtype}")

# ============================================================
# STEP 6: Close the video
# ============================================================
# Always close files when done!
# Why?
# - Releases system resources (memory, file handles)
# - Other processes can now access the file
# - Good coding practice

cap.release()

# ============================================================
# STEP 7: Summary
# ============================================================

print(f"\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total frames read:    {successful_reads}")
print(f"Video duration:       {duration_seconds:.2f} seconds")
print(f"Processing complete ✅")
print("="*60)

# ============================================================
# KEY LEARNING POINTS
# ============================================================
# 1. cv2.VideoCapture() opens a video file
# 2. cap.get() retrieves video properties
# 3. Loop with cap.read() gets frames one-by-one
# 4. Each frame is a numpy array (image data)
# 5. Always cap.release() when done
# 6. Video is just image sequences displayed quickly
