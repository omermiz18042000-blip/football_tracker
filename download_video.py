"""
Download video from YouTube using yt-dlp
Run this script to download the test video
"""

from yt_dlp import YoutubeDL
import os

# Create data folder if it doesn't exist
os.makedirs('data', exist_ok=True)

# Video URL
url = 'https://www.youtube.com/watch?v=p5Uf4k44E3w'

# Download settings
ydl_opts = {
    'format': 'best[ext=mp4]',
    'outtmpl': 'data/test_video_2.mp4',
}

print("Downloading video...")
print(f"URL: {url}")

with YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

print("Download complete! Video saved to data/test_video_2.mp4")
