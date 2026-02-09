# Extract frames from a video, and save them as images

import cv2
import os

def extract_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"{frame_count:08d}.jpg"), frame)
        frame_count += 1
    cap.release()
    return frame_count

if __name__ == "__main__":
    video_path = "/Users/shenyuang/Downloads/episode_000000.mp4"
    output_dir = "/Users/shenyuang/Downloads/episode_000000_frames"
    extract_frames(video_path, output_dir)