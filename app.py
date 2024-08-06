import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import json
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
import os

# Function to initialize the tracker
def initialize_tracker(frame):
    roi = cv2.selectROI('Frame', frame, False)
    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, roi)
    return tracker, roi

# Function to process a video file and save the line path
def process_video(video_path, output_file):
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', 800, 600)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read the video: {video_path}")
        cap.release()
        return []

    tracker, roi = initialize_tracker(frame)
    centers = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(v) for v in bbox]
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            radius = int(min(w, h) / 2)
            cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)
            centers.append((center_x, center_y))
            for i in range(1, len(centers)):
                cv2.line(frame, centers[i - 1], centers[i], (0, 0, 255), 2)
        else:
            tracker, roi = initialize_tracker(frame)

        frame_resized = cv2.resize(frame, (800, 600))
        cv2.imshow('Frame', frame_resized)

        # Check if this is the last frame
        if cap.get(cv2.CAP_PROP_POS_FRAMES) >= frame_count - 1:
            # Pause on the last frame and wait for user input
            cv2.putText(frame_resized, 'Press any key to finish...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Frame', frame_resized)
            cv2.waitKey(0)
            break
        else:
            # Normal frame playback
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    with open(output_file, 'w') as f:
        json.dump(centers, f)
    
    return centers

# Function to open file dialog and get selected video files
def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title="Select Video Files", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    return file_paths

# Function to normalize paths for comparison
def normalize_path(path):
    if len(path) < 2:
        return path

    x_coords, y_coords = zip(*path)
    x_norm = (x_coords - np.min(x_coords)) / (np.max(x_coords) - np.min(x_coords))
    y_norm = (y_coords - np.min(y_coords)) / (np.max(y_coords) - np.min(y_coords))
    t = np.linspace(0, 1, len(x_norm))
    interpolator_x = interp1d(t, x_norm, kind='linear')
    interpolator_y = interp1d(t, y_norm, kind='linear')
    uniform_t = np.linspace(0, 1, 100)
    x_uniform = interpolator_x(uniform_t)
    y_uniform = interpolator_y(uniform_t)

    return list(zip(x_uniform, y_uniform))

# Function to draw the path on the image
def draw_path(image, path, color):
    for i in range(1, len(path)):
        cv2.line(image, (int(path[i-1][0] * image.shape[1]), int(path[i-1][1] * image.shape[0])), 
                        (int(path[i][0] * image.shape[1]), int(path[i][1] * image.shape[0])), color, 2)

# Function to compare the shapes of the line paths and provide a similarity score
def compare_shapes(paths):
    def calculate_similarity(path1, path2):
        path1 = np.array(path1)
        path2 = np.array(path2)

        if len(path1) != len(path2):
            return 0

        distances = [euclidean(p1, p2) for p1, p2 in zip(path1, path2)]
        average_distance = np.mean(distances)
        
        # Normalize the distance to a similarity score (1 for identical, 0 for completely different)
        max_possible_distance = np.sqrt((1**2 + 1**2))
        similarity_score = max(0, 1 - average_distance / max_possible_distance)
        
        return similarity_score

    num_paths = len(paths)
    scores = []
    for i in range(num_paths):
        for j in range(i + 1, num_paths):
            path1_normalized = normalize_path(paths[i])
            path2_normalized = normalize_path(paths[j])
            score = calculate_similarity(path1_normalized, path2_normalized)
            
            # Adjust the score based on expected range
            # Assuming scores mostly fall within 0.70 to 0.80
            min_score, max_score = 0.70, 0.80
            adjusted_score = np.clip((score - min_score) / (max_score - min_score), 0, 1)

            img_height, img_width = 600, 800
            blank_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)

            draw_path(blank_image, path1_normalized, (0, 255, 0))  # Green for first path
            draw_path(blank_image, path2_normalized, (0, 0, 255))  # Red for second path

            cv2.putText(blank_image, f'Similarity Score: {adjusted_score:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow(f'Comparison {i+1} vs {j+1}', blank_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            scores.append((i + 1, j + 1, adjusted_score))
            print(f"Comparison between video {i+1} and video {j+1}: Adjusted Similarity Score = {adjusted_score:.2f}")

    return scores

# Main function to run the video processing and comparison
def main():
    video_files = open_file_dialog()
    if not video_files:
        print("No video files selected.")
        return

    paths = []
    for i, video_path in enumerate(video_files):
        output_file = f"line_path_{i+1}.json"
        print(f"Processing {video_path}...")
        centers = process_video(video_path, output_file)
        paths.append(centers)

    print("Comparing shapes...")
    scores = compare_shapes(paths)

    for i in range(len(video_files)):
        os.remove(f'line_path_{i+1}.json')
    
if __name__ == "__main__":
    main()
