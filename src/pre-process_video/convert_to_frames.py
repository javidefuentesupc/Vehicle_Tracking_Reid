import cv2
import os

data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data'))
processed_frames_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'processed_frames'))
#Include the filename of the video
data_filename = r'Name of the video'
full_data_path = data_dir +'/'+ data_filename
def save_frames_from_video(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"{frame_count + 1:06d}.jpg")  # Formats frame count with leading zeros
        cv2.imwrite(frame_filename, frame)
        print(f"Saved frame {frame_count + 1}")
        frame_count += 1
    cap.release()

output_folder = processed_frames_path

save_frames_from_video(full_data_path,output_folder)
