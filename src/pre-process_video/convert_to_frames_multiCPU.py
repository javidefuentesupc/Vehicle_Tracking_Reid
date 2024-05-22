import cv2
import os
from multiprocessing import Pool

data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data'))
#Include the filename of the video
data_filename = r'Name of the video'
full_data_path = data_dir +'/'+ data_filename
processed_frames_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'processed_frames'))

def process_frame(frame_number, video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        frame_filename = os.path.join(output_folder, f"{frame_number + 1:06d}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved frame {frame_number + 1}")

def save_frames_from_video(video_path, output_folder, start_frame=0, end_frame=None, num_processes=4):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set end_frame if not specified
    if end_frame is None:
        end_frame = total_frames
    
    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_numbers = list(range(start_frame, end_frame))
    
    with Pool(processes=num_processes) as pool:
        pool.starmap(process_frame, [(frame_number, video_path, output_folder) for frame_number in frame_numbers])
    
    cap.release()

output_folder = processed_frames_path

save_frames_from_video(full_data_path, output_folder)
