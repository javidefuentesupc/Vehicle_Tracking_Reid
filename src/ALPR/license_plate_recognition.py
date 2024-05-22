import numpy as np
import cv2
from ultralytics import YOLO
import subprocess
import os
import re

alpr_exe = r'Path to the alpr.exe'
video_path = r'Path to the video'
track_file = r'Path to the tracking file'
output_file = r'Output file'


def get_bbox_values(tracking_file, frame_number):
    bbox_values = []
    with open(tracking_file, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            current_frame = int(values[0])
            if current_frame == frame_number:
                bbox = [float(value) for value in values[2:6]]
                bbox_values.append(bbox)
    return bbox_values

def detect_license_plate_frame(img, bbox):
    try:
        # Save the bounding box image to a temporary file
        cv2.imwrite('bbox_image.jpg', bbox)
        # Run ALPR executable on the bounding box image file
        result = subprocess.run([alpr_exe, '-c', 'eu', '-n', '1', 'bbox_image.jpg'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Delete the temporary bounding box image
        os.remove('bbox_image.jpg')
        if result.returncode == 0:
            # Parse the ALPR result to extract the license plate number and confidence level
            alpr_output = result.stdout.decode()
            match = re.search(r'(\w+)\s+confidence:\s+([\d.]+)', alpr_output)
            if match:
                plate_number = match.group(1)
                confidence = float(match.group(2))
                return (plate_number, confidence)
            else:
                return None  # Return None if parsing fails
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def run(video_file, track_file, output_file):
    cap = cv2.VideoCapture(video_file)
    # Check if the output file already exists
    file_exists = os.path.isfile(output_file)
    with open(track_file, 'r') as track_file_reader, open(output_file, 'a') as output_track_file_writer:
        # Write header only if the file doesn't exist yet
        if not file_exists:
            output_track_file_writer.write("Frame, ID, X, Y, Width, Height, License Plate, Confidence\n")
        for line in track_file_reader:
            values = line.strip().split(',')
            frame_number = int(values[0])
            try:
                track_id = int(values[1])  # Extract detection ID
                x, y, width, height = map(float, values[2:6])  # Extracting bounding box values
            except ValueError:
                print(f"Error processing line: {line}")
                continue  # Skip to the next line
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, frame = cap.read()
            if not ret:
                print('Error. Unable to read more files from the video :(')
                break
            cropped_img = frame[int(y):int(y + height), int(x):int(x + width)]
            result = detect_license_plate_frame(frame, cropped_img)
            if result:
                plate_number, confidence = result
                output_track_file_writer.write(f"{frame_number}, {track_id}, {x}, {y}, {width}, {height}, {plate_number}, {confidence}\n")
                print(f"Frame {frame_number}: ID: {track_id}, License Plate: {plate_number}, Confidence: {confidence}")
            else:
                print(f"Frame {frame_number}: ID: {track_id}, No license plate detected.")
                output_track_file_writer.write(f"{frame_number}, {track_id}, {x}, {y}, {width}, {height}, None, 0\n")
    cv2.destroyAllWindows()
    cap.release()

run(video_path, track_file, output_file)
