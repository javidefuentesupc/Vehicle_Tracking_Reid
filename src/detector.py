import os
from ultralytics import YOLO
import cv2

# Construct the absolute path to the data directory
data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
# Include the filename of the video
print(f'Data dir: {data_dir}')
data_filename = r'Name of the video'
full_data_path = os.path.join(data_dir, data_filename)
print(f'full_data_path: {full_data_path}')

cap = cv2.VideoCapture(full_data_path)

ret, frame = cap.read()
if not ret:
    raise FileNotFoundError(f"Failed to read video file: {full_data_path}")

H, W, _ = frame.shape

model = YOLO('yolov8s.pt')  # Load the model
threshold = 0.7

all_detections = []  # Initialize an empty list to store the detections
frame_count = 0  # Initialize frame count

while ret:
    frame_count += 1  # Increment frame count
    results = model.predict(frame, imgsz=2560, conf=0.2, device='cuda:0')[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold and int(class_id) in [2, 5, 7]:  # Check if class_id is 2, 5, or 7
            # Store the relevant detection information
            detection = {
                "frame_number": frame_count, 
                "class_id": int(class_id),
                "confidence": score,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            }
            all_detections.append(detection)

    ret, frame = cap.read()

cap.release()

# Save the detections along with frame numbers to a text file
output_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output_files'))
output_filename = 'detection_file.txt'
output_file_path = os.path.join(output_dir, output_filename)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

with open(output_file_path, 'w') as file:
    for detection in all_detections:
        frame_number = detection['frame_number']
        class_id = detection['class_id']
        confidence = detection['confidence']
        bbox = detection['bbox']
        
        x1, y1, x2, y2 = bbox
        bb_left = x1
        bb_top = y1
        bb_width = x2 - x1
        bb_height = y2 - y1

        # Add the detection information to the file as floats
        file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
            frame_number, class_id, bb_left, bb_top, bb_width, bb_height, 1, -1.0, -1.0, -1.0))

print(f"Detections saved to {output_file_path}")
