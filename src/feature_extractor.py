import os
import torch
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image
import sys
import time as time
# Append the directory containing your model to sys.path if needed
baseline_directory = r'Path to the folder of the repo: reid-baseline-with-syncbn'
sys.path.append(baseline_directory)

from model.baseline import Baseline

from os.path import join, exists, split

# Function to preprocess an image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = F.to_tensor(image)
    image_tensor = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image_tensor

if __name__ == '__main__':
    # Set up paths and directories
    input_dir = r'Path to the dir with the frames of the video'
    output_dir = r'Output dir'
    if not exists(output_dir):
        os.makedirs(output_dir)

    # Load the ReID model
    model_weights_path = r'Path to the weights of the model'
    model = Baseline(num_classes=751, last_stride=1, model_path=model_weights_path) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    start_time = time.time()
    # Process each detection file in the directory
    for filename in os.listdir(join(input_dir, 'img1')):
        if filename.endswith('.jpg'):
            image_path = join(input_dir, 'img1', filename)
            output_file = join(output_dir, split(filename)[0] + '_features.npy')

            # Load detections from file
            detections = np.loadtxt(r'Path to the dection file.txt', delimiter=',')

            # List to store extracted features
            all_features = []

            # Process detections in batches
            batch_size = 25  # Adjust batch size as needed
            num_detections = len(detections)
            for i in range(0, num_detections, batch_size):
                batch_detections = detections[i:i+batch_size]

                # Process each detection in the batch
                for detection in batch_detections:
                    frame_number = int(detection[0])  # Access frame number
                    image_patch = preprocess_image(image_path).to(device)
                    

                    # Perform inference and get features from your ReID model
                    with torch.no_grad():
                        features = model(image_patch.unsqueeze(0))
                        features = features.squeeze().cpu().numpy()  

                    # Append the features to the list
                    all_features.append(features)

                    # Print progress
                    print(f"Processed {len(all_features)} out of {num_detections} detections")

            # Convert the list of features to a numpy array
            all_features = np.array(all_features)

            # Save the features to a .npy file
            np.save(output_file, all_features)
            print(f"Features saved to {output_file}")

            # Print the dimensions of the saved .npy file
            print("Dimensions of saved .npy file:", all_features.shape)
    end_time = time.time()
    print(f'Elapsed time: {end_time-start_time}')
