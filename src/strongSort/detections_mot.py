import torch
import cv2
import os
import numpy as np

from yolox.data.data_augment import preproc as preprocess
from yolox.exp import get_exp

def load_model(exp_file):
    exp = get_exp(exp_file, "yolox")
    model = exp.get_model()
    model.cuda()
    model.eval()
    return model

def detect_objects(model, img):
    img_info = {'height': img.shape[0], 'width': img.shape[1]}
    img = preprocess(img, None, 640, mode='val')
    img = torch.from_numpy(img).unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(img)

    return output, img_info

def save_detections_to_mot(detections, img_info, mot_file):
    with open(mot_file, 'w') as f:
        frame_number = 1
        for detection in detections:
            boxes = detection[:, 0:4]
            scores = detection[:, 4]
            labels = detection[:, 5]

            for box, score, label in zip(boxes, scores, labels):
                left, top, right, bottom = box
                left, top, right, bottom = int(left), int(top), int(right), int(bottom)
                f.write(f"{frame_number},{int(label)+1},{left},{top},{right-left},{bottom-top},{score},-1,-1,-1\n")
            frame_number += 1

def main(frames_dir, exp_file, mot_file):
    model = load_model(exp_file)

    frame_files = sorted(os.listdir(frames_dir))

    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)

        detections, img_info = detect_objects(model, frame)

        save_detections_to_mot(detections, img_info, mot_file)

if __name__ == "__main__":
    frames_dir ='/home/usuaris/imatge/javier.de.fuentes/StrongSORT/full_dataset/MOT17/test/MOT17-01/img1'
    exp_file = "yolox_s.pth.tar"  # Path to your YOLOX experiment file
    mot_file = "YOLO_detections.txt"
    main(frames_dir, exp_file, mot_file)

