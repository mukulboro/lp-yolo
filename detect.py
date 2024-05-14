import os
import shutil
import torch
import cv2
from utils import load_yaml, draw_detection
from utils.detector import Detector

torch.manual_seed(33)

if __name__ == "__main__":
    input_size = 483
    class_names = ["emb", "pro", "reg"]
    S, B, num_classes = 7, 2, 3
    conf, iou = 1e-10, 1e-10
    
    source = "datasets/augmented_data/IMG_20240412_095426_contrast.jpg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = "detect"
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    detector = Detector(input_size, S, B, num_classes)
    # Image Detection
    if source.split('.')[-1] in ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tif']:
        img = cv2.imread(source)
        img_name = os.path.basename(source)
        detection = detector.detect(img, conf, iou)
        if detection.size()[0] != 0:
            img = draw_detection(img, detection, class_names)
            cv2.imwrite(os.path.join(output_dir, img_name), img)
