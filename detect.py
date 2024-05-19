import os
import torch
from tqdm import tqdm
from utils import draw_detection
from utils.detector import Detector
import cv2

torch.manual_seed(33)

if __name__ == "__main__":
    input_size = 483
    class_names = ["emb", "pro", "reg"]
    S, B, num_classes = 7, 2, 3
    conf, iou = 0.8, 0.5
    images_dir = "datasets/augmented_data"
    sources_dir = os.listdir(images_dir)
    sources_dir = [x for x in sources_dir if not x.endswith(".json")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = "detect"
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    for source in tqdm(sources_dir):
        detector = Detector(input_size, S, B, num_classes)
        if source.split('.')[-1] in ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tif']:
            img = cv2.imread(f"{images_dir}/{source}")
            img_name = os.path.basename(source)
            detection = detector.detect(img, conf, iou)
            if detection.size()[0] != 0:
                img = draw_detection(img, detection, class_names)
                cv2.imwrite(os.path.join(output_dir, img_name), img)
    detected = os.listdir("detect")
    print(f"[FINAL] Total Images: {len(sources_dir)} || Detected Total : {len(detected)}")
