import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.ops import box_convert
from torchvision import transforms
from utils import encode
import json


class YoloDataset(Dataset):
    def __init__(self, img_path, S=7, B=2, num_classes=3, eval_mode=False, class_names = ["emb", "pro", "reg"]):
        self.image_path = img_path
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.eval_mode = eval_mode
        all_files = os.listdir(img_path)
        self.annotation_files = [x for x in all_files if x.endswith(".json")]
        self.image_files = [x for x in all_files if x not in self.annotation_files]
        self.class_names = class_names
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def eval(self, eval_mode=True):
        self.eval_mode = eval_mode
        return

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # read image
        img_filename = self.image_files[idx]
        img = Image.open(f"{self.image_path}/{img_filename}", mode='r')
        img = self.transforms(img)

        # read each image's corresponding label (.json)
        labels = []
        annotation_file_name = (img_filename.split("."))[0]+".json"
        with open(f"{self.image_path}/{annotation_file_name}", 'r') as f:
            data = json.loads(f.read())
            for annot in data["annotations"]:
                c = self.class_names.index(annot["label"])
                w, h = annot["n_bb_w"], annot["n_bb_h"]
                x1, y1 = annot["n_x_min"], annot["n_y_min"]
                x2,y2 = annot["n_x_max"], annot["n_y_max"]
                converted_boxes = box_convert(
                    boxes= torch.Tensor([x2,y2,x1,y1]),
                    in_fmt="xyxy",
                    out_fmt="cxcywh"
                    )
                x, y = converted_boxes[0], converted_boxes[1]
        
                if self.eval_mode: 
                    labels.append((x, y, w, h, 1.0, c))
                else:
                    labels.append((x, y, w, h, c))
        
        if self.eval_mode: 
            return img, torch.Tensor(labels)

        encoded_labels = encode(labels, self.S, self.B, self.num_classes)  # convert label list to encoded label
        encoded_labels = torch.Tensor(encoded_labels)
        return img, encoded_labels
