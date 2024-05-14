import argparse
import os
import time
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw

from utils import load_yaml, metrics, draw_detection
from datasets import YoloDataset
from models import create_model

torch.manual_seed(0)

def evaluate(model, val_loader, S, B, num_classes):
    model.eval()  # Sets the module in evaluation mode
    ap = 0.0
    pbar = tqdm(val_loader, leave=True)
    count = 1
    for img, labels in pbar:
        preds = model(img)[0].detach().cpu()
        ap += metrics.mean_average_precision(preds, labels, S, B, num_classes)
        pbar.set_description(f"mAP = {(ap/count):.03f}")
        count += 1
    
    tqdm.write(f"Evaluation summary -- mAP = {(ap/count):.03f}")

    
if __name__ == "__main__":
  
    img_list_path = "datasets/augmented_data"
    class_names = ["emb", "pro", "reg"]
    S, B, num_classes, input_size = 7, 2, 3, 483

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # build model
    model = create_model(S, B, num_classes,)

    # get data loader
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    eval_dataset = YoloDataset(img_list_path, S, B, num_classes, transform)
    eval_dataset.eval()
    loader = DataLoader(eval_dataset)
    evaluate(model, loader, S, B, num_classes)
