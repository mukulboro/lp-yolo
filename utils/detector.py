import time
import torch
from torchvision import transforms
from torch.profiler import profile, record_function, ProfilerActivity
from PIL import Image
from utils import decode
from models import create_model

class Detector(object):
    def __init__(self, input_size, S, B, num_classes, weight_path=None):
        super().__init__()
        self.model = create_model(S, B, num_classes, weight_path)
        self.transforms = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ])
        self.S = S
        self.B = B
        self.num_classes = num_classes

    def detect(self, img, conf, iou):
        image = Image.fromarray(img).convert('RGB')
        image = self.transforms(image)
        image.unsqueeze_(0)
        pred = self.model(image)
        pred = pred[0].detach().cpu()
        decoded_pred = decode(pred, self.S, self.B, self.num_classes, conf, iou)
        return decoded_pred