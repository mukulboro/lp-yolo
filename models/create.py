import torch
from models import YoloLP


def create_model(S, B, num_classes, weight_path=None):
    if weight_path == None:
        model = YoloLP(S, B, num_classes)
        return model
