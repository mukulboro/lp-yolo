import torch
from models import YoloLP


def create_model(S, B, num_classes, model_type='ms', device='cpu'):
    model = YoloLP(S, B, num_classes)
    return model
