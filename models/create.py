import torch
from models import YoloLP


def create_model(S, B, num_classes, weight_path=None):
    model = YoloLP(S, B, num_classes)
    if weight_path == None:
        return model
    model.load_state_dict(torch.load(weight_path))
    return model
