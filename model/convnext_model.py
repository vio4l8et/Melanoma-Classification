import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

def load_model(weights_path="best_model.pth", device="cpu"):
    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 1)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model
