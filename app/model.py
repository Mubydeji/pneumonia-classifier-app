"""
Model definition and inference logic for the pneumonia classifier.

The architecture here must exactly match what was used in training
(pneumonia_classifier.ipynb), otherwise the downloaded state_dict won't
load correctly. Single-logit ResNet-50, binary classification via sigmoid.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

THRESHOLD = 0.30  # tuned during evaluation, see the training notebook

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def build_resnet50(num_classes=1):
    """
    Builds the same architecture used in training: ResNet-50 with its
    final layer replaced for single-logit binary classification. Weights
    are loaded separately after this, so no pretrained ImageNet weights
    are needed here.
    """
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(weights_path, device):
    """
    Loads trained weights into the model architecture and sets it to
    evaluation mode. Raises clearly if the checkpoint doesn't match the
    architecture, rather than failing silently.
    """
    model = build_resnet50()
    checkpoint = torch.load(weights_path, map_location=device)

    # Supports both a raw state_dict and a full checkpoint dict
    # (the training notebook saves checkpoints as dicts with a
    # 'model_state_dict' key alongside epoch/loss metadata).
    state_dict = checkpoint.get('model_state_dict', checkpoint) \
        if isinstance(checkpoint, dict) else checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(model, image_bytes, device):
    """
    Runs inference on a single uploaded image.
    Returns the raw pneumonia probability (float, 0-1).

    Raises ValueError if the image can't be opened, so the API layer
    can return a clear 4xx error instead of a raw 500.
    """
    try:
        image = Image.open(image_bytes).convert('RGB')
    except Exception as e:
        raise ValueError(f"Unable to read image: {e}")

    tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor)
        probability = torch.sigmoid(logit).item()

    return probability
