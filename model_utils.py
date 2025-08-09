# model_utils.py
# Utilities for loading the rice disease model and preprocessing images

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from PIL import Image
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model definition (copied from your model.py, simplified for inference)
class UltraOptimizedRiceClassifier(nn.Module):
    def __init__(self, num_classes=8, dropout_rate=0.5):
        super(UltraOptimizedRiceClassifier, self).__init__()
        self.backbone = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        backbone_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.feature_extractor = nn.Sequential(
            self.backbone,
            nn.Dropout(0.2)
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=backbone_features,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(backbone_features)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.9),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(128, num_classes)
        )
        self._initialize_weights()
    def _initialize_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    def forward(self, x):
        features = self.feature_extractor(x)
        features_att = features.unsqueeze(1)
        attended_features, _ = self.attention(features_att, features_att, features_att)
        attended_features = attended_features.squeeze(1)
        attended_features = self.attention_norm(attended_features)
        combined_features = features + 0.3 * attended_features
        output = self.classifier(combined_features)
        return output

# Class names (update if your dataset changes)
CLASS_NAMES = [
    'Healthy',
    'Insect',
    'Leaf Scald',
    'Rice',
    'Rice Blast',
    'Rice Leaffolder',
    'Rice Stripes',
    'Rice Tungro'
]

# Image transforms (validation/test)
def get_val_transform():
    return transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.CenterCrop(320),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

def load_model(model_path):
    model = UltraOptimizedRiceClassifier(num_classes=len(CLASS_NAMES))
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    return model

def predict_image(model, image: Image.Image):
    transform = get_val_transform()
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
    return CLASS_NAMES[pred_idx], confidence
