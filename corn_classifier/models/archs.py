import torch.nn as nn
import torchvision.models as models


def get_backbone(name: str = "resnet18", pretrained: bool = True, num_classes: int = 3):
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # заменить последний слой
        in_feat = m.fc.in_features
        m.fc = nn.Linear(in_feat, num_classes)
        return m
    else:
        raise ValueError("Unknown backbone")
