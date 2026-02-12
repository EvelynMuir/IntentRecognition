from typing import Optional

import torch
import torch.nn as nn
import timm


class ResNet101Backbone(nn.Module):
    """ResNet101 backbone for feature extraction."""

    def __init__(
        self,
        num_classes: int = 28,
        pretrained: bool = True,
        image_size: int = 224,
    ) -> None:
        """Initialize ResNet101 Backbone.

        :param num_classes: Number of output classes (not used, kept for compatibility).
        :param pretrained: Whether to use pretrained weights.
        :param image_size: Input image size (not used, kept for compatibility).
        """
        super().__init__()

        # Load pretrained ResNet101 using timm
        if pretrained:
            resnet = timm.create_model('resnet101', pretrained=True)
        else:
            resnet = timm.create_model('resnet101', pretrained=False)
        
        # Remove the last GAP and FC layers, keep only feature extraction part
        # ResNet101 structure: conv1 -> bn1 -> relu -> maxpool -> layer1-4 -> avgpool -> fc
        # We want to keep everything except avgpool and fc (last 2 layers)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(2048, num_classes)
        
        # ResNet101 output channels: 2048
        self.out_channels = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Input tensor of shape (batch_size, 3, image_size, image_size).
        :return: Feature map of shape (batch_size, 2048, H, W).
                 For 224x224 input, output is (batch_size, 2048, 7, 7).
        """
        x = self.backbone(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

