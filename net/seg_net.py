"""
Train a DeepLabV3 segmentation network
Ref: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
Paper: https://arxiv.org/pdf/1802.02611v3.pdf
"""

import torch
import torch.nn as nn
from torchvision import models

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(DeepLabV3, self).__init__()
        # load model 
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained, progress=True)
        classifier = models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)
        self.model.classifier = classifier
        # self.m = nn.LogSoftmax(dim=1)
        # self.m = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        # x = self.m(x['out'])
        return x['out']

if __name__ == "__main__":
    model = DeepLabV3(3)
    inputs = torch.rand(3,3,224,224)
    res = model(inputs)
    print(res.shape)