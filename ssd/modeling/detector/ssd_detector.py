from torch import nn
import torch

from ssd.modeling.backbone import build_backbone
from ssd.modeling.box_head import build_box_head
from ssd.modeling.neck import build_neck


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg) # vgg.py
        self.neck = build_neck(cfg) # build neck
        self.box_head = build_box_head(cfg) # box_head.py and box_predictor.py
        

    def forward(self, images, targets=None):
        features = self.backbone(images)
        if self.neck:
            att_features = self.neck(features)
            detections, detector_losses = self.box_head(att_features, targets)
        else:
            detections, detector_losses = self.box_head(features, targets) # NOTE:Original

        if self.training:
            return detector_losses
        return detections
