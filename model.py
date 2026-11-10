"""Detector construction (torchvision Faster R-CNN) and the tiny TTA adapter."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
import torch.nn as nn

def _freeze_bn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            f = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            f.weight.copy_(child.weight.data); f.bias.copy_(child.bias.data)
            f.running_mean.copy_(child.running_mean); f.running_var.copy_(child.running_var)
            setattr(module, name, f)
        else:
            _freeze_bn(child)

def build_model(num_classes, backbone="mobilenet", pretrained=True, min_size=512, max_size=1024):
    """num_classes EXCLUDES background. backbone: 'mobilenet' (fast) or 'resnet50' (accurate)."""
    from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_fpn,
                                              fasterrcnn_resnet50_fpn_v2)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    w = "DEFAULT" if pretrained else None
    kw = dict(min_size=min_size, max_size=max_size, box_score_thresh=0.05)
    if backbone == "resnet50":
        m = fasterrcnn_resnet50_fpn_v2(weights=w, weights_backbone=w, **kw)
    elif backbone == "mobilenet":
        m = fasterrcnn_mobilenet_v3_large_fpn(weights=w, weights_backbone=w, **kw)
    else:
        raise ValueError(backbone)
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes + 1)
    return m


class FeatureAffine(nn.Module):
    """Wraps a detector backbone and adds a learnable per-channel scale/shift on every FPN
    level (identity at init). Only these ~2-3k parameters are updated at test time, which
    keeps adaptation cheap and hard to destabilise."""

    def __init__(self, backbone):
        super().__init__()
        self.body = backbone
        self.out_channels = backbone.out_channels
        with torch.no_grad():
            keys = list(backbone.eval()(torch.zeros(1, 3, 128, 128, device=next(backbone.parameters()).device)).keys())
        self.keys_ = keys
        self.gamma = nn.ParameterDict({f"l{k}": nn.Parameter(torch.ones(self.out_channels)) for k in keys})
        self.beta = nn.ParameterDict({f"l{k}": nn.Parameter(torch.zeros(self.out_channels)) for k in keys})

    def forward(self, x):
        feats = self.body(x)
        return type(feats)((k, v * self.gamma[f"l{k}"].view(1, -1, 1, 1) + self.beta[f"l{k}"].view(1, -1, 1, 1))
                           for k, v in feats.items())

    def adapt_parameters(self):
        return list(self.gamma.values()) + list(self.beta.values())


def wrap_for_tta(model):
    """Call AFTER loading the checkpoint (it changes state-dict keys)."""
    model.backbone = FeatureAffine(model.backbone)
    return model


def load_checkpoint(path, device="cpu", pretrained=False, backbone=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(len(ck["classes"]), backbone or ck["backbone"], pretrained=pretrained,
                        min_size=ck["min_size"], max_size=ck["max_size"])
    model.load_state_dict(ck["model"])
    _freeze_bn(model)
    return model.to(device), ck


def save_checkpoint(path, model, classes, backbone, min_size, max_size, extra=None):
    torch.save({"model": model.state_dict(), "classes": classes, "backbone": backbone,
                "min_size": min_size, "max_size": max_size, **(extra or {})}, path)
