from detectron2.modeling.backbone.resnet import build_resnet_backbone
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.config.config import CfgNode as CN
from detectron2.layers import ShapeSpec
import torch

def load_additional_backbone(cfg):
    cfg_copy = cfg.clone()
    backbone_name = cfg_copy.MERGE.BACKBONE_NAME

    # Custom Options
    if 'resnet18' in backbone_name:
        cfg_copy.MODEL.RESNETS.DEPTH = 18
        cfg_copy.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
        
    elif 'resnet34' in backbone_name:
        cfg_copy.MODEL.RESNETS.DEPTH = 34
        cfg_copy.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
        
    elif 'resnet50' in backbone_name:
        cfg_copy.MODEL.RESNETS.DEPTH = 50
        cfg_copy.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
        
    elif 'resnet101' in backbone_name:
        cfg_copy.MODEL.RESNETS.DEPTH = 101
        cfg_copy.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
        
    else:
        raise('Select Proper Backbone Type')
    
    multiplier = int(cfg.MODEL.RESNETS.RES2_OUT_CHANNELS / cfg_copy.MODEL.RESNETS.RES2_OUT_CHANNELS)
    input_shape = ShapeSpec(channels=len(cfg_copy.MODEL.PIXEL_MEAN))
    
    if 'fpn' in backbone_name:
        backbone = build_resnet_fpn_backbone(cfg_copy, input_shape)
    else:
        backbone = build_resnet_backbone(cfg_copy, input_shape)
    
    # Unfreeze All Parameters
    for param in backbone.parameters():
        param.requires_grad = True
    return backbone, multiplier