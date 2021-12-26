from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

# adding additional default values built on top of the default values in detectron2

_CC = _C

# FREEZE Parameters
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.PROPOSAL_GENERATOR.FREEZE = False
_CC.MODEL.ROI_HEADS.FREEZE_FEAT = False

# choose from "FastRCNNOutputLayers" and "CosineSimOutputLayers"
_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
# scale of cosine similarity (set to -1 for learnable scale)
_CC.MODEL.ROI_HEADS.COSINE_SCALE = 20.0

# Backward Compatible options.
_CC.MUTE_HEADER = True


# Merge PARAMS
_CC.MERGE = CN()
_CC.MERGE.EXPAND = False
_CC.MERGE.PROPOSAL_LIST = []
_CC.MERGE.BACKBONE_NAME = 'resnet18_fpn'

_CC.DATASET = ''
_CC.METHOD = ''