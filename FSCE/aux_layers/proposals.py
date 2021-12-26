from fsdet.modeling.proposal_generator.rpn import RPN
from fsdet.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
import torch
from detectron2.config import configurable
from copy import deepcopy


@PROPOSAL_GENERATOR_REGISTRY.register()
class Merge_RPN(RPN):
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
    ):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.
        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        return anchors, pred_objectness_logits, pred_anchor_deltas
    
    
def merge_proposals(generator, images, gt_instances, anchor_list=[], objectness_list=[], delta_list=[], train=True):
    merge_anchors = anchor_list[0]
    
    # objectness
    merge_objectness = []
    for ix in range(len(objectness_list[0])):
        merge_objectness.append((objectness_list[0][ix] + objectness_list[1][ix]) / 2)
    
    # deltas
    merge_deltas = []
    for ix in range(len(delta_list[0])):
        merge_deltas.append((delta_list[0][ix] + delta_list[1][ix]) / 2)
    
    # Forward Merged Proposals
    proposals, proposal_losses = generator.predict_proposals(merge_anchors, merge_objectness, merge_deltas, images, gt_instances, train)
    return proposals, proposal_losses