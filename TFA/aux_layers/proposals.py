from detectron2.modeling.proposal_generator.rpn import RPN
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
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
        
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]
        return anchors, pred_objectness_logits, pred_anchor_deltas
    
    
    def get_proposals_and_loss(self, anchors, pred_objectness_logits, pred_anchor_deltas, images, gt_instances, training):
        if training:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
            
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses
    
    

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
    proposals, proposal_losses = generator.get_proposals_and_loss(merge_anchors, merge_objectness, merge_deltas, images, gt_instances, train)
    return proposals, proposal_losses