import torch
import logging
from torch import nn
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from .build import META_ARCH_REGISTRY
from .gdl import decouple_layer, AffineLayer
from defrcn.modeling.roi_heads import build_roi_heads
from aux_layers.load_network import load_additional_backbone
from aux_layers.proposals import merge_proposals

__all__ = ["GeneralizedRCNN"]

@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self._SHAPE_ = self.backbone.output_shape()
        self.proposal_generator = build_proposal_generator(cfg, self._SHAPE_)
        self.roi_heads = build_roi_heads(cfg, self._SHAPE_)
        self.normalizer = self.normalize_fn()
        self.affine_rpn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.RPN.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.res5.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposal_losses, detector_losses, distill_losses, _, _ = self._forward_once_(batched_inputs, gt_instances)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        
        # Distillation Loss
        if self.cfg.MERGE.EXPAND:
            losses.update(distill_losses)
            
            for key in losses.keys():
                if 'distill' in key:
                    losses[key] = losses[key] * max(1e-6, (1 - (1 - 1e-6) * self.iter / self.cfg.SOLVER.WARMUP_ITERS))
                else:
                    losses[key] = losses[key] * min(1, (1e-6 + (1 - 1e-6) * self.iter / self.cfg.SOLVER.WARMUP_ITERS))
        return losses


    def inference(self, batched_inputs):
        assert not self.training
        _, _, _, results, image_sizes = self._forward_once_(batched_inputs, None)
        processed_results = []
        for r, input, image_size in zip(results, batched_inputs, image_sizes):
            height = input.get("height", image_size[0])
            width = input.get("width", image_size[1])
            r = detector_postprocess(r, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def _forward_once_(self, batched_inputs, gt_instances=None):
        if gt_instances is None:
            train = False
        else:
            train = True
            
        images = self.preprocess_image(batched_inputs)
        
        # Backbone Forward
        if self.cfg.MERGE.EXPAND:
            features = {}
            features_b = self.backbone(images.tensor)
            features_n = self.backbone_few(images.tensor)
            
            for key in features_b.keys():
                features[key] = (features_b[key] + features_n[key]) / 2
            
            # Backbone Distillation
            aux_loss = []            
            for key in ['p2', 'p3', 'p4', 'p5']:
                aux_loss.append(torch.mean(torch.abs(features_b[key] - features_n[key])))
            aux_loss = torch.mean(torch.stack(aux_loss))
            distill_losses = {'distill_loss': aux_loss}
            
            # Proposal Generator
            if ('base' in self.cfg.MERGE.PROPOSAL_LIST) and ('novel' in self.cfg.MERGE.PROPOSAL_LIST):
                base_anchors, base_objectness, base_deltas = self.proposal_generator(images, features_b, gt_instances)
                novel_anchors, novel_objectness, novel_deltas = self.proposal_generator_novel(images, features, gt_instances)
                proposals, proposal_losses = merge_proposals(self.proposal_generator, images, gt_instances, [base_anchors, novel_anchors], [base_objectness, novel_objectness], [base_deltas, novel_deltas], train=train)
            elif 'novel' in self.cfg.MERGE.PROPOSAL_LIST:
                proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            else:
                raise('Select Proper Proposal Types')
            
            results, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
            
        else:
            distill_losses = None
            features = self.backbone(images.tensor)

            # Proposal Network
            features_de_rpn = features
            if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
                scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
                features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
            proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

            # Roi Heads
            features_de_rcnn = features
            if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
                scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
                features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
            results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)

        return proposal_losses, detector_losses, distill_losses, results, images.image_sizes

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def normalize_fn(self):
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (torch.Tensor(
            self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1))
        pixel_std = (torch.Tensor(
            self.cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1))
        return lambda x: (x - pixel_mean) / pixel_std
