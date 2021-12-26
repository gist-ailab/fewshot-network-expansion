import logging

import torch
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from torch import nn
from aux_layers.load_network import load_additional_backbone
from aux_layers.proposals import merge_proposals

from fsdet.modeling.roi_heads import build_roi_heads

# avoid conflicting with the existing GeneralizedRCNN module in Detectron2
from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.iter = 0
        
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")
            
        # Addon Networks
        if self.cfg.MERGE.EXPAND:
            self.load_backbone()
            self.load_proposal()

        self.to(self.device)

    def load_backbone(self):
        self.backbone_size = {}
        self.backbone_size['base'] = self.backbone.output_shape()
        
        # Novel samples' Backbone
        self.backbone_few, self.multiplier = load_additional_backbone(self.cfg)
        
        # Unfreeze Backbone
        for p in self.backbone_few.parameters():
            p.requires_grad = True
        
        self.backbone_size['few'] = self.backbone_few.output_shape()
    
    
    def load_proposal(self):
        if ('base' in self.cfg.MERGE.PROPOSAL_LIST) and ('novel' in self.cfg.MERGE.PROPOSAL_LIST):
            del self.proposal_generator
            
            cfg_copy = self.cfg.clone()
            cfg_copy.MODEL.PROPOSAL_GENERATOR.NAME = "Merge_RPN"
            self.proposal_generator = None
            self.proposal_generator_base = build_proposal_generator(cfg_copy, self.backbone.output_shape())
            self.proposal_generator_novel = build_proposal_generator(cfg_copy, self.backbone.output_shape())
                    
            if self.cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
                for p in self.proposal_generator_base.parameters():
                    p.requires_grad = False
                print("froze proposal generator parameters")
                
            for p in self.proposal_generator_novel.parameters():
                p.requires_grad = True
            
        
        elif 'novel' in self.cfg.MERGE.PROPOSAL_LIST:
            # Unfreeze Proposal Generator for Novel sets
            for p in self.proposal_generator.parameters():
                p.requires_grad = True
            
        else:
            raise('Select Proper Proposal Types')
        
        
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None


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
            
        else:
            features_b, features_n = None, None
            features = self.backbone(images.tensor)

        # RPN forward
        if self.proposal_generator or self.proposal_generator_base or self.proposal_generator_novel:
            if self.cfg.MERGE.EXPAND:
                if ('base' in self.cfg.MERGE.PROPOSAL_LIST) and ('novel' in self.cfg.MERGE.PROPOSAL_LIST):
                    base_anchors, base_objectness, base_deltas = self.proposal_generator_base(images, features_b, gt_instances)
                    novel_anchors, novel_objectness, novel_deltas = self.proposal_generator_novel(images, features, gt_instances)
                    proposals, proposal_losses = merge_proposals(self.proposal_generator_base, images, gt_instances, [base_anchors, novel_anchors], [base_objectness, novel_objectness], [base_deltas, novel_deltas], train=True)
                elif 'novel' in self.cfg.MERGE.PROPOSAL_LIST:
                    proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
                
                else:
                    raise('Select Proper Proposal Types')
                
            else:
                proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [
                x["proposals"].to(self.device) for x in batched_inputs
            ]
            proposal_losses = {}


        # RoI Heads
        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances
        )

        # Loss
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

    def inference(
        self, batched_inputs, detected_instances=None, do_postprocess=True
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        
        # Backbone Forward
        if self.cfg.MERGE.EXPAND:
            features = {}
            features_b = self.backbone(images.tensor)
            features_n = self.backbone_few(images.tensor)
            
            for key in features_b.keys():
                features[key] = (features_b[key] + features_n[key]) / 2
            
        else:
            features_b, features_n = None, None
            features = self.backbone(images.tensor)
            
        if detected_instances is None:
            # RPN forward
            if self.proposal_generator or self.proposal_generator_base or self.proposal_generator_novel:
                if self.cfg.MERGE.EXPAND:
                    if ('base' in self.cfg.MERGE.PROPOSAL_LIST) and ('novel' in self.cfg.MERGE.PROPOSAL_LIST):
                        base_anchors, base_objectness, base_deltas = self.proposal_generator_base(images, features_b, gt_instances=None)
                        novel_anchors, novel_objectness, novel_deltas = self.proposal_generator_novel(images, features, gt_instances=None)
                        proposals, _ = merge_proposals(self.proposal_generator_base, images, None, [base_anchors, novel_anchors], [base_objectness, novel_objectness], [base_deltas, novel_deltas], train=False)
                    elif 'novel' in self.cfg.MERGE.PROPOSAL_LIST:
                        proposals, _ = self.proposal_generator(images, features, gt_instances=None)
                    
                    else:
                        raise('Select Proper Proposal Types')
                    
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [
                    x["proposals"].to(self.device) for x in batched_inputs
                ]

            results, _ = self.roi_heads(images, features, proposals, None)
            
        else:
            detected_instances = [
                x.to(self.device) for x in detected_instances
            ]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        return images


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )

        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances
        )
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
