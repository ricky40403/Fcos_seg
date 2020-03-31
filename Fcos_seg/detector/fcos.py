import math
import torch
import torch.nn.functional as F
from torch import nn

import Fcos_seg.backbone as backbone
import Fcos_seg.detector as detector
from Fcos_seg.detector.fcos_post import FcosPost
from Fcos_seg.loss.fcos_loss import FcosLoss
from Fcos_seg.detector.scale import Scale

from Fcos_seg.utils.dist_helper import get_rank

class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels, total_stage):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        
        num_classes = cfg.MODEL.NUM_CLASS
        # this may auto calculate (to do)
        self.fpn_strides = [8, 16, 32, 64, 128]
        self.norm_reg_targets = cfg.MODEL.NORM_REG
        self.centerness_on_reg = cfg.MODEL.CENTERNESS_ON_REG        

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.HAED_RANGE):            

            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])
        
        

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
                
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred))
                
        return logits, bbox_reg, centerness


class FCOS(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels=256):
        super(FCOS, self).__init__()
 
        
        # BACKBONE
        self.backbone = backbone.__dict__[cfg.MODEL.BACKBONE](pretrained = False)
        
        # FPN
        # auto computing output stage        
        # find out fpn and out information
        rand_tensor = torch.rand(1, 3, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
        features = self.backbone(rand_tensor)                
        # prepare for fpn, random generate an input image        
        self.fpn = detector.fpn.FPN(features, cfg.MODEL.FPN.EXTRA_NUM, cfg.MODEL.FPN.OUT_C)
        # total number of level in heads
        self.level_nums = len(features) + cfg.MODEL.FPN.EXTRA_NUM        
        del rand_tensor, features
        
        self.head = FCOSHead(cfg, in_channels, self.level_nums)


        self.loss = FcosLoss(cfg)
        self.post = FcosPost(cfg)        
        
        self.fpn_strides = [8, 16, 32, 64, 128]
        
    
    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations
    
    
    def get_locations(self, images, targets):
        in_h, in_w = images.tensors[0].size()[-2:]
        backbone_features = self.backbone(images.tensors)        
        features = self.fpn(backbone_features)
        for l, f in enumerate(features):
            print("Level {} : {}".format(l, f.size()))
        locations = self.compute_locations(features)
        
        return locations
        
        
    
    def forward(self, images, targets=None):
        
        in_h, in_w = images.tensors[0].size()[-2:]
        backbone_features = self.backbone(images.tensors)        
        features = self.fpn(backbone_features)        
        
        
        box_cls, box_regression, centerness = self.head(features)
        
        locations = self.compute_locations(features)
        
        
        if self.training:
            loss_box_cls, loss_box_reg, loss_centerness = self.loss(locations, box_cls, box_regression, centerness, targets, images)
            
            losses = {
                "loss_cls": loss_box_cls,
                "loss_reg": loss_box_reg,
                "loss_centerness": loss_centerness
            }
            return None, losses
        
        else:
            boxes = self.post(locations, box_cls, box_regression, centerness, images.image_sizes)
            return boxes, None
        
        
        
        