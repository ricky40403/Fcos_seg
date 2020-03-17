"""fcos.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import Fcos.backbone as backbone
import Fcos.detector as detector
from Fcos.utils import *
from Fcos.loss.fcos_loss import FcosLoss



class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.NUM_CLASS - 1
        self.fpn_strides = [8, 16, 32, 64, 128]        
        self.centerness_on_reg = cfg.MODEL.CENTERNESS_ON_REG        

        cls_tower = []
        bbox_tower = []
        for _ in range(4):
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

        self.scales = nn.ModuleList([detector.scale.Scale(init_value=1.0) for _ in range(5)])

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
            bbox_pred = F.relu(bbox_pred)
            
            if self.training:
                bbox_reg.append(bbox_pred)
            else:
                bbox_reg.append(bbox_pred * self.fpn_strides[l])
            
        return logits, bbox_reg, centerness

class FCOS(nn.Module):
    def __init__(self, cfg, pretrained = False):
        super(FCOS, self).__init__()
        
        
        # BACKBONE
        self.backbone = backbone.__dict__[cfg.MODEL.BACKBONE](pretrained = pretrained)
        
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
        self.inf = cfg.MODEL.INF
        # maybe can auto compute receptive field.
        self.fields = [[-1, 64],
                     [64, 128],
                     [128, 256],
                     [256, 512],
                     [512, self.inf]]
        
        self.head = FCOSHead(cfg, cfg.MODEL.FPN.OUT_C)
        
        self.loss = FcosLoss(cfg)
        self.post_process = detector.fcos_post.FCOSPostprocessor(cfg.TEST.THRES,
                                                                 cfg.TEST.TOP_N, 
                                                                 cfg.TEST.NMS_THRES, 
                                                                 cfg.TEST.POST_TOP_N, 
                                                                 cfg.INPUT.MIN_SIZE_TEST, 
                                                                 cfg.MODEL.NUM_CLASS )
    
    
    def compute_locations(self, features, im_h, im_w):
        locations = []
        for _, feature in enumerate(features):
            fh, fw = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                im_h, im_w, fh, fw,
                feature.device
            )
            
            locations.append(locations_per_level)
            
        return locations

    # auto matically compute locations by the relationship between img and feature size
    # so it need not care the fpn stride
    # maybe can compute stride here for automation
    def compute_locations_per_level(self, img_h, img_w, f_h, f_w, device):
        y = torch.linspace(0, img_h-1, f_h).to(device)
        x = torch.linspace(0, img_w-1, f_w).to(device)
        center_y, center_x = torch.meshgrid(y, x)
        center_y = center_y.squeeze(0).contiguous()
        center_x = center_x.squeeze(0).contiguous()
        
        return torch.stack((center_x.view(-1), center_y.view(-1)), dim = 1) # [h * w, (cx, cy)]
        
    
    
    def forward(self, x, targets = None, img_size = None):

        
        im_h, im_w = x.size()[-2:]
        
        backbone_features = self.backbone(x)
        fpn_features = self.fpn(backbone_features)
                
        box_cls, box_regression, centerness = self.head(fpn_features)
        locations = self.compute_locations(fpn_features, im_h, im_w)
        
        
        if self.training:
            cls_loss, reg_loss, centerness_loss = self.loss(
                locations,
                box_cls, box_regression, centerness, targets
            )
            
            losses = {
                'loss_cls': cls_loss,
                'loss_box': reg_loss,
                'loss_center': centerness_loss,
            }
            
            return None, losses
        else:
            
            results = self.post_process(locations, box_cls, box_regression, centerness, img_size)
            
            return results, None
        
        
        




