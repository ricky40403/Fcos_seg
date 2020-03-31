"""
This file contains specific functions for computing losses of FCOS
file
"""

import os

import torch
from torch import nn
import torch.distributed as dist
from torch.nn import functional as F

from Fcos_seg.loss.iou_loss import IOULoss
from Fcos_seg.loss.sigmoid_focal_loss import SigmoidFocalLoss

# from Fcos_seg.utils.boxlist_ops import boxlist_iou
# from Fcos_seg.utils.boxlist_ops import cat_boxlist


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


class FcosLoss(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.LOSS.FOCAL_GAMMA,
            cfg.MODEL.LOSS.FOCAL_ALPHA
        )
        self.fpn_strides = [8, 16, 32, 64, 128]        
        self.iou_loss_type = cfg.MODEL.LOSS.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.NORM_REG
        
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.inf = cfg.MODEL.INF

    def compute_target_per_level(self, location_per_level, targets, field_size_per_level):
        
        
        # xs, ys = location_per_level[:, 0], location_per_level[:, 1]
        
        paried_label = []
        paired_reg = []
        # loop batch
        for target_per_im in targets:
            assert target_per_im.mode == "xyxy"
            target_label = target_per_im.get_field("labels")
            target_boxes = target_per_im.bbox
            # print("box num: {}".format(len(target_boxes)))
            # target_area = target_per_im.area()
            
            
            l = location_per_level[:, None, 0] - target_boxes[:, 0][None]
            t = location_per_level[:, None, 1] - target_boxes[:, 1][None]
            r = target_boxes[:, 2][None] - location_per_level[:, None, 0]
            b = target_boxes[:, 3][None] - location_per_level[:, None, 1]
            
            
            ltrb_bbox = torch.stack([l, t, r, b], dim=2)
            
            # print(lrtb_bbox.size())
            
            # Step1 filter out piexl not in box
            # todo sampling method, cur use all
            is_in_boxes = ltrb_bbox.min(dim=2)[0] > 0
        
            # Step2: Filter out area not in receptive fields
            cared_in_level = ltrb_bbox.max(dim=2)[0]
            cared_in_level = (cared_in_level >= field_size_per_level[0]) & (cared_in_level <= field_size_per_level[1])
            
            
            # Step3: If there still have more than one target in the same pixel, choose the smallest one
            area = (r - l + 1) * (b - t + 1)          
            
            
            # set flag of is_in_boxes and cared_in_level
            area[is_in_boxes==0] = self.inf
            area[cared_in_level==0] = self.inf
            locations_to_min_area, locations_to_gt_inds = area.min(dim=1)
                        
            # preserve the targets
            reg_per_img = ltrb_bbox[range(len(location_per_level)), locations_to_gt_inds]            
            labels_per_im  = target_label[locations_to_gt_inds]
            labels_per_im [locations_to_min_area == self.inf] = 0            
            
            paried_label.append(labels_per_im)
            paired_reg.append(reg_per_img)
        
        return paried_label, paired_reg
        
        
    def compute_receptive_fields(self):
        # this may auto compute the receptive field size (to do)
        # current use pre-define size
        fields = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, self.inf],
        ]
        
        return fields


    def compute_center_from_reg(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)
    
    def test_target_samples(self, locations, targets, images):
        
        field_size = self.compute_receptive_fields()
        
        out_cls = []
        out_reg = []
        for l in range(len(locations)):
            target_cls_per_level, target_reg_per_level = self.compute_target_per_level(locations[l], targets, field_size[l])            
            target_cls_per_level = torch.cat(target_cls_per_level, dim = 0).reshape(-1)
            target_reg_per_level = torch.cat(target_reg_per_level, dim = 0).reshape(-1, 4)
            
            out_cls.append(target_cls_per_level)
            out_reg.append(target_reg_per_level)
        return out_cls, out_reg, locations
            

    def __call__(self, locations, box_cls, box_regression, centerness, targets, images):
        
        
        num_classes = box_cls[0].size(1)
        
        # convert list of level to torch tensors
        pred_cls = []
        pred_reg = []
        pred_center = []
        target_cls = []
        target_reg = []        
        
        field_size = self.compute_receptive_fields()
        
        # loop level
        for l in range(len(locations)):            
            pred_cls.append(box_cls[l].permute(0, 2, 3, 1).contiguous().reshape(-1, num_classes)) # N x h x w, C
            pred_reg.append(box_regression[l].permute(0, 2, 3, 1).contiguous().reshape(-1, 4)) # N x h x w, 4
            pred_center.append(centerness[l].permute(0, 2, 3, 1).contiguous().reshape(-1)) # N x h x w
            
            # tmp_field = torch.FloatTensor(field_size[l]).to(locations[l].device)
            # # return [N [h x w]], [N, [h x w, 4]]
            target_cls_per_level, target_reg_per_level = self.compute_target_per_level(locations[l], targets, field_size[l])
            
            target_cls_per_level = torch.cat(target_cls_per_level, dim = 0).reshape(-1)
            target_reg_per_level = torch.cat(target_reg_per_level, dim = 0).reshape(-1, 4)
            
            
            if self.norm_reg_targets:
                target_reg_per_level = target_reg_per_level / self.fpn_strides[l]
            
            
            # cat the batch and append to target list
            target_cls.append(target_cls_per_level) # N x h x w
            target_reg.append(target_reg_per_level) # N x h x w, 4
            
            
        # cat level by lebel
        pred_cls_flatten = torch.cat(pred_cls, dim = 0)
        pred_reg_flatten = torch.cat(pred_reg, dim = 0)
        pred_center_flatten = torch.cat(pred_center, dim = 0)        
        
        target_cls_flatten = torch.cat(target_cls, dim = 0)
        target_reg_flatten = torch.cat(target_reg, dim = 0)
        
        pos_inds = torch.nonzero(target_cls_flatten > 0).squeeze(1)
        pred_reg_flatten = pred_reg_flatten[pos_inds]
        target_reg_flatten = target_reg_flatten[pos_inds]
        pred_center_flatten = pred_center_flatten[pos_inds]
            
        # follow the normalize method of the origin respository
        num_gpus = get_num_gpus()
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)
        

        cls_loss = self.cls_loss_func(
            pred_cls_flatten,
            target_cls_flatten.int()
        ) / num_pos_avg_per_gpu
        
        
        
        if pos_inds.numel() > 0:
            target_center_flatten = self.compute_center_from_reg(target_reg_flatten)           
            
            # follow origin repository
            target_center_flatten_avg_per_gpu = \
                reduce_sum(target_center_flatten.sum()).item() / float(num_gpus)
                
            reg_loss = self.box_reg_loss_func(
                pred_reg_flatten,
                target_reg_flatten,
                target_center_flatten
            ) / target_center_flatten_avg_per_gpu
            
            centerness_loss = self.centerness_loss_func(
                pred_center_flatten,
                target_center_flatten
            ) / num_pos_avg_per_gpu
            
        else:
            reg_loss = pred_reg_flatten.sum()
            reduce_sum(pred_center_flatten.new_tensor([0.0]))
            centerness_loss = pred_center_flatten.sum()
            
        return cls_loss, reg_loss, centerness_loss