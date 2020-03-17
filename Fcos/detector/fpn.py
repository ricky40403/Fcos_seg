"""fpn.py
The feature pyramid network https://arxiv.org/abs/1612.03144
"""

import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_features, extra_feature_num, out_c = 256):
        super(FPN, self).__init__()         
        self.fpn_1x1blocks = nn.ModuleList([])
        # self.fpn_out_features = []
        self.in_feaures_num = len(in_features)
        
        # 1x1 conv
        for feature in in_features:
            _, in_c, _, _ = feature.size()
            self.fpn_1x1blocks.append(nn.Conv2d(in_c, out_c, kernel_size = 1))

        # 3x3 conv
        self.fpn_3x3blocks = nn.ModuleList([])
        self.total_feature_num = self.in_feaures_num + extra_feature_num
        for index in range(self.total_feature_num):
            if index >= self.in_feaures_num:
                self.fpn_3x3blocks.append(nn.Conv2d(out_c, out_c, kernel_size = 3, stride = 2, padding=1))
            else:
                self.fpn_3x3blocks.append(nn.Conv2d(out_c, out_c, kernel_size = 3, padding=1))

                   
        
    def up_and_add(self, deep, fine):
        
        return F.interpolate(deep, size=(fine.size()[2], fine.size()[3]),
                    mode='bilinear', align_corners=True) + fine
        
    
    def forward(self, features):
        
        # features should be C3, C4, C5 as example
        
        # 1x1 part
        FPN_feature = []
        for idx, feature in enumerate(features):
            FPN_feature.append(self.fpn_1x1blocks[idx](feature))        

        # up and add from deeper feature
        total_fpn_stage = len(FPN_feature)
        for idx, fpn_feature in reversed(list(enumerate(FPN_feature))):
            # the deepest feature do nothing
            if idx == (total_fpn_stage-1):
                deep_feature = fpn_feature
            else:
                # fine_feature = FPN_feature[idx]
                deep_feature = self.up_and_add(deep_feature, fpn_feature)
                FPN_feature[idx] = deep_feature
    
        out_features = []
        # final 3x3 conv for fpn
        for idx in range(self.total_feature_num):
            # extra conv stage
            if idx >= self.in_feaures_num:
                feature = self.fpn_3x3blocks[idx](feature)
            # origin input feature stage
            else:
                feature = self.fpn_3x3blocks[idx](FPN_feature[idx])
            
            out_features.append(feature)
        
        return out_features

            
            
            
        
        
        