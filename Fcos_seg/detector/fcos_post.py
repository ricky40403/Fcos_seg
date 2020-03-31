import torch
from torchvision import ops


from Fcos_seg.utils.box_list import BoxList
from Fcos_seg.utils.boxlist_ops import cat_boxlist
from Fcos_seg.utils.boxlist_ops import boxlist_ml_nms
from Fcos_seg.utils.boxlist_ops import remove_small_boxes


class FcosPost(torch.nn.Module):

    def __init__(self, cfg):

        super(FcosPost, self).__init__()        
        self.pre_nms_thresh = cfg.TEST.PRE_THRES        
        self.nms_thresh = cfg.TEST.NMS_THRES
        self.pre_nms_top_n = cfg.TEST.TOP_N        
        self.post_nms_top_n = cfg.TEST.POST_TOP_N        
        
        self.num_classes = cfg.MODEL.NUM_CLASS
        
        self.min_size = 0

    def forward_per_level(self, locations, box_cls , box_reg, box_center, image_sizes):
        
        N, C, H, W = box_cls.shape
                
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_reg = box_reg.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_reg = box_reg.reshape(N, -1, 4)
        box_center = box_center.view(N, 1, H, W).permute(0, 2, 3, 1)
        box_center = box_center.reshape(N, -1).sigmoid()
        
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)
        
        box_cls = box_cls * box_center[:, :, None]
        
        pred_results_per_level = []
        # loop batch images
        for i in range(N):
            
            # filter candidates
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]
            
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1
            
            per_box_reg = box_reg[i]
            per_box_reg = per_box_reg[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]
            
            
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_reg = per_box_reg[top_k_indices]
                per_locations = per_locations[top_k_indices]
                
            detections = torch.stack([
                per_locations[:, 0] - per_box_reg[:, 0],
                per_locations[:, 1] - per_box_reg[:, 1],
                per_locations[:, 0] + per_box_reg[:, 2],
                per_locations[:, 1] + per_box_reg[:, 3],
            ], dim=1)
            
            
            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", torch.sqrt(per_box_cls))
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            pred_results_per_level.append(boxlist)
            
        return pred_results_per_level
        

    def forward(self, locations, box_cls, box_regression, centerness, image_sizes):

        
        preds = []
        for loc, cls_p, box_p, c_p in zip(locations, box_cls, box_regression, centerness):
            preds.append(self.forward_per_level(loc, cls_p, box_p, c_p, image_sizes))
            
        # gather levels_batches to batches_levels
        boxlists = list(zip(*preds))
        # cat the boxes in levels per image
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        
        # nms of all levels per image
        for i, box_per_batch in enumerate(boxlists): 
            # box should be in mode xyxy
            box_mode = box_per_batch.mode
            box_size = box_per_batch.size
            
            boxes = box_per_batch.bbox
            scores = box_per_batch.get_field("scores")
            labels = box_per_batch.get_field("labels")
            results_perbatch = []
            # nms for each class (or nms of all classes? (to be checked))
            for c in range(1, self.num_classes):
                target_idx = (labels == c).view(-1)
                score_per_class = scores[target_idx]
                boxes_per_class = boxes[target_idx, :].view(-1, 4)
                
                keep =  ops.nms(boxes_per_class, score_per_class, self.nms_thresh)
                
                # the remaining
                boxes_per_class = boxes_per_class[keep]
                score_per_class = score_per_class[keep]
                n_keep = len(boxes_per_class)
                # cast to same device
                cid_per_class = torch.full(
                    (n_keep, 1), c, dtype = torch.float32, device=score_per_class.device
                )
                
                # if has predictions
                if n_keep > 0:
                    results_perbatch.append(torch.cat([cid_per_class, score_per_class.unsqueeze(-1), boxes_per_class], dim = 1))

            # detections remain after nms
            n_detection = len(results_perbatch)
            # cat detections to tensor
            if n_detection > 0:
                results_perbatch = torch.cat(results_perbatch, dim = 0)

            
            # if still more than post top n
            if n_detection > self.post_nms_top_n > 0:
                scores = results_perbatch[:, 1]
                img_threshold, _ = torch.kthvalue(
                    scores.cpu(), n_detection - self.post_top_n + 1
                )
                keep = scores >= img_threshold.item()
                keep = torch.nonzero(keep).squeeze(1)
                results_perbatch = results_perbatch[keep]
                
            
            # back to boxlist
            tmp_box = BoxList(results_perbatch[:, 2:], box_size, mode = box_mode)
            tmp_box.add_field("labels", results_perbatch[:, 0])
            tmp_box.add_field("scores", results_perbatch[:, 1])
            tmp_box = tmp_box.clip_to_image(remove_empty=False)
            tmp_box = remove_small_boxes(tmp_box, self.min_size)
            boxlists[i] = tmp_box
            
        
        
        return boxlists
