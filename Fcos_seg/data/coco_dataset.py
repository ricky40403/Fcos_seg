


import torch
import torchvision

from Fcos_seg.utils.box_list import BoxList



def has_only_empty_bbox(annot):
    return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)


def has_valid_annotation(annot):
    if len(annot) == 0:
        return False

    if has_only_empty_bbox(annot):
        return False

    return True



class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
            self, cfg, ann_file, root, remove_images_without_annotations=True, transform=None):
        
        super(COCODataset, self).__init__(root, ann_file)
        
        self.ids = sorted(self.ids)
        
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids
            
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        self.norm_mean = cfg.INPUT.PIXEL_MEAN
        self.norm_std = cfg.INPUT.PIXEL_STD
        self.transform = transform
        
    
    def __getitem__(self, index):
        
        img, anno = super().__getitem__(index)
        
        # filter crowd annotations
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        
        boxes = [o['bbox'] for o in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')
        
        classes = [o['category_id'] for o in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        target.clip_to_image(remove_empty=True)

        if self.transform is not None:
            img, target = self.transform(img, target)

        
        return img, target, index
    
    def get_image_meta(self, index):
        id = self.id_to_img_map[index]
        img_data = self.coco.imgs[id]
        
        return img_data
        


    