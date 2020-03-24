import torch
import torchvision

from Fcos_seg.data.data_meta import CITYSCPAE_CLASS_NAME
from Fcos_seg.utils.box_list import BoxList












class CityScapeDataset(torchvision.datasets.cityscapes.Cityscapes):
    
    def __init__(self, cfg, root, split = "train", mode = "fine", target_type =["polygon", "semantic"], trans = None):
        
        super(CityScapeDataset, self).__init__(root, split, mode, target_type)
        
        
        self.instance_class = [
            "person", "rider", #human
            "car", "truck", "bus", "train", "motorcycle", "bicycle" # vehicle
        ]
        
        self.all_class_id_to_instance_id = {CITYSCPAE_CLASS_NAME.index(instance_name): i for i, instance_name in enumerate(self.instance_class)}
        
        self.instance_id_to_all_class_id = {v: k for k, v in self.all_class_id_to_instance_id.items()}
        
        self.trans = trans
        
        
    def __getitem__(self, index):
        
        img, (poly, semantic) = super().__getitem__(index)
        
        
        objs = [obj for obj in poly["objects"] if obj['label'] in self.instance_class]
        # set as instance id
        classes = [self.instance_class.index(obj["label"]) for obj in objs]
        boxes = []
        for obj in objs:
            
            
            poly_x, poly_y = zip(*obj["polygon"])
            x1 = min(poly_x)
            y1 = min(poly_y)
            x2 = max(poly_x)
            y2 = max(poly_y)
            
            boxes.append([x1, y1, x2, y2])
        
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        classes = torch.tensor(classes)
        target = BoxList(boxes, img.size, mode='xyxy')
        target.fields["labels"] = classes
        target.clip(remove_empty=True)
        
        
        if self.trans is not None:
            img, target = self.trans(img, target)
        
        
        
        
        return img, target, index