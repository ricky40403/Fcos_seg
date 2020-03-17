import os
import sys
import argparse

import torch
sys.path.append('..')
from Fcos.data.coco_dataset import COCODataset
from Fcos.data.transform import get_transform
from Fcos.data.data_meta import COCO_CLASS_NAME
from Fcos.core.config import get_cfg_defaults 
from Fcos.utils.sampler_helper import get_sampler, make_batch_data_sampler
from Fcos.utils.dataset_helper import collate_fn

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test COCO dataset"
    )
    
    parser.add_argument('data', metavar='DIR',
                    help='path to coco folder')
    
    parser.add_argument('-g', '--group_batch', action="store_true",
                    help='using group batch')


    return parser, parser.parse_args()


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    
def main():
    
    _, args = parse_args()
    
    cfg = get_cfg_defaults()
    
    
    
    COCO_train_img_PATH = os.path.join(args.data, "images", "train2017")
    COCO_train_xml_PATH = os.path.join(args.data, "annotations", "instances_train2017.json")
    
    COCO_val_img_PATH = os.path.join(args.data, "images", "val2017")
    COCO_val_xml_PATH = os.path.join(args.data, "annotations", "instances_val2017.json")    
    train_dataset = COCODataset(cfg, COCO_train_xml_PATH, COCO_train_img_PATH, True, transform= get_transform(cfg, train = True))
    val_dataset = COCODataset(cfg, COCO_val_xml_PATH, COCO_val_img_PATH, True, transform= get_transform(cfg, train = False))
    
    train_sampler = get_sampler(train_dataset, shuffle = False, distributed = False)
    val_sampler = get_sampler(val_dataset, shuffle = False, distributed = False)
    
    
    
    batch_per_gpu = cfg.TRAIN.BATCH
    
    if args.group_batch:
        print("Group batch")
        train_sampler = make_batch_data_sampler(train_dataset, train_sampler, aspect_grouping = [1], images_per_batch = batch_per_gpu)
        val_sampler = make_batch_data_sampler(val_dataset, val_sampler, aspect_grouping = [1], images_per_batch = batch_per_gpu)        


    train_output_path = "test_coco_dataset/train"
    val_output_path = "test_coco_dataset/val"
    print("==> testing output folder for train: {}".format(train_output_path))
    print("==> testing output folder for val: {}".format(val_output_path))
    
    
    if not os.path.exists(train_output_path):
        os.makedirs(train_output_path)
    if not os.path.exists(val_output_path):
        os.makedirs(val_output_path)
    
    
    if args.group_batch:
        train_loader = torch.utils.data.DataLoader(train_dataset,                                                
                                                batch_sampler = train_sampler,
                                                num_workers = cfg.TRAIN.WORKER,
                                                collate_fn = collate_fn(cfg))
        
        val_loader = torch.utils.data.DataLoader(val_dataset,                                                
                                                batch_sampler = val_sampler,
                                                num_workers = cfg.TRAIN.WORKER,
                                                collate_fn = collate_fn(cfg))
        
        
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size = batch_per_gpu,                                          
                                                batch_sampler = train_sampler,
                                                num_workers = cfg.TRAIN.WORKER,
                                                collate_fn = collate_fn(cfg))
        
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size = batch_per_gpu,                                          
                                                batch_sampler = val_sampler,
                                                num_workers = cfg.TRAIN.WORKER,
                                                collate_fn = collate_fn(cfg))
        
    
    # only test 100 image
    iter_num = 10
    cur_iter_num = 0    
    
    un_norm = UnNormalize(mean = (103.53, 116.28, 123.675), std = (57.375, 57.12, 58.395))
    
    # testing train dataset
    for imgs, targets, idx in train_loader:
        if cur_iter_num >= iter_num:
            break
        
        for b in range(len(imgs.tensors)):
            img_per_batch = imgs.tensors[b]
            targets_per_batch = targets[b]
            idx_per_batch = idx[b]
            
            
            
            img_per_batch =  un_norm(img_per_batch)
            
            im_w, im_h = imgs.sizes[b]            
            img_per_batch = img_per_batch[:, :im_w, :im_h]
            
            PIL_image = transforms.ToPILImage()(img_per_batch)
            drawObj = ImageDraw.Draw(PIL_image)
            
            boxes = targets_per_batch.box
            classes = targets_per_batch.fields["labels"]
            for i, box in enumerate(boxes):
                drawObj.rectangle((box[0], box[1], box[2], box[3]), outline="red")
                drawObj.text((box[0], box[1]), "Class: {}".format(COCO_CLASS_NAME[int(classes[i])]), fill="red")
                
            
            PIL_image.save(os.path.join(train_output_path, "train_{}.jpg".format(idx_per_batch)))
        
        cur_iter_num += 1
    
    
    # testing val dataset
    cur_iter_num = 0   
    for imgs, targets, idx in val_loader:
        if cur_iter_num >= iter_num:
            break
        
        for b in range(len(imgs.tensors)):
            img_per_batch = imgs.tensors[b]
            targets_per_batch = targets[b]
            idx_per_batch = idx[b]            
                        
            img_per_batch =  un_norm(img_per_batch)
            
            im_w, im_h = imgs.sizes[b]            
            img_per_batch = img_per_batch[:, :im_w, :im_h]
            
            PIL_image = transforms.ToPILImage()(img_per_batch)
            drawObj = ImageDraw.Draw(PIL_image)
            
            boxes = targets_per_batch.box
            classes = targets_per_batch.fields["labels"]
            for i, box in enumerate(boxes):
                drawObj.rectangle((box[0], box[1], box[2], box[3]), outline="red")
                drawObj.text((box[0], box[1]), "Class: {}".format(COCO_CLASS_NAME[int(classes[i])]), fill="red")
                
            
            PIL_image.save(os.path.join(val_output_path, "val_{}.jpg".format(idx_per_batch)))
                
                
        cur_iter_num += 1
            
            
            
            
            
            
        
    





if __name__ == "__main__":
    main()
