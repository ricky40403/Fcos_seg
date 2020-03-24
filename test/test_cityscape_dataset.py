import os
import sys
import argparse
sys.path.append('..')

import torch

from Fcos_seg.core.config import get_cfg_defaults 
from Fcos_seg.data.cityscape_dataset import CityScapeDataset
from Fcos_seg.data.data_meta import CITYSCPAE_CLASS_NAME
from Fcos_seg.data.transform import get_transform
from Fcos_seg.utils.dataset_helper import collate_fn
from Fcos_seg.utils.norm_helper import UnNormalize

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Cityscape dataset"
    )
    
    parser.add_argument('data', metavar='DIR',
                    help='path to Cityscape folder')
    
    


    return parser, parser.parse_args()



def main():
    
    _, args = parse_args()
    
    cfg = get_cfg_defaults()
    cfg.INPUT.MIN_SIZE_TRAIN = (512,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MIN_SIZE_TEST = 1024
    
    train_dataset = CityScapeDataset(cfg, args.data, split = "train", mode = "fine", trans= get_transform(cfg, train = True))
    val_dataset = CityScapeDataset(cfg, args.data, split = "val", mode = "fine", trans= get_transform(cfg, train = True))
    
    
    batch_per_gpu = cfg.TRAIN.BATCH
    
    train_output_path = "test_cityscape_dataset/train"
    val_output_path = "test_cityscape_dataset/val"
    print("==> testing output folder for train: {}".format(train_output_path))
    print("==> testing output folder for val: {}".format(val_output_path))
    
    
    if not os.path.exists(train_output_path):
        os.makedirs(train_output_path)
    if not os.path.exists(val_output_path):
        os.makedirs(val_output_path)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,                                                
                                            batch_size = batch_per_gpu,
                                            num_workers = cfg.TRAIN.WORKER,
                                            collate_fn = collate_fn(cfg)
                                            )
    
    val_loader = torch.utils.data.DataLoader(val_dataset,                                                
                                            batch_size = batch_per_gpu,
                                            num_workers = cfg.TEST.WORKER,
                                            collate_fn = collate_fn(cfg)
                                            )

    un_norm = UnNormalize(mean = (103.53, 116.28, 123.675), std = (57.375, 57.12, 58.395))
    # only test 100 image
    iter_num = 10
    cur_iter_num = 0   
    for imgs, targets, idx in train_loader:
        
        if cur_iter_num >= iter_num:
            break
        
        for b in range(len(imgs.tensors)):
            img_per_batch = imgs.tensors[b]
            
            targets_per_batch = targets[b]
            idx_per_batch = idx[b]            
            
            
            img_per_batch =  un_norm(img_per_batch)
            
            im_w, im_h = imgs.sizes[b]            
            # img_per_batch = img_per_batch[:, :im_w, :im_h]
            
            PIL_image = transforms.ToPILImage()(img_per_batch)
            drawObj = ImageDraw.Draw(PIL_image)
            
            boxes = targets_per_batch.box
            classes = targets_per_batch.fields["labels"]
            
            for i, box in enumerate(boxes):
                drawObj.rectangle((box[0], box[1], box[2], box[3]), outline="green")
                drawObj.text((box[0], box[1]), "Class: {}".format(train_dataset.instance_class[int(classes[i])]), fill="green")
                
            
            PIL_image.save(os.path.join(train_output_path, "train_{}.jpg".format(idx_per_batch)))
        
        cur_iter_num += 1

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
                drawObj.rectangle((box[0], box[1], box[2], box[3]), outline="green")
                drawObj.text((box[0], box[1]), "Class: {}".format(train_dataset.instance_class[int(classes[i])]), fill="green")
                
            
            PIL_image.save(os.path.join(val_output_path, "val_{}.jpg".format(idx_per_batch)))
        
        cur_iter_num += 1


if __name__ == "__main__":
    main()
