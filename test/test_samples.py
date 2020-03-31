import os
import sys
import argparse

import torch
sys.path.append('..')
from Fcos_seg.data.coco_dataset import COCODataset
from Fcos_seg.data.transform import get_transform
from Fcos_seg.data.data_meta import COCO_CLASS_NAME
from Fcos_seg.core.config import get_cfg_defaults 
from Fcos_seg.utils.sampler_helper import get_sampler, make_batch_data_sampler
from Fcos_seg.utils.dataset_helper import collate_fn
from Fcos_seg.utils.norm_helper import UnNormalize
from Fcos_seg.loss.fcos_loss import FcosLoss
from Fcos_seg.detector.fcos import FCOS

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test COCO dataset"
    )
    
    parser.add_argument('data', metavar='DIR',
                    help='path to coco folder')
    
    


    return parser, parser.parse_args()


    
def main():
    
    _, args = parse_args()
    
    cfg = get_cfg_defaults()
    
    
    
    COCO_train_img_PATH = os.path.join(args.data, "images", "train2017")
    COCO_train_xml_PATH = os.path.join(args.data, "annotations", "instances_train2017.json")
    
    COCO_val_img_PATH = os.path.join(args.data, "images", "val2017")
    COCO_val_xml_PATH = os.path.join(args.data, "annotations", "instances_val2017.json")    
    train_dataset = COCODataset(cfg, COCO_train_xml_PATH, COCO_train_img_PATH, True, transform= get_transform(cfg, train = True))
    # val_dataset = COCODataset(cfg, COCO_val_xml_PATH, COCO_val_img_PATH, True, transform= get_transform(cfg, train = False))
    
    train_sampler = get_sampler(train_dataset, shuffle = False, distributed = False)
    # val_sampler = get_sampler(val_dataset, shuffle = False, distributed = False)   
    
    
    batch_per_gpu = cfg.TRAIN.BATCH
    
    
    train_sampler = make_batch_data_sampler(train_dataset, train_sampler, aspect_grouping = [1], images_per_batch = batch_per_gpu)
    # val_sampler = make_batch_data_sampler(val_dataset, val_sampler, aspect_grouping = [1], images_per_batch = batch_per_gpu)        


    train_output_path = "test_coco_dataset/train_samples"
    # val_output_path = "test_coco_dataset/val"
    print("==> testing output folder for train: {}".format(train_output_path))
    # print("==> testing output folder for val: {}".format(val_output_path))
    
    
    if not os.path.exists(train_output_path):
        os.makedirs(train_output_path)
    # if not os.path.exists(val_output_path):
    #     os.makedirs(val_output_path)
    
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset,                                                
                                            batch_sampler = train_sampler,
                                            num_workers = cfg.TRAIN.WORKER,
                                            collate_fn = collate_fn(cfg))
    
    # val_loader = torch.utils.data.DataLoader(val_dataset,                                                
    #                                         batch_sampler = val_sampler,
    #                                         num_workers = cfg.TRAIN.WORKER,
    #                                         collate_fn = collate_fn(cfg))        
        
    
    
    # only test 100 image
    iter_num = 10
    cur_iter_num = 0    
    
    un_norm = UnNormalize(mean = (103.53, 116.28, 123.675), std = (57.375, 57.12, 58.395))
    
    
    model = FCOS(cfg)
    test_loss_sample = FcosLoss(cfg)
    
    # testing train dataset
    for imgs, targets, idx in train_loader:
        
        if cur_iter_num >= iter_num:
            break
        
        locations = model.get_locations(imgs, targets)    
        cls_sample, reg_sample, locations = test_loss_sample.test_target_samples(locations, targets, imgs)
        
        for l in range(len(locations)):
            cls_sample_per_level = cls_sample[l]
            reg_sample_per_level = reg_sample[l]
            
            img_per_batch = imgs.tensors[0].clone()
            img_per_batch = img_per_batch * 0
            PIL_image = transforms.ToPILImage()(img_per_batch)
            drawObj = ImageDraw.Draw(PIL_image)            
            for loc in locations[l]:
                drawObj.rectangle((loc[0]-2, loc[1]-2,loc[0]+2, loc[1]+2), outline="red")
            
            PIL_image.save(os.path.join(train_output_path, "samplelocation_level{}.jpg".format(l)))            
            
            
            for b in range(len(cls_sample)):
                img_per_batch = imgs.tensors[b].clone()
                img_per_batch =  un_norm(img_per_batch)
                im_w, im_h = imgs.image_sizes[b]            
                img_per_batch = img_per_batch[:, :im_w, :im_h]                
                
                # batch_cls_per_level = cls_sample_per_level[b]
                # batch_reg_per_level = reg_sample_per_level[b]
                
                
                f_size = locations[l].shape[0]                
                
                batch_cls_per_level = cls_sample_per_level[b*f_size:(b+1)*f_size]
                batch_reg_per_level = reg_sample_per_level[b*f_size:(b+1)*f_size,:]
                print(batch_cls_per_level.size())
                print(batch_reg_per_level.size())
                
                pos_inds = torch.nonzero(batch_cls_per_level > 0).squeeze(1)
                
                
                PIL_image = transforms.ToPILImage()(img_per_batch)
                drawObj = ImageDraw.Draw(PIL_image)
                
                
                for p in pos_inds:                
                    x1 = (locations[l][p][0] - batch_reg_per_level[p, 0]).cpu()
                    y1 = (locations[l][p][1] - batch_reg_per_level[p, 2]).cpu()
                    x2 = (locations[l][p][0] + batch_reg_per_level[p, 1]).cpu()
                    y2 = (locations[l][p][1] + batch_reg_per_level[p, 3]).cpu()
                    
                    drawObj.rectangle((x1, y1, x2, y2), outline="red")
                    drawObj.text((x1, y1), "Class: {}".format(COCO_CLASS_NAME[int(batch_cls_per_level[p])]), fill="red")

                
                PIL_image.save(os.path.join(train_output_path, "sample_{}_level{}.jpg".format(idx[b], l)))
            
            
        
        exit()
        
    
    
            
            
            
            
            
        
    





if __name__ == "__main__":
    main()
