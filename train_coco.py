import os
import sys
import argparse
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
from torch.utils.tensorboard import SummaryWriter

from Fcos.core.config import get_cfg_defaults
from Fcos.core.coco_eval import coco_evaluate
from Fcos.data.coco_dataset import COCODataset
from Fcos.data.transform import get_transform
from Fcos.detector.fcos import FCOS
from Fcos.utils.dist_helper import get_rank, synchronize, reduce_loss_dict, all_gather
from Fcos.utils.sampler_helper import get_sampler, make_batch_data_sampler
from Fcos.utils.dataset_helper import collate_fn




def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Network"
    )
    
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training',
        default=None,
        type=str
    )

    parser.add_argument('data', metavar='DIR',
                    help='path to folder which contain the images and the annotations')   
    
    parser.add_argument('--dist_backend', default='nccl', type=str)
    
    parser.add_argument('--dist_url', default='env://', type=str)       
    
    
    parser.add_argument('--local_rank', type=int, default=0)
    
    parser.add_argument('--resume_epoch', type=int, default=0)
    
    parser.add_argument('-c', '--checkpoint_path', type=str, default='checkpoints', help='path to checkpoints')
    
    
    
    return parser, parser.parse_args()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def training(args, epoch, loader, model, optimizer, device, writer = None):
    
    model.train()
    if get_rank() == 0:    
        pbar = tqdm(loader, dynamic_ncols=True)
    else:
        pbar = loader
    
    total_data_len = len(pbar)
    iteration = 1
    
    
    # does not need idx
    for images, targets, _ in pbar:
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        _, loss_dict = model(images.tensors, targets=targets)
        
        # loss_cls = loss_dict['loss_cls'].mean()
        # loss_reg = loss_dict['loss_box'].mean()
        # loss_center = loss_dict['loss_center'].mean()
        
        # loss = loss_cls + loss_reg + loss_center
        
        losses = sum(loss for loss in loss_dict.values())     
        
        # for record
        loss_reduced = reduce_loss_dict(loss_dict)
        loss_cls = loss_reduced['loss_cls'].mean().item()
        loss_reg = loss_reduced['loss_box'].mean().item()
        loss_center = loss_dict['loss_center'].mean().item()
        
        optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        
        if get_rank() == 0:
            pbar.set_description(
                "Epoch: {}, lr: {}, cls_loss: {}, reg_loss: {}, center_loss:{}.".format(epoch+1, get_lr(optimizer), loss_cls, loss_reg, loss_center)
            )
            
            
            if (writer is not None) and (get_rank()==0):
                writer.add_scalar("Loss/cls_loss", loss_cls, epoch * total_data_len +  iteration)
                writer.add_scalar("Loss/reg_loss", loss_reg, epoch * total_data_len +  iteration)
                writer.add_scalar("Loss/center_loss", loss_center, epoch * total_data_len +  iteration)            
        iteration += 1



def accumulate_predictions(predictions):
    
    all_predictions = all_gather(predictions)

    if get_rank() != 0:
        return

    predictions = {}

    for p in all_predictions:
        predictions.update(p)

    ids = list(sorted(predictions.keys()))

    if len(ids) != ids[-1] + 1:
        print('Evaluation results is not contiguous')

    predictions = [predictions[i] for i in ids]

    return predictions


@torch.no_grad()
def validation(args, epoch, dataset, loader, model, device, writer = None):
    if args.distributed:
        model = model.module
    
    torch.cuda.empty_cache()
    model.eval()
    
    if get_rank() == 0:
        pbar = tqdm(loader, dynamic_ncols=True)
    else:
        pbar = loader
        
        
    pred_results = {}    
    for images, targets, data_idx in pbar:
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        pred, _ = model(images.tensors, img_size = images.sizes)
        
        pred = [p.to('cpu') for p in pred]
        pred_results.update({id: p for id, p in zip(data_idx, pred)})
        # break
    
    pred_results = accumulate_predictions(pred_results)
    
    if get_rank() != 0:
        return    
    
    coco_result = coco_evaluate(dataset, pred_results)
    stat = coco_result.stats
    # should had only master node now
    writer.add_scalar("Result_AP/AP", stat[0], epoch)
    writer.add_scalar("Result_AP/AP50", stat[1], epoch)
    writer.add_scalar("Result_AP/AP75", stat[2], epoch)
    writer.add_scalar("Result_AP/AP_small", stat[3], epoch)
    writer.add_scalar("Result_AP/AP_medium", stat[4], epoch)
    writer.add_scalar("Result_AP/AP_large", stat[5], epoch)
    writer.add_scalar("Result_AR/AR_Dt1", stat[6], epoch)
    writer.add_scalar("Result_AR/AR_Dt10", stat[7], epoch)
    writer.add_scalar("Result_AR/AR_Dt100", stat[8], epoch)
    writer.add_scalar("Result_AR/AR_small", stat[9], epoch)
    writer.add_scalar("Result_AR/AR_medium", stat[10], epoch)
    writer.add_scalar("Result_AR/AR_large", stat[11], epoch)

    

def main():
    _, args = parse_args()
    
    if args.cfg_file is None:
        sys.exit("Please Choose Config file")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)
    
    
    COCO_train_img_PATH = os.path.join(args.data, "images", "train2017")
    COCO_train_xml_PATH = os.path.join(args.data, "annotations", "instances_train2017.json")
    
    COCO_val_img_PATH = os.path.join(args.data, "images", "val2017")
    COCO_val_xml_PATH = os.path.join(args.data, "annotations", "instances_val2017.json")
    
    # handle distribute
    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1
    
    images_per_batch = cfg.TRAIN.BATCH
    batch_per_gpu = images_per_batch // n_gpu
    
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
        synchronize()
        if get_rank() == 0 : 
            print("==> Using Distributed traning ......") 
    
    
    
    
    # using group batch from detectron2
    train_dataset = COCODataset(cfg, COCO_train_xml_PATH, COCO_train_img_PATH, True, transform= get_transform(cfg, train = True))
    val_dataset = COCODataset(cfg, COCO_val_xml_PATH, COCO_val_img_PATH, True, transform= get_transform(cfg, train = False))
    
    train_sampler = get_sampler(train_dataset, shuffle = True, distributed = args.distributed)
    
    val_shuffle = False if not args.distributed else True
    val_sampler = get_sampler(val_dataset, shuffle = val_shuffle, distributed = args.distributed)
    
    train_sampler = make_batch_data_sampler(train_dataset, train_sampler, aspect_grouping = [1], images_per_batch = batch_per_gpu)
    val_sampler = make_batch_data_sampler(val_dataset, val_sampler, aspect_grouping = [1], images_per_batch = batch_per_gpu)       
    
    train_loader = torch.utils.data.DataLoader(train_dataset,                                                
                                                batch_sampler = train_sampler,
                                                num_workers = cfg.TRAIN.WORKER,
                                                collate_fn = collate_fn(cfg))
    
    val_loader = torch.utils.data.DataLoader(val_dataset,                                                
                                            batch_sampler = val_sampler,
                                            num_workers = cfg.TRAIN.WORKER,
                                            collate_fn = collate_fn(cfg))
    
    
    # get model and trans to cuda
    device = "cuda"
    model = FCOS(cfg)
    
    optimizer = torch.optim.SGD(model.parameters(), lr = cfg.TRAIN.LR, weight_decay=0.0001, momentum = cfg.TRAIN.MOMENTUM)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.MILESTONES, gamma=0.1)
    total_epoch = cfg.TRAIN.EPOCH  
    cur_epoch = 0
    
    # is resume
    if args.resume_epoch > 0:
        if get_rank() == 0:
            print("==> Resume from epoch {}".format(args.resume_epoch))
        checkpoint = torch.load("checkpoints/epoch-{}.pt".format(args.resume_epoch))
        cur_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optim'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['model'])
        del checkpoint

    
    model = model.to(device)
    

    # writer
    writer = None
    if get_rank() == 0:
        writer = SummaryWriter(log_dir = "logs")

    # checkpoints
    checkpoint_path = args.checkpoint_path
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    
    
    if args.distributed:  
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[args.local_rank],
                                                    output_device=args.local_rank,
                                                    broadcast_buffers=False
                                                    )
    
    
    for epoch in range(cur_epoch, total_epoch):
        training(args, epoch, train_loader, model, optimizer, device, writer)
        
        if get_rank() == 0:
            torch.save(
                {'epoch': epoch,
                 'model': model.module.state_dict(),
                 'optim': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict()},
                f'checkpoints/epoch-{epoch + 1}.pt',
            )
        
        
        validation(args, epoch, val_dataset, val_loader, model, device, writer)
        scheduler.step()
        # exit()
        
        
        
        
    
if __name__ == "__main__":
    main()
