import math
import sys
import time
from sqlalchemy import true
import torch
import torchvision.models.detection.mask_rcnn
import random
from . import detection_utils as utils
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset


def get_multitasks(epoch,warmup,losses=None):
    """Task To Train on

    Args:
        epoch (_type_): _description_
        warmup (_type_): _description_
        losses (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    ## Do this Better by getting names from targets
    all_tasks=["detection","backbone","keypoints","neck","rating"]
    ## If losses are present then sample task based on probability of
    loss_weights=[]
    valid_tasks={}
    
    if epoch<warmup:
        # importan to first have good detection
        #Set detection true other false if epoch is less than warmup-
        for idx in range(len(all_tasks)):
            if (idx==0):
                valid_tasks[all_tasks[idx]]=True
                print(f"Training for Task: {all_tasks[idx]}")
            else:
                valid_tasks[all_tasks[idx]]=False
                
    elif losses is not None:
        for task in all_tasks:
            if task in losses.keys():
                loss_weights.append(losses[task])
            else:
                # Append some very high loss to prioritize its turn
                loss_weights.append(float(12345))
            if loss_weights[-1]==0.0:
                # have some small probability to choose the task
                loss_weights[-1]+=1e-4
        # normalized weights
        nw=[i/sum(loss_weights) for i in loss_weights]
        i=random.choices(population=list(range(len(all_tasks))),weights=nw)[0]
        for idx in range(len(all_tasks)):
            if (idx==i):
                valid_tasks[all_tasks[idx]]=True
                print(f"Using Loss weights, Training for Task: {all_tasks[idx]}")
            else:
                valid_tasks[all_tasks[idx]]=False      
    else:
        # Choose randomly 
        i=random.choice(list(range(len(all_tasks))))
        for idx in range(len(all_tasks)):
            if (idx==i):
                valid_tasks[all_tasks[idx]]=True
                print(f"Training for Task: {all_tasks[idx]}")
            else:
                valid_tasks[all_tasks[idx]]=False    
         
                
    return valid_tasks



def freeze_multitask(model,valid_tasks):
    """Freeze Branches which are not used for this task

    Args:
        valid_tasks (Dict[strings:bool]): Dict of valid tasks  
    """
    if valid_tasks:
        for k,v in valid_tasks.items():
            if (k=="detection"):
                model.rpn.requires_grad_(requires_grad=v)
                model.roi_heads.box_roi_pool.requires_grad_(requires_grad=v)
                model.roi_heads.box_head.requires_grad_(requires_grad=v)
                model.roi_heads.box_predictor.requires_grad_(requires_grad=v)
                # If mask is present then also freeze the corresponding mask
                if "mask_head" in model.roi_heads.roi_head_names:
                    model.roi_heads.mask_roi_pool["masks"].requires_grad_(requires_grad=v)
                    model.roi_heads.mask_head["masks"].requires_grad_(requires_grad=v)
                    model.roi_heads.mask_predictor["masks"].requires_grad_(requires_grad=v)
            if (k=="keypoints"):
                if "keypoint_head" in model.roi_heads.roi_head_names:
                    model.roi_heads.keypoint_head.requires_grad_(requires_grad=v)
                    model.roi_heads.keypoint_predictor.requires_grad_(requires_grad=v)
                    model.roi_heads.keypoint_roi_pool.requires_grad_(requires_grad=v)
            if ("class_head" in model.roi_heads.roi_head_names):
                for k2 in model.roi_heads.class_head.keys():
                    if k2==k:
                        model.roi_heads.class_roi_pool[k].requires_grad_(requires_grad=v)  
                        model.roi_heads.class_head[k].requires_grad_(requires_grad=v)  
                        model.roi_heads.class_predictor[k].requires_grad_(requires_grad=v)                
            # Search for more mask heads in mask_head            
            if ("mask_head" in model.roi_heads.roi_head_names):
                for k2 in model.roi_heads.mask_head.keys():
                        if k2==k:
                            model.roi_heads.mask_roi_pool[k].requires_grad_(requires_grad=v)
                            model.roi_heads.mask_head[k].requires_grad_(requires_grad=v)
                            model.roi_heads.mask_predictor[k].requires_grad_(requires_grad=v)   


def adjust_losses(losses,valid_tasks):
    """Adjust Multi task loss yero out losses which should not contrinute in the current iteration

    Args:
        losses (Dict{key:val}): Loss for each task
        valid_tasks (Dict{key:false}): Dict with valid and invalid task for each task
    """
    if valid_tasks:
        for k,v in valid_tasks.items():
            if k=="detection":
                if not v:
                    losses["loss_classifier"]*=0.0
                    losses["loss_box_reg"]*=0.0
                    losses["loss_objectness"]*=0.0
                    losses["loss_rpn_box_reg"]*=0.0
                    if "masks" in losses:
                        losses["masks"]*=0.0
            if k=="keypoints":
                if not v:
                    losses["loss_keypoint"]*=0.0
            if k in losses.keys():
                if not v:
                    losses[k]=0.0


class MultiTask_optimizers:
    
    optimizers={}
    schedulers={}
    
    def get_optimizer(self,valid_tasks,model):
            optimizer = None
            scheduler= None
            for task,v in valid_tasks.items():
                if v==True:
                    if task in self.optimizers:
                        optimizer=self.optimizers[task]
                        scheduler=self.schedulers[task]
                        return optimizer,scheduler
                    else:
                        params = [p for p in model.parameters() if p.requires_grad]
                        self.optimizers[task]= torch.optim.AdamW(params, lr=1e-3,weight_decay=0)
                        optimizer=self.optimizers[task]
                        self.schedulers[task]= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.9,patience=10)
                        scheduler=self.schedulers[task]
                        return optimizer,scheduler
            
        
        
def train_one_epoch(model, data_loader, device, epoch, print_freq,optim,prevlog=None,warmup=40):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    # Sample Task
    valid_tasks=get_multitasks(epoch,warmup,prevlog)
    #
    # Equivalent to conditioning on one hot encoding of task
    freeze_multitask(model,valid_tasks=valid_tasks)
    
    target_task=None
    for task,v in valid_tasks.items():
        if v:
            target_task=task
    
    assert target_task!=None, f"Target task is {target_task} while valid tasks are {valid_tasks}"
    
    optimizer,scheduler=optim.get_optimizer(valid_tasks,model)

    #for k,v in model.roi_heads.mask_predictor.items():
    #    for name, params in v.named_parameters():
    #        print(k,name, params.requires_grad)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        images = list(image.to(device) for image in images)
        
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets,valid_tasks)
        # Multiplicative Task selection 
        adjust_losses(loss_dict,valid_tasks=valid_tasks)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        #if lr_scheduler is not None:
        #    lr_scheduler.step()
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    scheduler.step(metrics=metric_logger.meters.get("loss").value)
    return metric_logger,{target_task:loss_value}


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator