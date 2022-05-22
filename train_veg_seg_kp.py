
import torchvision.transforms as T
import torch
#from utils.data_loader import  VegDataset
from utils.data_loader_seg_keypoints import  VegDetection,Vegkeypoint
from utils.engine import train_one_epoch_seg_kp, evaluate,MultiTask_optimizers
import os
from tqdm import tqdm
from model.vegnet_v1 import *
import utils.detection_utils as detection_utils
import yaml
import argparse
import sys

torch.backends.cudnn.benchmark=True
argp=argparse.ArgumentParser()
argp.add_argument("--use-cudnn",type=bool,default=True)
argp.add_argument("--save-interval",type=int,default=20,help="Epochs after whcih to save a checkpoint")
argp.add_argument("--save-path",type=str,default="/media/asad/8800F79D00F79104/multitask_weights/multitask_weights",help="Epochs after which to save a checkpoint")
argp.add_argument("--lr",type=float,default=0.001,help="Learning rate")
argp.add_argument("--epochs",type=float,default=100,help="Maximum Epochs to train")
argp.add_argument("--config",type=str,default="model_config_seg_kp.yaml",help="Model Configuration")
argp.add_argument("--data-dir",type=str,default="data/cucumber",help="Data Directory")
argp.add_argument("--pretrain",type=str,default=None,help="Pretrain weights")
args=argp.parse_args()

#Get configurations      
config_file=args.config
with open(config_file, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def get_transform():
    transform = T.Compose([T.ToTensor()])
    return transform


def build_model(config):
    model = vegnet_resnet50_fpn(pretrained=False,trainable_backbone_layers=5,num_classes=config["numclasses"],class_map=config["classes"],num_keypoints=config["keypoints"]["num"],kp_name=config["keypoints"]["name"],
                                segm_names=config["segmentation"]["name"],segm_labels=config["segmentation"]["name"],segm_classes=config["segmentation"]["classes"],
                                class_names=config["classification"]["name"],class_labels=config["classification"]["name"],class_numclass=config["classification"]["numclasses"],
                                class_classes=config["classification"]["classes"],reg_names=config["regression"]["name"],reg_labels=config["regression"]["name"])
    return model
 
 
if __name__=="__main__":
    # Simple training loop 
    data_dir=args.data_dir
    model=build_model(config)    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device=torch.device("cpu")
    model.to(device)
    pretrain= args.pretrain
    if pretrain:
        print(f"Loading pretrained !!!!")
        state_dict=torch.load(pretrain)
        model.load_state_dict(state_dict=state_dict,strict=False)
    
    det_train="/media/asad/8800F79D00F79104/hubdata/cucumber/detection"
    kp_train="/media/asad/8800F79D00F79104/hubdata/cucumber/keypoints"
    

    det_train_data=VegDetection(det_train,classes=["cucumber"],vis=False,transform=get_transform())
    kp_train_data=Vegkeypoint(kp_train,vis=False,transform=get_transform())

    det_data_loader = torch.utils.data.DataLoader(det_train_data, batch_size=2, shuffle=True,collate_fn=detection_utils.collate_fn)
    kp_data_loader = torch.utils.data.DataLoader(kp_train_data, batch_size=2, shuffle=True,collate_fn=detection_utils.collate_fn)
    
    
    data_loaders={"detection":det_data_loader,"keypoints":kp_data_loader}
    
    num_epochs = args.epochs
    prevlog={}
    optim=MultiTask_optimizers()
    for epoch in tqdm(range(num_epochs+2)):
        # train for one epoch, printing every 10 iterations
        # Pass prev log None to choose task at random after warm up otherwise choose task based on the loss
        # Warm up is required so the detection task is working good first
        logs,targetloss=train_one_epoch_seg_kp(model, device, epoch, print_freq=5,optim=optim,dataloaders=data_loaders,prevlog=prevlog,warmup=5000)
        prevlog.update(targetloss)
        # update the learning rate
        if epoch%args.save_interval==0:
            PATH=args.save_path
            os.makedirs(PATH,exist_ok=True)
            torch.save(model.state_dict(), os.path.join(PATH,f"{epoch}_epoch_weights"))
            