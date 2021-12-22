# Example to train Vegnet 

import torchvision.transforms as T
import torch
from utils.data_loader import  VegDataset
from utils.engine import train_one_epoch, evaluate
import os
from tqdm import tqdm
from model.vegnet_v1 import *
import utils.detection_utils as detection_utils
import yaml
import argparse

torch.backends.cudnn.benchmark=True
argp=argparse.ArgumentParser()
argp.add_argument("--use-cudnn",type=bool,default=True)
argp.add_argument("--save-interval",type=str,default=100,help="Epochs after whcih to save a checkpoint")
argp.add_argument("--lr",type=float,default=0.001,help="Learning rate")
argp.add_argument("--epochs",type=float,default=1000,help="Maximum Epochs to train")
argp.add_argument("--config",type=str,default="model_config.yaml",help="Model Configuration")
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
    model = vegnet_resnet50_fpn(pretrained=True,num_classes=config["numclasses"],class_map=config["classes"],num_keypoints=config["keypoints"]["num"],kp_name=config["keypoints"]["name"],
                                segm_names=config["segmentation"]["name"],segm_labels=config["segmentation"]["name"],segm_classes=config["segmentation"]["classes"],
                                class_names=config["classification"]["name"],class_labels=config["classification"]["name"],class_numclass=config["classification"]["numclasses"],
                                class_classes=config["classification"]["classes"],reg_names=config["regression"]["name"],reg_labels=config["regression"]["name"])
    return model
 
 
if __name__=="__main__":
    # Simple training loop 
    data_dir=args.data_dir
    dataset=VegDataset(data_dir, transform=get_transform())
    model=build_model(config)    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True,collate_fn=detection_utils.collate_fn)
    model=build_model(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device=torch.device("cpu")
    model.to(device)
    pretrain= args.pretrain
    if pretrain:
        print(f"Loading pretrained !!!!")
        state_dict=torch.load(pretrain)
        model.load_state_dict(state_dict=state_dict,strict=False)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,collate_fn=detection_utils.collate_fn)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=1000,
                                                    gamma=0.1)

    num_epochs = args.epochs
    for epoch in tqdm(range(num_epochs+2)):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        if epoch%args.save_interval==0:
            PATH="weights"
            os.makedirs(PATH,exist_ok=True)
            torch.save(model.state_dict(), os.path.join(PATH,f"{epoch}_epoch_weights"))
            
