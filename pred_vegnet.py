from torch.utils import data
import torchvision.transforms as T
import torch
from utils.utils import  vis_data
from utils.data_loader import  VegDataset
from utils.engine import train_one_epoch, evaluate
import os
import utils.detection_utils as detection_utils
from model.vegnet_v1 import *
import yaml

torch.backends.cudnn.benchmark=True

#build model
config_file="model_config.yaml"
with open(config_file, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def get_transform():
    transform = T.Compose([T.ToTensor()])
    return transform

        
def vis_and_process_preds(image,prediction,dataset,test_th=0.3):
    scores=prediction["scores"].cpu()
    classes=prediction["labels"].cpu()
    valid_scores=scores>=test_th
    boxes=prediction["boxes"][valid_scores].cpu()
    keypoints=prediction["keypoints"][valid_scores].cpu()
    masks=prediction["masks"][:,0][valid_scores].detach().squeeze().cpu()
    bb=prediction["backbone"][:,0][valid_scores].detach().squeeze().cpu()
    rate=prediction["rating"][valid_scores].cpu()
    neck=prediction["neck"][valid_scores].cpu()
    image=image.cpu()
    label_map=[dataset.rev_class_map[i.item()] for i in classes]
    # visualize results
    vis_data(image,masks,boxes,keypoints,label_map, \
        other_masks={"backbone":bb},clas={"neck":neck,"rating":rate},\
        seg_labels=dataset.segm_classes,clas_labels=dataset.class_classes,kp_labels=dataset.kp_classes)
        
 
def build_model(config):
    model = vegnet_resnet50_fpn(pretrained=True,num_classes=config["numclasses"],class_map=config["classes"],num_keypoints=config["keypoints"]["num"],kp_name=config["keypoints"]["name"],
                                segm_names=config["segmentation"]["name"],segm_labels=config["segmentation"]["name"],segm_classes=config["segmentation"]["classes"],
                                class_names=config["classification"]["name"],class_labels=config["classification"]["name"],class_numclass=config["classification"]["numclasses"],
                                class_classes=config["classification"]["classes"],reg_names=config["regression"]["name"],reg_labels=config["regression"]["name"])
    return model
 

root_dir="data/cucumber"
dataset=VegDataset(root_dir, transform=get_transform())
model=build_model(config)    
# read and build dataset loader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True,collate_fn=detection_utils.collate_fn)
model=build_model(config)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device=torch.device("cpu")
model.to(device)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,collate_fn=detection_utils.collate_fn)
for i,(images,data) in enumerate(data_loader):
    images = list(image.to(device) for image in images)
    model.load_state_dict(torch.load(os.path.join("mask_backbone_keypoint__rating_weights","10000_epoch_weights")))
    model.eval()
    predictions = model(images)   
    prediction=predictions[0]
    vis_and_process_preds(images[0],prediction,dataset)