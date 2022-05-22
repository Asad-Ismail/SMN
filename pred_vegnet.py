from matplotlib.pyplot import axes
from torch.utils import data
import torchvision.transforms as T
import torch
from utils.util import  vis_data
from utils.data_loader_seperate import  VegDataset
from utils.engine import train_one_epoch, evaluate
import os
import utils.detection_utils as detection_utils
from model.vegnet_v1 import *
import yaml
import argparse
import cv2
from utils.data_loader_seg_keypoints import  VegDetection,Vegkeypoint

argp=argparse.ArgumentParser()
argp.add_argument("--use-cudnn",type=bool,default=True)
argp.add_argument("--config",type=str,default="model_config_seg_kp.yaml",help="Model Configuration")
argp.add_argument("--data-dir",type=str,default="data/cucumber",help="Data Directory")
argp.add_argument("--weights",type=str,default="/media/asad/8800F79D00F79104/multitask_weights/multitask_weights/100_epoch_weights",help="Model Configuration")
args=argp.parse_args()
torch.backends.cudnn.benchmark=args.use_cudnn

#build config
config_file=args.config
with open(config_file, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def get_transform():
    transform = T.Compose([T.ToTensor()])
    return transform

        
def vis_and_process_preds(image,prediction,dataset,test_th=0.8):
    scores=prediction["scores"].cpu()
    classes=prediction["labels"].cpu()
    valid_scores=scores>=test_th
    classes=classes[valid_scores]
    boxes=prediction["boxes"][valid_scores].cpu()
    keypoints=prediction["keypoints"][valid_scores].cpu()
    #masks=prediction["masks"][:,0][valid_scores].detach().squeeze().cpu()
    masks=prediction["masks"][valid_scores].squeeze(dim=1).detach().cpu()
    #bb=prediction["backbone"][:,0][valid_scores].detach().squeeze().cpu()
    #rate=prediction["rating"][valid_scores].cpu()
    #neck=prediction["neck"][valid_scores].cpu()
    image=image.cpu()
    label_map=[dataset.rev_class_map[i.item()] for i in classes]
    # visualize results
    #vis_data(image,masks,boxes,label_map,keypoints, other_masks={"backbone":bb},clas={"neck":neck,"rating":rate},\
    #    seg_labels=dataset.segm_classes,clas_labels=dataset.class_classes,kp_labels=dataset.kp_classes)
    
    #vis_data(image,masks,boxes,keypoints,label_map, \
    #    clas={"neck":neck,"rating":rate},\
    #    seg_labels=dataset.segm_classes,clas_labels=dataset.class_classes,kp_labels=dataset.kp_classes)
    #vis_data(image,masks,boxes,label_map,keypoints,seg_labels=dataset.segm_classes,clas_labels=dataset.class_classes,kp_labels=dataset.kp_classes) 
    vis_data(image,masks,boxes,label_map,keypoints,seg_labels=["cucumbers"],clas_labels=["cucumber"],kp_labels=["cucumber"]) 
    #vis_data(image,masks,boxes,label_map,seg_labels=dataset.segm_classes,clas_labels=dataset.class_classes,kp_labels=dataset.kp_classes)  
 
def build_model(config):
    model = vegnet_resnet50_fpn(pretrained=True,num_classes=config["numclasses"],class_map=config["classes"],num_keypoints=config["keypoints"]["num"],kp_name=config["keypoints"]["name"],
                                segm_names=config["segmentation"]["name"],segm_labels=config["segmentation"]["name"],segm_classes=config["segmentation"]["classes"],
                                class_names=config["classification"]["name"],class_labels=config["classification"]["name"],class_numclass=config["classification"]["numclasses"],
                                class_classes=config["classification"]["classes"],reg_names=config["regression"]["name"],reg_labels=config["regression"]["name"])
    return model
 
 
 
if __name__=="__main__":
    data_dir=args.data_dir
    #dataset=VegDataset(data_dir, transform=get_transform())
    model=build_model(config)    
    # read and build dataset loader
    #data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True,collate_fn=detection_utils.collate_fn)
    model=build_model(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device=torch.device("cpu")
    model.to(device)
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    #kp_train="/media/asad/8800F79D00F79104/hubdata/cucumber/keypoints"
    #dataset=Vegkeypoint(kp_train,vis=False,transform=get_transform())
    
    det_train="/media/asad/8800F79D00F79104/hubdata/cucumber/detection"
    dataset=VegDetection(det_train,classes=["cucumber"],vis=False,transform=get_transform())
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,collate_fn=detection_utils.collate_fn)
    #data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,  shuffle=False, num_workers=1,collate_fn=detection_utils.collate_fn)
    for i,(images,data) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        #writer.add_graph(model, [images])
        #writer.close()  
        predictions = model(images) 

        prediction=predictions[0]
        vis_and_process_preds(images[0],prediction,dataset)
    #imgdir="data/cucumber"
    #for fp in os.listdir(imgdir):
    #    if fp.endswith(".jpg",".png",".jpeg"):
    #        images=[cv2.imread(os.path.join(imgdir,fp))]
    #        images = list(image.to(device) for image in images)
    #        predictions = model(images)   
    #        prediction=predictions[0]
    #        vis_and_process_preds(images[0],prediction,dataset)