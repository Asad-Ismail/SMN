
from torch.utils import data
import torchvision
#import vision_transforms as vTransform
import torchvision.transforms as T
import torch
from utils import load_backbone, load_points_dataset
import detection_utils
from data_loader import  VegDataset
from engine import train_one_epoch, evaluate
import detection_utils as utils
import os
import numpy as np
import cv2
from tqdm import tqdm
from veg_model.vegnet_v1 import *
from torch.utils.mobile_optimizer import optimize_for_mobile
import yaml


#build model
config_file="model_config.yaml"
with open(config_file, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)



def get_transform():
    transform = T.Compose([T.ToTensor()])
    return transform

def vis_mask(img,mask,color=(0,255,0)):
    indxes=np.where(mask>=0.5)
    ys=indxes[0]
    xs=indxes[1]
    for i in range(len(ys)):
        cv2.circle(img,(xs[i],ys[i]),1,color,-1)
        
 
def build_model(config):
    model = vegnet_resnet50_fpn(pretrained=True,num_classes=config["numclasses"],class_map=config["classes"],num_keypoints=config["keypoints"]["num"],kp_name=config["keypoints"]["name"],
                                segm_names=config["segmentation"]["name"],segm_labels=config["segmentation"]["name"],segm_classes=config["segmentation"]["classes"],
                                class_names=config["classification"]["name"],class_labels=config["classification"]["name"],class_numclass=config["classification"]["numclasses"],
                                class_classes=config["classification"]["classes"],reg_names=config["regression"]["name"],reg_labels=config["regression"]["name"])
    return model
 
 


model=build_model(config)    
# read and build dataset loader
kp_file = "/media/asad/adas_cv_2/test_kp.csv"
backbone="/media/asad/adas_cv_2/backbone_labels.json"
rating="/media/asad/adas_cv_2/fruit_rating.json"
root_dir = "/media/asad/ADAS_CV/datasets_Vegs/pepper/one_annotated/train/"
gt_points = load_points_dataset(kp_file)
backbone= load_backbone(backbone)
rating= load_backbone(rating)
dataset=VegDataset(root_dir, root_dir, kp_file, gt_points,backbone=backbone,rating=rating,resize=None, vis=True, transform=get_transform())

for i,t in dataset:
    print(i)
    print(t)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1,collate_fn=detection_utils.collate_fn)


model=build_model(config)

# tracing and onnx conversion
#model.eval()
#example = [torch.rand(3, 300, 400)]
#out=model(example)
#example = [torch.rand(3, 300, 400)]
#torch.onnx.export(model, example, "veg.onnx", opset_version = 11)
#traced_script_module = torch.jit.trace(model, example,strict=False)
#traced_script_module_optimized = optimize_for_mobile(traced_script_module)
#traced_script_module_optimized._save_for_lite_interpreter("model.ptl")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device=torch.device("cpu")
model.to(device)
#print(model)
train=False
#pretrain="mask_keypoint_weights/950_epoch_weights"
pretrain=None
if pretrain:
    state_dict=torch.load(pretrain)
    model.load_state_dict(state_dict=state_dict,strict=False)
if train: 
    #model.load_state_dict(torch.load(os.path.join("output_weights_vision","740_epoch_weights")))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,collate_fn=detection_utils.collate_fn)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=1000,
                                                    gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 3000

    for epoch in tqdm(range(num_epochs+2)):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        #evaluate(model, data_loader_test, device=device)
        if epoch%1000==0:
            PATH="mask_backbone_keypoint__rating_weights"
            os.makedirs(PATH,exist_ok=True)
            torch.save(model.state_dict(), os.path.join(PATH,f"{epoch}_epoch_weights"))
            

else:
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,collate_fn=detection_utils.collate_fn)
    for images,data in data_loader:
    #images,targets = next(iter(data_loader))
        images = list(image.to(device) for image in images)
        model.load_state_dict(torch.load(os.path.join("mask_backbone_keypoint__rating_weights","3000_epoch_weights")))
        #model.load_state_dict(torch.load("/home/asad/projs/vegNetPytorch/mask_keypoint_weights/950_epoch_weights"))
        model.eval()
        predictions = model(images)   
        prediction=predictions[0]
        boxes=prediction["boxes"]
        scores=prediction["scores"]
        keypoints=prediction["keypoints"]
        masks=prediction["masks"]
        bb=prediction["backbone"]
        rate=prediction["rating"]
        #VISUALIZE DETECTION
        cv_image=images[0].detach().cpu().numpy()*255
        cv_image=np.ascontiguousarray(np.moveaxis(cv_image, 0, -1),dtype=np.uint8)
        print(scores)
        for i in range(boxes.shape[0]):
            bbox=np.int0(boxes[i].detach().cpu().numpy())
            point=np.int0(keypoints[i].detach().cpu().numpy())
            mask=masks[i].detach().squeeze().cpu().numpy()
            back=bb[i].detach().squeeze().cpu().numpy()
            vis_mask(cv_image,mask)
            vis_mask(cv_image,back,(255,0,255))
            cv2.circle(cv_image,point[0,:2],20,(0,255,255),-1)
            cv2.circle(cv_image,point[1,:2],20,(0,255,255),-1)
            cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
        
        scale_percent = 30 # percent of original size
        width = int(cv_image.shape[1] * scale_percent / 100)
        height = int(cv_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(cv_image, dim, interpolation = cv2.INTER_AREA)
        
        cv2.imshow("Results",resized)
        cv2.waitKey(0)

