import torch
import numpy as np
from torch.utils.data import Dataset
import os
import cv2
from scipy.spatial import distance
import json
from utils import *
from PIL import Image
import yaml
import torchvision.transforms as T


config_file="model_config_cuc.yaml"
with open(config_file, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


class VegDataset(Dataset):
    """Veg Dataset loader."""

    def __init__(
        self,
        root_dir,
        vis=False,
        transform=None,
        use_cache=True
    ):
        """
        Args:
            root_dir (string): Directory with all the images.
            label_dir (str): Directory of labels in json format
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.valid_img_extensions = (".jpg", ".jpeg", ".png")
        self.valid_label_extensions = (".json",".csv")
        self.vis = vis
        self.classes=config["classes"]
        self.segm_names=config["segmentation"]["name"]
        self.segm_classes=config["segmentation"]["classes"]
        self.class_names=config["classification"]["name"]
        self.class_classes=config["classification"]["classes"]
        self.kp_names=config["keypoints"]["name"]
        self.kp_classes=config["keypoints"]["classes"]
        self.class_file= config["classification"]["file"]
        self.keypoints_file = config["keypoints"]["file"]
        self.segmentation_file= config["segmentation"]["file"]
        ## overall class map
        self.class_map={k:i for k,i in zip(self.classes,range(1,len(self.classes)+1))}
        self.rev_class_map={v:k for k,v in self.class_map.items()}
        self.use_cache=use_cache
        self.cache={}
        self.intialize_dataset()

    def intialize_dataset(self):
        # Also append the first classes of segmentation
        self.segm=load_segmentation_dataset(self.segmentation_file,config["classes"]+config["segmentation"]["name"][1:])
        self.kp=load_points_dataset_2(self.keypoints_file,self.kp_names)
        self.clas=load_class_dataset(self.class_file,self.class_names)
        self.imgs = []
        self.labels = []
        for metadata in self.segm.keys():
            self.imgs.append(metadata)

    def __len__(self):
        return len(self.imgs)

    def get_classes(self,cls,anno):
        px = [x for x, y in anno]
        py = [y for x, y in anno]
        classpoint = self.find_classes(cls, px,py)
        return classpoint[1]  
    
    
    def find_classes(self, claspoint, px,py):
        for point in claspoint:
            # very high intitial distances
            xmin=min(px)
            xmax=max(px)
            ymin=min(py)
            ymax=max(py)

            if point[0][0]>= xmin and point[0][0]<= xmax and point[0][1]>= ymin and point[0][1]<= ymax:
                return point
        raise(f"No Class annotaiton found!!")
    
    
    def get_keypoints(self,kps,anno):
        px = [x for x, y in anno]
        py = [y for x, y in anno]
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        keypoints = self.find_keyPoints(kps, poly)
        ## fix points which are none
        if keypoints[0] is not None:
            x1, y1 = keypoints[0]
        else:
            x1, y1 = 0.0, 0.0
        if keypoints[1] is not None:
            x2, y2 = keypoints[1]
        else:
            x2, y2 = 0.0, 0.0
        return [x1, y1,1, x2, y2,1]    
    
    
    def find_keyPoints(self, imgPoints, nz_points):
        """Associate the points to right fruit
        Args:
            imgPoints ([list]): All keypoints in the image
            nz_points ([type]): Desired fruit non zero points to match

        Returns:
            [tuple]: tuple of keypoints(head/tail) belongig to the fruit
        """
        mindistances = []
        for point in imgPoints:
            cpoint_head = None
            cpoint_tail = None
            # very high intitial distances
            dist_tail = float("inf")
            dist_head = float("inf")
            if point is not None and point[0] is not None:
                cpoint_head = min(nz_points, key=lambda c: distance.euclidean(c, point[0]))
                dist_head = distance.euclidean(cpoint_head, point[0])
            if point is not None and point[1] is not None:
                cpoint_tail = min(nz_points, key=lambda c: distance.euclidean(c, point[1]))
                dist_tail = distance.euclidean(cpoint_tail, point[1])
            mindistances.append(dist_tail + dist_head)
        index_min = min(range(len(mindistances)), key=mindistances.__getitem__)
        res = imgPoints[index_min]
        return res

    def combine_annotation(self,idx,imagename,img):
        """Combine image,annotation and keypoints to one annotation"""
        if idx in self.cache:
            #print(f"Using cache version of data!!")
            return self.cache[idx]
        h, w = img.shape[:2]
        segms = self.segm[imagename]
        kps=self.kp[imagename]
        clas=self.clas[imagename]
        class_segm=[]
        rem_segm={}
        all_bbox = []
        all_mask_imgs ={}
        all_kps = []
        all_classes = {}
        for seg in segms:
            if seg["class_id"] in self.classes:
                anno=seg["annotation"]
                mask_img = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask_img, np.int0([anno]), -1, 255, -1)
                class_segm.append(self.class_map[seg["class_id"]])
                px = [x for x, y in anno]
                py = [y for x, y in anno]
                all_bbox.append([np.min(px), np.min(py), np.max(px), np.max(py)])
                if "masks" in all_mask_imgs:
                    all_mask_imgs["masks"].append(mask_img)
                else:
                    all_mask_imgs["masks"]=[mask_img]
                # Get keypoints
                if seg["class_id"] in self.kp_classes:       
                    kp=self.get_keypoints(kps,anno)
                    all_kps.append(kp)
                else :
                    # append that keypoint is not visible
                    kp=[0,0,0,0,0,0]
                    all_kps.append(kp)
                # Get other classification classes
                for i,k in enumerate(clas.keys()):
                    if seg["class_id"] in self.class_classes[i]: 
                        c=self.get_classes(clas[k],anno=anno)
                        if k in all_classes:
                            all_classes[k].append(c)
                        else:
                            all_classes[k]=[c]
                    else:
                        #0 is ignore label
                        if k in all_classes:
                            all_classes[k].append(0)
                        else:
                            all_classes[k]=[0]
        ## segmentations apart form object segmentation
        for seg in segms:
            if seg["class_id"] in self.segm_names[1:]:
                anno=seg["annotation"]
                mask_img = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask_img, np.int0([anno]), -1, 255, -1)
                if seg["class_id"] in rem_segm:
                    rem_segm[seg["class_id"]].append(mask_img)
                else:
                    rem_segm[seg["class_id"]]=[mask_img]
        # For target boxes, masks and labels should be there other are optionals
        boxes = torch.as_tensor(all_bbox, dtype=torch.float32)
        labels = torch.as_tensor(class_segm, dtype=torch.int64)
        masks = torch.as_tensor(all_mask_imgs["masks"], dtype=torch.uint8)//255
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        keypoints=torch.as_tensor(all_kps,dtype=torch.float32)
        iscrowd = torch.zeros(labels.shape[0], dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor(idx)
        target["area"] = area
        # keypoints in order of Nxkx3
        target["keypoints"]=keypoints.reshape(len(all_kps),2,3)
        target["iscrowd"] = iscrowd
        
        ## append rest of classifications
        for k,v in all_classes.items():
            target[k]=torch.as_tensor(v, dtype=torch.int64)
        
        # append rest of segmentaitons
        for k,v in rem_segm.items():
            v=torch.as_tensor(v, dtype=torch.uint8)//255
            target[k]=torch.as_tensor(v, dtype=torch.int64)
        if self.use_cache:
            self.cache[idx]=(img,target)
        return img,target
        #print(target.keys())
        

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.imgs[idx])
        image =  cv2.imread(img_name)
        image,target = self.combine_annotation(idx,self.imgs[idx],image)   
        #print(target["keypoints"])     
        if self.transform is not None:
            image= self.transform(image)
        if self.vis:
            label_map=[self.rev_class_map[i.item()] for i in target["labels"]]
            vis_gen(image,target["masks"],target["boxes"],target["keypoints"],label_map, \
                    other_masks={"backbone":target["backbone"]},clas={"neck":target["neck"],"rating":target["rating"]},\
                    seg_labels=self.segm_classes,clas_labels=self.class_classes,kp_labels=self.kp_classes)
        return image,target


def get_transform():
    transform = T.Compose([T.ToTensor()])
    return transform     

if __name__ == "__main__":
    # Check the data loader
    root_dir = "/home/asad/projs/SMN/data/cucumber/"
    dataset=VegDataset(root_dir, vis=True,transform=get_transform())
    print(f"Length of veg data is {len(dataset)}")
    for i in range(len(dataset)):
        data = dataset[i]
        print(data)
