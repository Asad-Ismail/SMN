import torch
import numpy as np
from torch.utils.data import Dataset
import os
import cv2
from scipy.spatial import distance
import json
from utils import *
from PIL import Image


class VegDataset(Dataset):
    """Veg Dataset loader."""

    def __init__(
        self,
        root_dir,
        label_dir,
        points_file,
        keypoints,
        backbone=None,
        rating=None,
        resize=1024,
        vis=True,
        transform=None,
    ):
        """
        Args:
            root_dir (string): Directory with all the images.
            label_dir (str): Directory of labels in json format
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.transform = transform
        self.points_file = points_file
        self.keypoints = keypoints
        self.valid_img_extensions = (".jpg", ".jpeg", ".png")
        self.valid_label_extensions = ".json"
        self.resize = resize
        self.vis = vis
        self.backbone=backbone
        self.rating=rating
        self.intialize_dataset()

    def intialize_dataset(self):
        self.imgs = []
        self.labels = []
        for filename in os.listdir(self.root_dir):
            if filename.endswith(self.valid_img_extensions):
                self.imgs.append(filename)
                label_path = os.path.join(self.label_dir, filename[:-4] + ".json")
                assert os.path.exists(
                    label_path
                ), "Label for image {filename} not found!!"
                self.labels.append(filename[:-4] + ".json")

    def __len__(self):
        return len(self.imgs)
    

    def match_bakbone(self,imagename):
        idx=None
        data=self.backbone["images"]
        for d in data:
            if d["file_name"]==imagename:
                idx=d["id"]
                break
        assert idx is not None, f"Backbone annotaiton of {imagename} not found!!"
        annos=[]
        data=self.backbone["annotations"]
        for d in data:
            if d["image_id"]==idx:
                tmp=[]
                seg=d["segmentation"][0]
                for i in range(0,len(seg)-1,2):
                    tmp.append([seg[i],seg[i+1]]) 
                annos.append(tmp)
        assert annos!=[],"The annotation for Backbone of {imagename} is empty"
        return annos
    
    def match_rating(self,imagename):
        data=self.rating
        for d in data:
            if d["image"]==imagename:
                return int(d["annotations"][0])
        raise ValueError(f"The {imagename} does not exists")
        
        

    def get_imagePoints(self, imgPoints, nz_points):
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
                cpoint_head = min(
                    nz_points, key=lambda c: distance.euclidean(c, point[0])
                )
                dist_head = distance.euclidean(cpoint_head, point[0])
            if point is not None and point[1] is not None:
                cpoint_tail = min(
                    nz_points, key=lambda c: distance.euclidean(c, point[1])
                )
                dist_tail = distance.euclidean(cpoint_tail, point[1])
            mindistances.append(dist_tail + dist_head)
        index_min = min(range(len(mindistances)), key=mindistances.__getitem__)
        res = imgPoints[index_min]
        return res

    def combine_annotation(self, idx,img_file, label_file, img,back_ann=None,rating=None):
        """Combine image,annotation and keypoints to one annotation"""
        with open(os.path.join(self.label_dir, label_file)) as f:
            imgs_anns = json.load(f)
        annos = imgs_anns["shapes"]
        h, w = img.shape[:2]
        num = len(annos)
        all_bbox = []
        all_mask_imgs = np.zeros((num, h, w), dtype=np.uint8)
        all_points = []
        all_classes = []
        # Get annotations for fruits
        for i, anno in enumerate(annos):
            mask_img = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask_img, np.int0([anno]), -1, 255, -1)
            px = [x for x, y in anno]
            py = [y for x, y in anno]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            keypoints = self.get_imagePoints(self.keypoints[img_file], poly)
            ## fix points which are none
            if keypoints[0] is not None:
                x1, y1 = keypoints[0]
            else:
                x1, y1 = 0.0, 0.0
            if keypoints[0] is not None:
                x2, y2 = keypoints[1]
            else:
                x2, y2 = 0.0, 0.0
            # x,y,visibility
            all_points.append([x1, y1,1, x2, y2,1])
            all_bbox.append([np.min(px), np.min(py), np.max(px), np.max(py)])
            all_mask_imgs[i] = mask_img
            all_classes.append(1)
        
            

        #all_mask_imgs = np.moveaxis(all_mask_imgs, 0, -1)
        
        boxes = torch.as_tensor(all_bbox, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(all_classes, dtype=torch.int64)
        masks = torch.as_tensor(all_mask_imgs, dtype=torch.uint8)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        keypoints=torch.as_tensor(all_points,dtype=torch.float32)
        iscrowd = torch.zeros(labels.shape[0], dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        #normalizing masks between 0 and 1
        target["masks"] = masks//255
        target["image_id"] = torch.tensor(idx)
        target["area"] = area
        # keypoints in order of Nxkx3
        target["keypoints"]=keypoints.reshape(len(all_points),2,3)
        #print(target["keypoints"])
        target["iscrowd"] = iscrowd
        
        
        # Get annotations for backbone
        if back_ann is not None:
            num_back = len(back_ann)
            back_mask_imgs = np.zeros((num_back, h, w), dtype=np.uint8)
            for i, anno in enumerate(back_ann):
                mask_img = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask_img, np.int0([anno]), -1, 255, -1)
                back_mask_imgs[i] = mask_img
            back_masks = torch.as_tensor(back_mask_imgs, dtype=torch.uint8)
            back_masks=back_masks//255
            target["backbone"]=back_masks
        
        if rating:
            rating = torch.as_tensor([rating], dtype=torch.int64)
            target["rating"]=rating
        
        return img, target

        
    
    def resize_transform(self,image, data):
        
        masks = data["masks"]*255
        masks=masks.permute(1,2,0)
        keypoints = data["keypoints"]
        keypoints=keypoints.numpy().astype(np.int32)
        back_masks=data["backbone"]*255
        back_masks=back_masks.permute(1,2,0)
        # if vis:
        #    vis_dataset(image, masks, bbox, keypoints, "Original Data")

        image, window, scale, padding, _ = resize_image(image, min_dim=self.resize, max_dim=self.resize)
        masks = resize_mask(masks, scale, padding)
        back_masks = resize_mask(back_masks, scale, padding)
        bboxes = extract_bboxes(masks)
        keypoints = resize_points(keypoints, scale, window)

        if self.vis:
            vis_dataset(image, masks, bboxes, keypoints, "Transformed Data",box_format="xy",back_masks=back_masks)

        # replace original data with transformed data
        data["masks"] = torch.tensor(masks, dtype=torch.uint8)
        data["boxes"] = torch.tensor(bboxes, dtype=torch.float32)
        data["keypoints"] = torch.tensor(keypoints, dtype=torch.int64)
        
        return image,data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.imgs[idx])
        image =  cv2.imread(img_name)
        label_name = self.labels[idx]
        if self.backbone:
            back_ann=self.match_bakbone(self.imgs[idx])
        if self.rating:
            rate=self.match_rating(self.imgs[idx])
        image,target = self.combine_annotation(idx,self.imgs[idx], label_name, image,back_ann=back_ann,rating=rate)
        
        if self.resize:
            image, target = self.resize_transform(image,target)
            
        if self.transform is not None:
            image= self.transform(image)

        return image,target

        

if __name__ == "__main__":
    # Check the data loader
    kp_file = "/media/asad/adas_cv_2/test_kp.csv"
    root_dir = "/media/asad/ADAS_CV/datasets_Vegs/pepper/one_annotated/train/"
    gt_points = load_points_dataset(kp_file)
    vegdata = VegDataset(root_dir, root_dir, kp_file, gt_points, vis=False)
    print(f"Length of veg data is {len(vegdata)}")
    for i in range(len(vegdata)):
        data = vegdata[i]
        print(data)
