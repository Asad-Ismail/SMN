from distutils import extension
from cv2 import detail_DpSeamFinder
import hub
import numpy.random as random
import json
import os,sys
from tqdm import tqdm
import cv2
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

src_kp="hub://aismail2/complete_cucumber_keypoints"
dst_kp="/media/asad/8800F79D00F79104/hubdata/cucumber/keypoints"

src_det="hub://aismail2/cucumber_OD"
dst_det="/media/asad/8800F79D00F79104/hubdata/cucumber/detection"


def save_json_image(json_name,image_name,img,data,dst_dir=""):
    outfile={}
    outfile["img_name"]=image_name
    height,width=img.shape[:2]
    outfile["height"]=height
    outfile["width"]=width
    outfile["shapes"]=data
    with open(os.path.join(dst_dir,json_name), 'w') as out:
        json.dump(outfile, out)
    cv2.imwrite(os.path.join(dst_dir,image_name),img[...,::-1])
    
def get_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    max_cnt=None
    max_sze=0
    for cnt in contours:
        if len(cnt)>max_sze:
            max_sze=len(cnt)
            max_cnt=cnt
    if max_cnt is None:
        print("Contour is None")
        exit()
    max_cnt=max_cnt.squeeze().tolist()
    return max_cnt

def save_annotation(ds,i,dir_path=None):
    i=int(i)
    image=ds.images[i].numpy()
    masks=ds.masks[i].numpy().astype(np.uint8)*255
    all_cnts=[]
    for j in range(masks.shape[-1]):
        cnt=get_contours(masks[...,j])
        all_cnts.append(cnt)
    image_name=str(i)+".png"
    json_name=str(i)+".json"
    save_json_image(json_name,image_name,image,all_cnts,dir_path)
    
    
def save_pickle_image(pkl_name,image_name,img,data,dst_dir=""):
    #outfile={}
    #outfile["img_name"]=data
    #height,width=img.shape[:2]
    with open(os.path.join(dst_dir,pkl_name), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    cv2.imwrite(os.path.join(dst_dir,image_name),img[...,::-1])
    
def save_annotation_kp(ds,i,dir_path=None):
    i=int(i)
    image=ds.images[i].numpy()
    kps=ds.keypoints[i].numpy()
    boxes=ds.boxes[i].numpy()
    data={"kp":kps,"bbox":boxes}
    image_name=str(i)+".png"
    pickle_name=str(i)+".pkl"
    save_pickle_image(pickle_name,image_name,image,data,dir_path)

def get_data():
    np.random.seed(12)
    # make train and valid dataset in dst dir
    ## Get Detection data
    #ds = hub.load(src_det)
    #print(f"Detection Dataset size is {len(ds)}")
    #val_size=10
    #val_indices=np.random.choice(np.arange(len(ds.tensors["images"])), val_size)
    #train_dir=dst_det+"/train"
    #val_dir=dst_det+"/val"
    #os.makedirs(train_dir,exist_ok=True)
    #os.makedirs(val_dir,exist_ok=True)
    #for i in tqdm(val_indices):
    #    save_annotation(ds,i,val_dir)
    #train_indices=[i for i in range(len(ds.tensors["images"])) if i not in val_indices]
    #for i in tqdm(train_indices):
    #    save_annotation(ds,i,train_dir)
    
    ## Get Point Data
    ## Get Detection data
    ds = hub.load(src_kp)
    print(f"Keypoints Dataset size is {len(ds)}")
    #print(dir(ds))
    val_size=10
    val_indices=np.random.choice(np.arange(len(ds.tensors["images"])), val_size)
    train_dir=dst_kp+"/train"
    val_dir=dst_kp+"/val"
    os.makedirs(train_dir,exist_ok=True)
    os.makedirs(val_dir,exist_ok=True)
    for i in tqdm(val_indices):
        save_annotation_kp(ds,i,val_dir)
    train_indices=[i for i in range(len(ds.tensors["images"])) if i not in val_indices]
    for i in tqdm(train_indices):
        save_annotation_kp(ds,i,train_dir)


def write_text(image,text,point=(0,0),color=(255,0,0)):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 3
    # Using cv2.putText() method
    image = cv2.putText(image, text, point, font, fontScale, color, thickness, cv2.LINE_AA)
    return image

def vis_mask(vis_img,indicies,color=(255,120,0)):
    for j in range(len(indicies[0])):
        x = indicies[1][j]
        y = indicies[0][j]
        # viusalize masks
        cv2.circle(vis_img, (x, y), 1, color, 1)

def resize_images_cv(img):
    scale_percent = 40 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def vis_data(image, masks, bboxs,classes,keypoints=None,**kwargs):
    vis_img = (image.detach().numpy()*255).astype(np.uint8)
    vis_img=np.moveaxis(vis_img, 0, -1)
    vis_img=vis_img.copy()
    class_color={i:[random.uniform(0,255) for _ in range(3)] for i in np.unique(classes)}
    # offset for drawing text
    off_x=20
    off_y=50
    for i in range(masks.shape[0]):
        mask = masks[i][...,None].detach().numpy()
        bbox = np.int0(bboxs[i].detach().numpy().copy())
        indicies = np.where(mask >= 0.5)
        cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), class_color[classes[i]], 2)
        # If the mask is valid
        # write class name
        write_text(vis_img,classes[i],((bbox[0], bbox[1])))
        if (len(indicies[0])>0):
            vis_mask(vis_img,indicies=indicies,color=class_color[classes[i]])
            # Visualize bounding box
            #cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), class_color[classes[i]], 2)
            # write class name
            #write_text(vis_img,classes[i],((bbox[0], bbox[1])))

        ## Visualize keypoints
        if "kp_labels" in kwargs and keypoints is not None:
                keypoint = np.int0(keypoints[i].detach().numpy())
                # Visualize Keypoints
                cv2.circle(vis_img, (keypoint[0][0], keypoint[0][1]), 1, (0, 255, 255), 20)
                write_text(vis_img,"Head",(keypoint[0][0]+off_x, keypoint[0][1]+off_y))
                cv2.circle(vis_img, (keypoint[1][0], keypoint[1][1]), 1, (0, 255, 255), 20)
                write_text(vis_img,"Tail",(keypoint[1][0]+off_x, keypoint[1][1]+off_y))       
         
    vis_img=resize_images_cv(vis_img)
    #if"" kwargs["DEBUG"]:
    #cv2.imwrite("ann_input.png",vis_img)
    cv2.imshow("Input and labels", vis_img)
    cv2.waitKey(0)

class VegDetection(Dataset):
    """Veg Detection Dataset loader."""
    def __init__( self,root_dir,classes=["cucumber"],vis=False,transform=None,use_cache=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            label_dir (str): Directory of labels in json format
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = (".jpg", ".jpeg", ".png")
        self.classes=classes
        self.class_map={k:i for k,i in zip(self.classes,range(1,len(self.classes)+1))}
        self.rev_class_map={v:k for k,v in self.class_map.items()}
        self.vis = vis
        ## overall class map
        self.use_cache=use_cache
        self.cache={}
        self.intialize_dataset()
        
    def get_veg_dicts(self,data_dir):
        """_summary_

        Args:
            img_dir (str): Get Annotation Directory 

        Returns:
            _type_: Dataset dictoonary
        """
        json_files = [
            json_file
            for json_file in os.listdir(data_dir)
            if json_file.endswith(".json")
        ]
        dataset_dicts = []
        for idx, json_file in tqdm(enumerate(json_files),total=len(json_files)):
            for ext in self.extensions:
                filename = json_file.split(".")[0] + ext
                c_fname = os.path.join(data_dir, filename)
                img = cv2.imread(c_fname)
                if img is not None:
                    break
            if img is None:
                print(f"Image Not Found for {json_file}")
                raise (f"Image Not Found for {json_file}")
            #print(f"Processing json {json_file}")
            with open(os.path.join(data_dir, json_file)) as f:
                imgs_anns = json.load(f)
            record = {}
            height, width = img.shape[:2]
            record["file_name"] = c_fname
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
            annos = imgs_anns["shapes"]
            objs = []
            for anno in annos:
                px = [x for x, y in anno]
                py = [y for x, y in anno]
                poly = [(x, y ) for x, y in zip(px, py)]
                #poly = [p for x in poly for p in x]
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "segmentation": poly,
                    "category_id": 1,
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
            #break
        return dataset_dicts

    def intialize_dataset(self):
        # Also append the first classes of segmentation
        self.labels=self.get_veg_dicts(self.root_dir+"/train")
        self.imgs = []
        for item in self.labels:
            self.imgs.append(item["file_name"])
        assert len(self.imgs)==len(self.labels)

    def get_Data(self,idx):
        label=[]
        all_bbox = []
        all_mask =[]
        all_kps = []
        annotations=self.labels[idx]
        h=annotations["height"]
        w=annotations["width"]
        for ann in annotations["annotations"]:
            mask_img = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask_img, np.int0([ann["segmentation"]]), -1, 255, -1)
            label.append(ann["category_id"])
            all_mask.append(mask_img)
            px = [x for x, y in ann["segmentation"]]
            py = [y for x, y in ann["segmentation"]]
            all_bbox.append([np.min(px), np.min(py), np.max(px), np.max(py)])
            # Dummy keypoint
            kp=[0,0,0,0,0,0]
            all_kps.append(kp)
        
        # For target boxes, masks and labels should be there other are optionals
        boxes = torch.as_tensor(all_bbox, dtype=torch.float32)
        labels = torch.as_tensor(label, dtype=torch.int64)
        masks = torch.as_tensor(all_mask, dtype=torch.uint8)//255
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
        return target
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.imgs[idx]
        image =  cv2.imread(img_name)
        target = self.get_Data(idx)   
        if self.transform is not None:
            image= self.transform(image)
        if self.vis:
            label_map=[self.rev_class_map[i.item()] for i in target["labels"]]
            vis_data(image,target["masks"],target["boxes"],label_map)
        return image,target
    
    def __len__(self):
        return len(self.imgs)



class Vegkeypoint(Dataset):
    """Veg Detection Dataset loader."""
    def __init__( self,root_dir,classes=["cucumber"],kp_names=["Head","Tail"],vis=False,transform=None,use_cache=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            label_dir (str): Directory of labels in json format
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = (".jpg", ".jpeg", ".png")
        self.classes=classes
        self.class_map={k:i for k,i in zip(self.classes,range(1,len(self.classes)+1))}
        self.rev_class_map={v:k for k,v in self.class_map.items()}
        self.kp_names=kp_names
        self.vis = vis
        ## overall class map
        self.use_cache=use_cache
        self.cache={}
        self.intialize_dataset()
        
    def get_veg_keypoints(self,data_dir):
        """_summary_

        Args:
            img_dir (str): Get Annotation Directory 

        Returns:
            _type_: Dataset dictoonary
        """
        pkl_files = [
            pkl_file
            for pkl_file in os.listdir(data_dir)
            if pkl_file.endswith(".pkl")
        ]
        dataset_dicts = []
        for idx, pkl_file in tqdm(enumerate(pkl_files),total=len(pkl_files)):
            for ext in self.extensions:
                filename = pkl_file.split(".")[0] + ext
                c_fname = os.path.join(data_dir, filename)
                img = cv2.imread(c_fname)
                if img is not None:
                    break
            if img is None:
                print(f"Image Not Found for {pkl_file}")
                raise (f"Image Not Found for {pkl_file}")
            #print(f"Processing json {json_file}")
            with open(os.path.join(data_dir, pkl_file), 'rb') as f:
                data = pickle.load(f)
            record = {}
            height, width = img.shape[:2]
            record["file_name"] = c_fname
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
            record["kps"] = data["kp"]
            record["bbox"] = data["bbox"]
            dataset_dicts.append(record)
            #break
        return dataset_dicts

    def intialize_dataset(self):
        # Also append the first classes of segmentation
        self.labels=self.get_veg_keypoints(self.root_dir+"/train")
        self.imgs = []
        for item in self.labels:
            self.imgs.append(item["file_name"])
        assert len(self.imgs)==len(self.labels)

    def get_Data(self,idx):
        label=[]
        all_bbox = []
        all_mask =[]
        all_kps = []
        annotations=self.labels[idx]
        h=annotations["height"]
        w=annotations["width"]
        for  i in range(annotations["kps"].shape[-1]):
            mask_img = np.zeros((h, w), dtype=np.uint8)    
            all_mask.append(mask_img)
            # Dummy keypoint
            kp=[0,0,0,0,0,0]
            kp=annotations["kps"][:,i]
            all_bbox.append(annotations["bbox"][i,:])
            label.append(1)
            all_kps.append(kp)
        
        # For target boxes, masks and labels should be there other are optionals
        boxes = torch.as_tensor(all_bbox, dtype=torch.float32)
        labels = torch.as_tensor(label, dtype=torch.int64)
        masks = torch.as_tensor(all_mask, dtype=torch.uint8)//255
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
        return target
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.imgs[idx]
        image =  cv2.imread(img_name)
        target = self.get_Data(idx)   
        if self.transform is not None:
            image= self.transform(image)
        if self.vis:
            label_map=[self.rev_class_map[i.item()] for i in target["labels"]]
            vis_data(image,target["masks"],target["boxes"],label_map,keypoints=target["keypoints"],kp_labels=self.kp_names)
        return image,target
    
    def __len__(self):
        return len(self.imgs)



def get_transform():
    transform = T.Compose([T.ToTensor()])
    return transform     

if __name__=="__main__":
    #get_data()
    dl=VegDetection(dst_det,classes=["cucumber"],vis=True,transform=get_transform())
    #dl_kp=Vegkeypoint(dst_kp,vis=True,transform=get_transform())
    for idx in range(len(dl)):
        image,target=dl[idx]