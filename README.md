## All with One Image Net (SMN)
## [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Asad-Ismail/SMN/issues) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FAsad-Ismail%2FSMN&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)



A Scaleable MultiTask Network Framework for image-based Object detection, Instance segmenation, Keypoint detection, Classification and Regression.
Our purpose is to solve to main problems while using deep neural network in production

1. Design scaleable multitask network with shared bakbone
2. Handle dataset with sparse supervision

### Example Multi Task 

In an image we want to detect cucumbers, color board segment fruits, segment their backbone, and also classify there neck and overall rating
  <p align="center">
    <img src="figs/ann_input.png" alt="animated" width=700 height=500 />
  </p>
  
### Example Sparse Supervision Data

  <p align="center">
    <img src="figs/data.png" alt="animated" width=600 height=200/>
  </p>
  
* Pipeiline of an Instance segmentation network like MaskRCNN with Keypoint Detector followed by seperate CNNs for per fruit classification/regression
 
First CNN network is for detection, segmentation, and keypoint detection like MaskRCNN which extracts the fruit patches or ROIs which are then fed to another network for multilabel classification and regression of each fruit. Figures below show this pipeline


  <p align="center">
    <img src="figs/MT.gif" alt="animated"  width=650 height=400  />
  </p>
                                        

  <p align="center">
    <img src="figs/masrcnn.png" alt="animated" width=650 height=300 />
  </p>
  
  
  <p align="center">
    <img src="figs/RPN.png" alt="animated" width=500 height=150 />
  </p>
  
  <p align="center">
    <img src="figs/resnet34.png" alt="animated" width=650 height=200 />
  </p>


* The second way is multi task learning. We propose here a multitask network with configureable inputs and that can be exapnded to do N segmentaiton,N classification and N regression tasks where N can be specified using a simple config file. The idea is to use shared features between task and use ROIs generated by RPN network to narrow the area of classification and regression. Giving whole input image to the network and performing multilabel classficiation and regression can require a lot of data so using ROIs provide a inductive bias to reduce the amount of data for training. Multitask networks in addition to providing better generalizations aslo can be paralleleized and be computed very efficently.

<p align="center">
    <img src="figs/SMN.png" alt="animated" width=750 height=400 />
  </p>
  This work provides a general way to do second type of multi task learning in a more general scaleable and userfriendly way

## Example
We will motivate with a real world example. Suppose we want to do fruit level phenotyping (Finding the visual traits of fruits) using imaging. We are intetrested in multiple traits of each fruit for example length, width, shape uniformity, backbone(curved_height), head, tail, neck and overall rating of each fruit. In summary we for this example we are interested in follwing traits of fruits 
1) Max Length and width of fruit, can be found by bounding boxes of objects (Object Detection)
2) Shape uniformity and length and width at different points of fruit, can be found by finding mask for each fruit (Instance segmentation)
3) Backbone (Curved Height), can be found by by finding masks of backbone (Instance semgentation)
4) Head and Tail of fruits are used for orientation correction of fruit, can be thought of image keypoint detection (Keypoint detection)
5) Neck and rating of fruit is determied by breeder and is given a categorical score form 1-5. where 1 is bad and 5 is good, they can be though of classification problem
In summary, we have object detection problem, multiple instance segmtentation problem, multiple multi label multi class classification problems and keypoint detection problem

### Example Input Image
  <p align="center">
    <img src="figs/input.png" alt="animated" width=700 height=450 />
  </p> 
  
### Example Annotated Image
  <p align="center">
    <img src="figs/ann_input.png" alt="animated" width=700 height=500 />
  </p>
  
### Prediction Results using multitask network

  <p align="center">
    <img src="figs/pred.png" alt="animated" width=700 height=500 />
  </p>


### Training and Predictions

1) Install requirements using pip install -r requirements.txt
2) In this framework we can specify all the tasks using a config file as shown below.Modify the config file according to the task
  
  <p align="center">
    <img src="figs/config.png" alt="animated" width=700 height=500 />
  </p>
  
3) To train the netowrk run train_vegnet.py  
4) To predcit run predict_vegnet.py 

***Quantitive results on each task for the private dataset to be released soon!!***

