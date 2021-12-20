## All with One Image Net (SMN)

A Scaleable MultiTask Network for image-based Object detection, Instance segmenation, Keypoint detection, Classification and Regression. 
We will motivate the problem by an example. Suppose we want to do vegetable phenotyping(Finding the traits of vegetables) using imaging. We are intetrested in multiple traits of each fruit for example length, width, shape uniformity, backbone(curved_height), Head,Tail, Neck and overall rating of each fruit. 
1) Max Length and width of fruit can be found by bounding boxes of objects (Object Detection)
2) Shape uniformity and length and width at different points of fruit can be found by finding mask for each fruit (Instance segmentation)
3) Backbone(Curved Height) can be found by by finding masks of backbone (Instance semgentation)
4) Head and Tail of fruits are used for orientation correction of fruit  they can be thpough of image keypoint detection (Keypoint detection)
5) Neck and rating of fruit is determied by breeder and is given a categorical score form 1-5. where 1 is bad and 5 is good. They can be though of classification problem
So we have object detection problem, instance segmtentation problem multiple multi class classification problems per objects and keypoint detection problem
### Example Input Image
  
  <p align="center">
    <img src="figs/img.png" alt="animated" width=700 height=500 />
  </p>
  
### Example Annotated Image
  <p align="center">
    <img src="figs/ann_input.png" alt="animated" width=700 height=500 />
  </p>


Typically there are two way to solve this. One is to have two CNNs one for detection, segmentation, and keypoint detection like MaskRCNN which extracts the fruit patches or ROIs which are then fed to another network for multilabel classification and regression of each fruit.

### KeyPoint Detector with MaskRCNN
  <p align="center">
    <img src="figs/pointnet.png" alt="animated" width=700 height=500 />
  </p>

In our example case N=2 for Neck and rating but in general can be N classification and N regression problems. The second way is multi task learning. We propose here a multitask network with configureable inputs and that can be exapnded to do N segmentaiton,N classification and N regression tasks where N can be specified by a config file

### Proposed solution
Typically there are two way to solve this. One is to have two neworks one for detection, segmentation, and keypoint detection like MaskRCNN which extracts the fruit patches or ROIs which are then fed to another newotk for n way classification of that fruit. In this case n=2 for Neck and rating but in general can be N classification and N regression problems.
A typical solution Network will look like below


We can esasliy specify the classfication, regression and semgentation heads as required using config file as shown below
  <p align="center">
    <img src="figs/config.png" alt="animated" width=700 height=500 />
  </p>



