## All with One Image Net (SMN)

A Scaleable MultiTask Network for image-based Object detection, Instance segmenation, Keypoint detection, Classification and Regression. 
We will motivate the problem by an example. Suppose we want to do vegetable phenotyping(Finding the traits of vegetables) using imaging. We are intetrested in multiple traits of each fruit for example length, width, shape uniformity, backbone(curved_height), Head,Tail, Neck and overall rating of each fruit. 
1) Max Length and width of fruit can be found by bounding boxes of objects (Object Detection)
2) Shape uniformity and length and width at different points of fruit can be found by finding mask for each fruit (Instance segmentation)
3) Backbone(Curved Height) can be found by by finding masks of backbone (Instance semgentation)
4) Head and Tail of fruits are used for orientation correction of fruit  they can be thpough of image keypoint detection (Keypoint detection)
5) Neck and rating of fruit is determied by breeder and is given a categorical score form 1-5. where 1 is bad and 5 is good. They can be though of classification problem
#### Example Input Image
  <p align="center">
    <img src="figs/img.png" alt="animated" width=700 height=500 />
  </p>
#### Example Annotated Image
  <p align="center">
    <img src="figs/ann_img.png" alt="animated" width=700 height=500 />
  </p>


  <p align="center">
    <img src="figs/multitask.png" alt="animated",width=500,height=500 />
  </p>
Coming Soon!!
