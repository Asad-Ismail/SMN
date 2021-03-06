import torch
from torch import nn
from collections import OrderedDict
from torchvision.ops import MultiScaleRoIAlign
import torch.nn.functional as F

from torchvision.models.detection._utils import overwrite_eps
from torch.hub import load_state_dict_from_url

from .faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers


print(f"Import Succesfull!!")


class Vegnet(FasterRCNN):
    """
    Implements VegNet based on FasterRCNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
            ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - keypoints (FloatTensor[N, K, 3]): the K keypoints location for each of the N instances, in the
          format [x, y, visibility], where visibility=0 means that the keypoint is not visible.
        - masks (Int64Tensor[N,H,W]): Binary masks of objects
        ........
        ........
        Any other quanttities that needs to be classified or regressed as specified in config file with shape (Inttensor[N] or Floattensor[N] )

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the keypoint loss.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
            ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction
        - keypoints (FloatTensor[N, K, 3]): the locations of the predicted keypoints, in [x, y, v] format.
        - masks (FloatTesnsor[N,H,W]): masks of objects
        .........
        .........
        Any other quantities that needs to be classified or regressed as specified in config file with shape (Inttensor[N] or Floattensor[N] )

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        keypoint_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
             the locations indicated by the bounding boxes, which will be used for the keypoint head.
        keypoint_head (nn.Module): module that takes the cropped feature maps as input
        keypoint_predictor (nn.Module): module that takes the output of the keypoint_head and returns the
            heatmap logits

    Example::

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import KeypointRCNN
        >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
        >>>
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        >>> # KeypointRCNN needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the RPN generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        >>>                                    aspect_ratios=((0.5, 1.0, 2.0),))
        >>>
        >>> # let's define what are the feature maps that we will
        >>> # use to perform the region of interest cropping, as well as
        >>> # the size of the crop after rescaling.
        >>> # if your backbone returns a Tensor, featmap_names is expected to
        >>> # be ['0']. More generally, the backbone should return an
        >>> # OrderedDict[Tensor], and in featmap_names you can choose which
        >>> # feature maps to use.
        >>> roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        >>>                                                 output_size=7,
        >>>                                                 sampling_ratio=2)
        >>>
        >>> keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        >>>                                                          output_size=14,
        >>>                                                          sampling_ratio=2)
        >>> # put the pieces together inside a KeypointRCNN model
        >>> model = KeypointRCNN(backbone,
        >>>                      num_classes=2,
        >>>                      rpn_anchor_generator=anchor_generator,
        >>>                      box_roi_pool=roi_pooler,
        >>>                      keypoint_roi_pool=keypoint_roi_pooler)
        >>> model.eval()
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=None, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 # Mask parameter
                 mask_roi_pool=None, mask_head=None, mask_predictor=None,
                 # keypoint parameters
                 keypoint_roi_pool=None, keypoint_head=None, keypoint_predictor=None,
                 num_keypoints=2,
                 #Classification head
                 class_roi_pool=None, class_head=None, class_predictor=None,class_classes=None,
                 class_map=None,
                 ### Build specific branches for each task
                 kp_name=None,
                 segm_names=None,segm_labels=None,segm_classes=None,
                 class_names=None,class_labels=None,class_numclass=None,
                 reg_names=None,reg_labels=None):

        assert isinstance(keypoint_roi_pool, (MultiScaleRoIAlign, type(None)))
        
        if min_size is None:
            # can fix the size for inference
            min_size = (800)
            #min_size = (640, 672, 704, 736, 768, 800)

        if num_classes is not None:
            if keypoint_predictor is not None:
                raise ValueError("num_classes should be None when keypoint_predictor is specified")

        out_channels = backbone.out_channels
        
        if segm_names:
            # Mask Detection stuff
            mask_roi_pool=nn.ModuleDict()
            mask_head=nn.ModuleDict()
            mask_predictor=nn.ModuleDict()
            valid_segm_classes=[]
            for i,name in enumerate(segm_names):
                #maskroi pool
                mask_roi_pool_tmp = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=28, sampling_ratio=2)
                #mask_roi_pool.append(mask_roi_pool_tmp)
                mask_roi_pool[name]=mask_roi_pool_tmp
                #maskroi heads
                mask_layers = (256, 256, 256, 256)
                mask_dilation = 1
                mask_head_tmp = MaskRCNNHeads(out_channels, mask_layers, mask_dilation,i)
                #mask_head.append(mask_head_tmp)
                mask_head[name]=mask_head_tmp
                #maskroi predictor
                mask_predictor_in_channels = 256  # == mask_layers[-1]
                mask_dim_reduced = 256
                mask_predictor_tmp = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes=len(segm_classes[i])+1,index=i)
                #mask_predictor.append(mask_predictor_tmp)
                mask_predictor[name]=mask_predictor_tmp
                # Find the index of valid classes like some 
                v_index=[j for j, e in enumerate(class_map,start=1) if e in segm_classes[i]]
                valid_segm_classes.append(v_index)
        
        if kp_name and kp_name!="None":
            ## Keypoint Detection stuff
            if keypoint_roi_pool is None:
                keypoint_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],output_size=14,sampling_ratio=2)

            if keypoint_head is None:
                keypoint_layers = tuple(512 for _ in range(8))
                keypoint_head = KeypointRCNNHeads(out_channels, keypoint_layers)

            if keypoint_predictor is None:
                keypoint_dim_reduced = 512  # == keypoint_layers[-1]
                keypoint_predictor = KeypointRCNNPredictor(keypoint_dim_reduced, num_keypoints)
        
        if class_names and class_names!="None":
            class_roi_pool=nn.ModuleDict()
            class_head=nn.ModuleDict()
            class_predictor=nn.ModuleDict()
            valid_class_classes=[]
            for i,name in enumerate(class_names):
                #classroi pool
                class_roi_pool_tmp = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
                #class_roi_pool.append(class_roi_pool_tmp)
                class_roi_pool[name]=class_roi_pool_tmp
                #classroi heads
                class_layers = (256, 256, 256, 256)
                class_dilation = 1
                class_head_tmp = ClassRCNNHeads(out_channels, class_layers, class_dilation,i)
                #class_head.append(class_head_tmp)
                class_head[name]=class_head_tmp
                #classroi predictor
                class_predictor_in_channels = 256  # == mask_layers[-1]
                class_dim_reduced = 256
                # num classes+1 to add background class also used for unknown class
                class_predictor_tmp = ClassRCNNPredictor(class_predictor_in_channels, class_dim_reduced, class_numclass[i]+1,i)
                #class_predictor.append(class_predictor_tmp)
                class_predictor[name]=class_predictor_tmp
                # Find the index of valid classes like some 
                v_index=[j for j, e in enumerate(class_map,start=1) if e in class_classes[i]]
                valid_class_classes.append(v_index)

        super(Vegnet, self).__init__(
            backbone, num_classes,
            # transform parameters
            min_size, max_size,
            image_mean, image_std,
            # RPN-specific parameters
            rpn_anchor_generator, rpn_head,
            rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_score_thresh,
            # Box parameters
            box_roi_pool, box_head, box_predictor,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            # specific names for brancehs
            kp_name=kp_name,
            segm_names=segm_names,segm_labels=segm_labels,
            class_names=class_names,class_labels=class_labels,
            reg_names=reg_names,reg_labels=reg_labels
            )
        
        if kp_name:
            self.roi_heads.keypoint_roi_pool = keypoint_roi_pool
            self.roi_heads.keypoint_head = keypoint_head
            self.roi_heads.keypoint_predictor = keypoint_predictor
        
        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor =mask_predictor
        self.roi_heads.mask_valid_classes = valid_segm_classes
        
        if class_names and class_names!="None":
            self.roi_heads.class_roi_pool = class_roi_pool
            self.roi_heads.class_head = class_head
            self.roi_heads.class_predictor = class_predictor
            self.roi_heads.class_valid_classes = valid_class_classes
        
        ### add names of all the classes for freezing and loss zeroing. Could be done better by integrating names in the original names list
        ## available for each task in ROI heads
        self.roi_heads.roi_head_names=[]
        for n,m in self.roi_heads.named_modules():
            if n:
                self.roi_heads.roi_head_names.append(n)



class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation,index):
        """
        Args:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
            index (int): unique number of layer
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d[f"mask_fcn_index_{layer_idx}"] = nn.Conv2d(
                next_feature, layer_features, kernel_size=3, stride=1, padding=dilation, dilation=dilation
            )
            d[f"relu_index_{layer_idx}"] = nn.ReLU(inplace=True)
            next_feature = layer_features

        super().__init__(d)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes,index):
        # index is to index the segmentation Predictor
        super().__init__(
                OrderedDict(
                    [
                        (f"conv5_mask_{index}", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
                        (f"relu_{index}", nn.ReLU(inplace=True)),
                        ## additional masks 
                        #(f"conv6_mask_{index}", nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)),
                        #(f"relu_6_{index}", nn.ReLU(inplace=True)),
                    
                        (f"mask_fcn_logits_{index}", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
                    ]
                )
            )
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)


class KeypointRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers):
        d = []
        next_feature = in_channels
        for out_channels in layers:
            d.append(nn.Conv2d(next_feature, out_channels, 3, stride=1, padding=1))
            d.append(nn.ReLU(inplace=True))
            next_feature = out_channels
        super(KeypointRCNNHeads, self).__init__(*d)
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)


class KeypointRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super(KeypointRCNNPredictor, self).__init__()
        input_features = in_channels
        deconv_kernel = 4
        self.kps_score_lowres = nn.ConvTranspose2d(
            input_features,
            num_keypoints,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )
        nn.init.kaiming_normal_(
            self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        self.up_scale = 2
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        return torch.nn.functional.interpolate(
            x, scale_factor=float(self.up_scale), mode="bilinear", align_corners=False, recompute_scale_factor=False
        )



class ClassRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation,index):
        """
        Args:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
            index (int): unique number of layer
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d[f"class_fcn_index_{layer_idx}"] = nn.Conv2d(
                next_feature, layer_features, kernel_size=3, stride=1, padding=dilation, dilation=dilation
            )
            d[f"relu_index_{layer_idx}"] = nn.ReLU(inplace=True)
            next_feature = layer_features

        super().__init__(d)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)


class ClassRCNNPredictor(nn.Module):
    def __init__(self, in_channels, dim_reduced, num_classes,index):
        # index is to index the segmentation Predictor
        super(ClassRCNNPredictor,self).__init__()
        self.conv_class=nn.Conv2d(in_channels, dim_reduced, 3,2)
        self.conv_class_2=nn.Conv2d(dim_reduced, dim_reduced, 3,2)
        self.fc_class= nn.Linear(1024, num_classes)
        

        for name, param in self.named_parameters():
            if "weight" in name and "conv" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)

    def forward(self, x):
        x= F.relu(self.conv_class(x))
        x= F.relu(self.conv_class_2(x))
        # Global average pooling
        #x = x.mean(dim=(-2, -1)) 
        x = x.flatten(start_dim=1)
        #print(x.shape)
        x = self.fc_class(x)
        return x



model_urls = {
    # legacy model for BC reasons, see https://github.com/pytorch/vision/issues/1606
    'keypointrcnn_resnet50_fpn_coco_legacy':
        'https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-9f466800.pth',
    'keypointrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth',
    "maskrcnn_resnet50_fpn_coco": 
        "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
}


def vegnet_resnet50_fpn(pretrained=False, progress=True,num_classes=2, num_keypoints=2,pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):
    """
    Constructs a Keypoint R-CNN model with a ResNet-50-FPN backbone.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
        - keypoints (``FloatTensor[N, K, 3]``): the ``K`` keypoints location for each of the ``N`` instances, in the
          format ``[x, y, visibility]``, where ``visibility=0`` means that the keypoint is not visible.

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the keypoint loss.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detected instances:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each instance
        - scores (``Tensor[N]``): the scores or each instance
        - keypoints (``FloatTensor[N, K, 3]``): the locations of the predicted keypoints, in ``[x, y, v]`` format.

    For more details on the output, you may refer to :ref:`instance_seg_output`.

    Keypoint R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.

    Example::

        >>> model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "keypoint_rcnn.onnx", opset_version = 11)

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        num_keypoints (int): number of keypoints, default 2
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    trainable_backbone_layers = _validate_trainable_layers(pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = Vegnet(backbone, num_classes, num_keypoints=num_keypoints, **kwargs)
    if pretrained:
        #key='keypointrcnn_resnet50_fpn_coco'
        key="maskrcnn_resnet50_fpn_coco"
        print(f"***********Loading Coco Pretrained Model ****************")
        if pretrained == 'legacy':
            key += '_legacy'
        state_dict = load_state_dict_from_url(model_urls[key],
                                              progress=progress)
        # to load when sizes do not match
        current_model_dict = model.state_dict()
        new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), state_dict.values())}
        model.load_state_dict(new_state_dict,strict=False)
        overwrite_eps(model, 0.0)
    return model
