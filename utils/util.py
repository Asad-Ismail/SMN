import numpy as np
import cv2
import random
import warnings
import scipy
from scipy.linalg.basic import solve_circulant
import skimage
import skimage.transform
from distutils.version import LooseVersion
import torch
import math
import json

from torch.functional import Tensor

np.random.seed(42)

def load_points_dataset(f_p):
    """load keypoints(head/tail)from the text file
    Args:
        f_p ([str]): File path containing the head and tail points (x,y) of each fruit in the image. Each Image can have multiple fruits

    Returns:
        [dict]: Dictionary of file names as keys and corresponding fruit points as values
    """
    with open(f_p, "r") as f:
        all_lines = f.readlines()
    points = {}
    i = 0
    while i < len(all_lines):
        if i > len(all_lines):
            break
        line = all_lines[i].split(",")
        label = line[0]
        file = line[3]
        first_point = None
        second_point = None
        if label == "head":
            first_point = (int(line[1]), int(line[2]))
        elif label == "tail":
            second_point = (int(line[1]), int(line[2]))
        i += 1
        if i < len(all_lines):
            line2 = all_lines[i].split(",")
            if line2[3] == file:
                if line2[0] == "head":
                    first_point = (int(line2[1]), int(line2[2]))
                elif line2[0] == "tail":
                    second_point = (int(line2[1]), int(line2[2]))
                i += 1
        if file in points:
            # file already in dictionary append the list
            # print(f"Appending the file to existing one {file}")
            points[file].append([first_point, second_point])
        else:
            points[file] = [[first_point, second_point]]
    return points



def load_points_dataset_2(f_p,label_name=["head","tail"]):
    """load keypoints(head/tail)from the text file
    Args:
        f_p ([str]): File path containing the head and tail points (x,y) of each fruit in the image. Each Image can have multiple fruits

    Returns:
        [dict]: Dictionary of file names as keys and corresponding fruit points as values
    """
    with open(f_p, "r") as f:
        all_lines = f.readlines()
    points = {}
    i = 0
    while i < len(all_lines):
        if i > len(all_lines):
            break
        line = all_lines[i].split(",")
        label = line[0]
        file = line[3]
        first_point = None
        second_point = None
        if label == label_name[0]:
            first_point = (int(line[1]), int(line[2]))
        elif label == label_name[1]:
            second_point = (int(line[1]), int(line[2]))
        i += 1
        if i < len(all_lines):
            line2 = all_lines[i].split(",")
            if line2[3] == file:
                if line2[0] == label_name[0]:
                    first_point = (int(line2[1]), int(line2[2]))
                elif line2[0] == label_name[1]:
                    second_point = (int(line2[1]), int(line2[2]))
                i += 1
        if not first_point and not second_point:
            continue
        if file in points:
            # file already in dictionary append the list
            # print(f"Appending the file to existing one {file}")
            points[file].append([first_point, second_point])
        else:
            points[file] = [[first_point, second_point]]
    return points



def load_class_dataset(f_p,label_name=["rating","neck"]):
    """load keypoints(head/tail)from the text file
    Args:
        f_p ([str]): File path containing the head and tail points (x,y) of each fruit in the image. Each Image can have multiple fruits

    Returns:
        [dict]: Dictionary of file names as keys and corresponding fruit points as values
    """
    with open(f_p, "r") as f:
        all_lines = f.readlines()
    points = {}
    i = 0
    while i < len(all_lines):
        if i > len(all_lines):
            break
        line = all_lines[i].split(",")
        label = line[0]
        splitted_labels=label.split("_")
        file = line[3]
        coords= None
        if splitted_labels[0] in label_name:
            coords = (int(line[1]), int(line[2]))
        i += 1
        if coords is None:
            continue
        if file in points:
            # file already in dictionary append the list
            # print(f"Appending the file to existing one {file}")
            if splitted_labels[0] in points[file]:
                points[file][splitted_labels[0]].append([coords,int(splitted_labels[1])])
            else:
                points[file][splitted_labels[0]]=[[coords,int(splitted_labels[1])]]
        else:
            points[file]={splitted_labels[0]:[[coords, int(splitted_labels[1])]]}
    return points

def load_segmentation_dataset(f_p,label_names=None):
    """"
    Returns:
        [dict]: Dictionary of list with names 
    """
    data=load_json(f_p)
    cat_map={}
    for cat in data["categories"]:
        if cat["name"] in label_names:
            cat_map[cat['id']]=cat["name"] 
    image_map={}
    for cat in data["images"]:
        image_map[cat['id']]=cat["file_name"] 
    annos={}
    for d in data["annotations"]:
        tmp=[]
        seg=d["segmentation"][0]
        for i in range(0,len(seg)-1,2):
            tmp.append([seg[i],seg[i+1]]) 
        if image_map[d["image_id"]] not in annos:
            annos[image_map[d["image_id"]]]=[{"class_id":cat_map[d["category_id"]],"annotation":tmp}]
        else:
            annos[image_map[d["image_id"]]].append({"class_id":cat_map[d["category_id"]],"annotation":tmp})
    return annos


def load_backbone(filename):
    with open(filename) as f:
        back_annotation = json.load(f)
    return back_annotation

def load_json(filename):
    with open(filename) as f:
        annotation = json.load(f)
    return annotation


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    #print(np.max(mask))
    #print(np.min(mask))
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([x1,y1,x2,y2])
    return boxes.astype(np.int32)


def resize(
    image,
    output_shape,
    order=1,
    mode="constant",
    cval=0,
    clip=True,
    preserve_range=False,
    anti_aliasing=False,
    anti_aliasing_sigma=None,
):
    """A wrapper for Scikit-Image resize().
    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image,
            output_shape,
            order=order,
            mode=mode,
            cval=cval,
            clip=clip,
            preserve_range=preserve_range,
            anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma,
        )
    else:
        return skimage.transform.resize(
            image,
            output_shape,
            order=order,
            mode=mode,
            cval=cval,
            clip=clip,
            preserve_range=preserve_range,
        )


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.
    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.
    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)), preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode="constant", constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode="constant", constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y : y + min_dim, x : x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.
    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y : y + h, x : x + w]
    else:
        mask = np.pad(mask, padding, mode="constant", constant_values=0)
    return mask

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


def write_text(image,text,point=(0,0),color=(255,0,0)):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # fontScale
    fontScale = 1

    # Line thickness of 2 px
    thickness = 3

    # Using cv2.putText() method
    image = cv2.putText(image, text, point, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    return image
    

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
        vis_mask(vis_img,indicies=indicies,color=class_color[classes[i]])
        # Visualize bounding box
        cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), class_color[classes[i]], 2)
        # write class name
        write_text(vis_img,classes[i],((bbox[0], bbox[1])))
        # Visualize segm classes
        if "other_masks" in kwargs and "seg_labels" in kwargs:
            for j,k in enumerate(kwargs["other_masks"].keys(),start=1):
                if classes[i] in kwargs["seg_labels"][j]: 
                    m=kwargs["other_masks"][k][i][...,None].detach().numpy()
                    indicies = np.where(m >= 0.1)
                    vis_mask(vis_img,indicies=indicies,color=(0,0,255))
                    #other_mask_id+=1
                    
        ## Visualize keypoints
        if "kp_labels" in kwargs and keypoints is not None:
            if classes[i] in kwargs["kp_labels"]: 
                keypoint = np.int0(keypoints[i].detach().numpy())
                # Visualize Keypoints
                cv2.circle(vis_img, (keypoint[0][0], keypoint[0][1]), 1, (0, 255, 255), 20)
                write_text(vis_img,"Head",(keypoint[0][0]+off_x, keypoint[0][1]+off_y))
                cv2.circle(vis_img, (keypoint[1][0], keypoint[1][1]), 1, (0, 255, 255), 20)
                write_text(vis_img,"Tail",(keypoint[1][0]+off_x, keypoint[1][1]+off_y))
                    
        # visualize classification
        if "clas" in kwargs and "clas_labels" in kwargs:
                for j,k in enumerate(kwargs["clas"].keys(),start=0):
                    if classes[i] in kwargs["clas_labels"][j]:
                        cl=kwargs["clas"][k][i].cpu().item()
                        point=(bbox[0]+(bbox[2]-bbox[0])//2+j*off_x,bbox[1]+(bbox[3]-bbox[1])//2+j*off_y)
                        write_text(vis_img,f"{k}: {cl}",point=point,color=(0,0,255))
         
    vis_img=resize_images_cv(vis_img)
    #if"" kwargs["DEBUG"]:
    cv2.imwrite("ann_input.png",vis_img)
    cv2.imshow("Input and labels", vis_img)
    cv2.waitKey(0)



def resize_points(points, scale, window):
    # window: (y1, x1, y2, x2)
    scaled_points = []
    for i in range(len(points)):
        two_point = np.array(points[i])
        two_point = scale * two_point
        two_point[0][0] = two_point[0][0] + window[1]
        two_point[0][1] = two_point[0][1] + window[0]
        two_point[1][0] = two_point[1][0] + window[1]
        two_point[1][1] = two_point[1][1] + window[0]
        scaled_points.append(two_point)
    return np.int0(scaled_points)




def generate_anchors_tensor(scales, ratios, shape, feature_stride, anchor_stride,device="cpu"):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate(
        [box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1
    )
    boxes=torch.tensor(boxes,dtype=torch.float32)
    return boxes



def visualize_anchors(
    img,
    anchors,
    backbone_shapes,
    RPN_ANCHOR_RATIOS,
    RPN_ANCHOR_STRIDE,
    RPN_ANCHOR_SCALES,
):
    vis_img = img.copy()
    num_levels = len(backbone_shapes)
    anchors_per_cell = len(RPN_ANCHOR_RATIOS)
    print("Anchors Count: ", anchors.shape[0])
    print("Scales: ", RPN_ANCHOR_SCALES)
    print("ratios: ", RPN_ANCHOR_RATIOS)
    print("Anchors per Cell: ", anchors_per_cell)
    print("Levels: ", num_levels)
    anchors_per_level = []
    for l in range(num_levels):
        num_cells = backbone_shapes[l][0] * backbone_shapes[l][1]
        anchors_per_level.append(anchors_per_cell * num_cells // RPN_ANCHOR_STRIDE ** 2)
        print("Anchors in Level {}: {}".format(l, anchors_per_level[l]))

    for level in range(num_levels):
        colors = [[0, 255, 0]]
        # Compute the index of the anchors at the center of the image
        level_start = sum(
            anchors_per_level[:level]
        )  # sum of anchors of previous levels
        level_anchors = anchors[level_start : level_start + anchors_per_level[level]]
        print(
            "Level {}. Anchors: {:6}  Feature map Shape: {}".format(
                level, level_anchors.shape[0], backbone_shapes[level]
            )
        )
        center_cell = np.array(backbone_shapes[level]) // 2
        center_cell_index = center_cell[0] * backbone_shapes[level][1] + center_cell[1]
        level_center = center_cell_index * anchors_per_cell
        center_anchor = anchors_per_cell * (
            (center_cell[0] * backbone_shapes[level][1] / RPN_ANCHOR_STRIDE ** 2)
            + center_cell[1] / RPN_ANCHOR_STRIDE
        )
        level_center = int(center_anchor)

        # Draw anchors. Brightness show the order in the array, dark to bright.
        for i, rect in enumerate(
            level_anchors[level_center : level_center + anchors_per_cell]
        ):
            y1, x1, y2, x2 = rect
            cv2.rectangle(
                vis_img, (int(x1), int(y1)), (int(x2), int(y2)), colors[level], 2
            )

    cv2.imshow("Center Anchor Boxes", vis_img)
    cv2.waitKey(0)


def generate_pyramid_anchors_tensor(scales, ratios, feature_shapes, feature_strides, anchor_stride,device="cpu"):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.
    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors_tensor(scales[i], ratios, feature_shapes[i], feature_strides[i], anchor_stride,device=device))
    return torch.cat(anchors, axis=0).to(device=device)



def compute_iou_tensor(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = torch.maximum(box[0], boxes[:, 0])
    y2 = torch.minimum(box[2], boxes[:, 2])
    x1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[3], boxes[:, 3])
    intersection = torch.maximum(x2 - x1, torch.tensor(0)) * torch.maximum(y2 - y1, torch.tensor(0))
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps_tesnor(boxes1, boxes2,device="cpu"):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = torch.zeros((boxes1.shape[0], boxes2.shape[0])).to(device=device)
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou_tensor(box2, boxes1, area2[i], area1)
    return overlaps


def apply_box_deltas_tesnor(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
    deltas: [N, (dy, dx, log(dh), log(dw))]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= torch.exp(deltas[:, 2])
    width *= torch.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return torch.stack([y1, x1, y2, x2], axis=1)


def vis_anchors_refined_anchors_(img, anchors, refined_anchors):
    vis_img = img.copy()
    for i, rect in enumerate(anchors):
        y1, x1, y2, x2 = rect
        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        y1, x1, y2, x2 = refined_anchors[i]
        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        break
    cv2.imshow("Matched Anchor Boxes", vis_img)
    cv2.waitKey(0)


def build_rpn_targets_tensor(anchors, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.
    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = torch.zeros([anchors.shape[0]], dtype=torch.int32).to(device=config.DEVICE)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = torch.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4)).to(device=config.DEVICE)

    no_crowd_bool = torch.ones([anchors.shape[0]], dtype=torch.bool).to(device=config.DEVICE)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = compute_overlaps_tesnor(anchors, gt_boxes,config.DEVICE)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = torch.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[torch.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1

    # 2. Set an anchor for each GT box (regardless of IoU value).
    # If multiple anchors have the same IoU match all of them
    # original was argwhere
    # gt_iou_argmax = torch.where(torch.tensor(overlaps == torch.max(overlaps, axis=0)))[:, 0]
    a = torch.max(overlaps, axis=0)[0]
    gt_iou_argmax = torch.where(overlaps == a)[0]

    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = torch.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        unif = torch.ones(ids.shape[0]).to(device=config.DEVICE)
        idx = unif.multinomial(extra, replacement=False)
        ids = ids[idx]
        # ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = torch.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - torch.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        # ids = np.random.choice(ids, extra, replace=False)
        unif = torch.ones(ids.shape[0]).to(device=config.DEVICE)
        idx = unif.multinomial(extra, replacement=False)
        ids = ids[idx]
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = torch.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = torch.tensor(
            [
                (gt_center_y - a_center_y) / a_h,
                (gt_center_x - a_center_x) / a_w,
                torch.log(gt_h / a_h),
                torch.log(gt_w / a_w),
            ]
        )
        # Normalize
        #rpn_bbox[ix] /= torch.tensor(config.RPN_BBOX_STD_DEV, dtype=torch.float32).to(device=config.DEVICE)
        ix += 1

    return rpn_match, rpn_bbox


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
    assumed to be outside the box.
    """
    box = box.type(torch.float32)
    gt_box = gt_box.type(torch.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = torch.log(gt_height / height)
    dw = torch.log(gt_width / width)

    return torch.stack([dy, dx, dh, dw], axis=1)



def process_box(box, score, image_shape, min_size):
    """
    Clip boxes in the image size and remove boxes which are too small.
    """

    box[:, [0, 2]] = box[:, [0, 2]].clamp(0, image_shape[0])
    box[:, [1, 3]] = box[:, [1, 3]].clamp(0, image_shape[1])

    w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    keep = torch.where((w >= min_size) & (h >= min_size))[0]
    box, score = box[keep], score[keep]
    return box, score


def roi_align(
    features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio
):
    if torch.__version__ >= "1.5.0":
        return torch.ops.torchvision.roi_align(
            features,
            rois,
            spatial_scale,
            pooled_height,
            pooled_width,
            sampling_ratio,
            False,
        )
    else:
        return torch.ops.torchvision.roi_align(
            features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio
        )


class RoIAlign:
    """
    Performs Region of Interest (RoI) Align operator described in Mask R-CNN

    """

    def __init__(self, output_size, sampling_ratio):
        """
        Arguments:
            output_size (Tuple[int, int]): the size of the output after the cropping
                is performed, as (height, width)
            sampling_ratio (int): number of sampling points in the interpolation grid
                used to compute the output value of each pooled output bin. If > 0,
                then exactly sampling_ratio x sampling_ratio grid points are used. If
                <= 0, then an adaptive number of grid points are used (computed as
                ceil(roi_width / pooled_w), and likewise for height). Default: -1
        """

        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.spatial_scale = None

    def setup_scale(self, feature_shape, image_shape):
        if self.spatial_scale is not None:
            return

        possible_scales = []
        for s1, s2 in zip(feature_shape, image_shape):
            scale = 2 ** int(math.log2(s1 / s2))
            possible_scales.append(scale)
        assert possible_scales[0] == possible_scales[1]
        self.spatial_scale = possible_scales[0]

    def __call__(self, feature, proposal, image_shape):
        """
        Arguments:
            feature (Tensor[N, C, H, W])
            proposal (Tensor[K, 4])
            image_shape (Torch.Size([H, W]))

        Returns:
            output (Tensor[K, C, self.output_size[0], self.output_size[1]])

        """
        idx = proposal.new_full((proposal.shape[0], 1), 0)
        roi = torch.cat((idx, proposal), dim=1)

        self.setup_scale(feature.shape[-2:], image_shape)
        return roi_align(
            feature.to(roi),
            roi,
            self.spatial_scale,
            self.output_size[0],
            self.output_size[1],
            self.sampling_ratio,
        )




class Matcher:
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, iou):
        """
        Arguments:
            iou (Tensor[M, N]): containing the pairwise quality between 
            M ground-truth boxes and N predicted boxes.

        Returns:
            label (Tensor[N]): positive (1) or negative (0) label for each predicted box,
            -1 means ignoring this box.
            matched_idx (Tensor[N]): indices of gt box matched by each predicted box.
        """
        
        value, matched_idx = iou.max(dim=0)
        label = torch.full((iou.shape[1],), -1, dtype=torch.float, device=iou.device) 
        
        label[value >= self.high_threshold] = 1
        label[value < self.low_threshold] = 0
        
        if self.allow_low_quality_matches:
            highest_quality = iou.max(dim=1)[0]
            gt_pred_pairs = torch.where(iou == highest_quality[:, None])[1]
            label[gt_pred_pairs] = 1

        return label, matched_idx



class BalancedPositiveNegativeSampler:
    def __init__(self, num_samples, positive_fraction):
        self.num_samples = num_samples
        self.positive_fraction = positive_fraction

    def __call__(self, label):
        positive = torch.where(label == 1)[0]
        negative = torch.where(label == 0)[0]

        num_pos = int(self.num_samples * self.positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.num_samples - num_pos
        num_neg = min(negative.numel(), num_neg)

        pos_perm = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        neg_perm = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx = positive[pos_perm]
        neg_idx = negative[neg_perm]

        return pos_idx, neg_idx


def visualize_inference(in_img, inv_normalize,results):
    
    vis_img = in_img.clone()
    vis_img = (inv_normalize(vis_img).data.numpy() * 255).astype(np.uint8)
    vis_img =  np.ascontiguousarray(np.moveaxis(vis_img, 0, -1))
    boxes=np.int0(results["boxes"])
    labels=results["labels"]
    scores=results["scores"]
    print(f"Labels max {labels.max()}")
    print(f"label min {labels.min()}")
    print(f"scores max {scores.max()}")
    print(f"scores min {scores.min()}")
    for i in range(boxes.shape[0]):
        bbox=boxes[i]
        cv2.rectangle(vis_img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (255, 255, 0), 2)
    cv2.imshow("Results",vis_img)
    cv2.waitKey(0)
        
    