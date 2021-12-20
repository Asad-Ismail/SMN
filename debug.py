from torchvision.ops import roi_align
import torch

torch.backends.cudnn.benchmark=True


gt_masks=torch.load("gt_masks.pt")
rois=torch.load("rois.pt")
M=torch.load("m.pt")
res=roi_align(gt_masks, rois, (M, M), 1.)[:, 0]
print(res.shape)