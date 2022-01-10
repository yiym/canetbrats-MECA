import medpy.metric.binary as medpyMetrics
import numpy as np
from medpy.metric import binary
# from scipy import ndimage
# import GeodisTK

def softDice(pred, target, smoothing=1, nonSquared=False):
    intersection = (pred * target).sum(dim=(1, 2, 3))
    if nonSquared:
        union = (pred).sum() + (target).sum()
    else:
        union = (pred * pred).sum(dim=(1, 2, 3)) + (target * target).sum(dim=(1, 2, 3))
    dice = (2 * intersection + smoothing) / (union + smoothing)

    #fix nans
    dice[dice != dice] = dice.new_tensor([1.0])

    return dice.mean()

def dice(pred, target):
    predBin = (pred > 0.5).float()
    return softDice(predBin, target, 0, True).item()

def diceLoss(pred, target, nonSquared=False):
    return 1 - softDice(pred, target, nonSquared=nonSquared)

def bratsDiceLoss(outputs, labels, nonSquared=False):

    #bring outputs into correct shape
    wt, tc, et = outputs.chunk(3, dim=1)
    s = wt.shape
    wt = wt.view(s[0], s[2], s[3], s[4])
    tc = tc.view(s[0], s[2], s[3], s[4])
    et = et.view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask, tcMask, etMask = labels.chunk(3, dim=1)
    s = wtMask.shape
    wtMask = wtMask.view(s[0], s[2], s[3], s[4])
    tcMask = tcMask.view(s[0], s[2], s[3], s[4])
    etMask = etMask.view(s[0], s[2], s[3], s[4])

    #calculate losses
    wtLoss = diceLoss(wt, wtMask, nonSquared=nonSquared)
    tcLoss = diceLoss(tc, tcMask, nonSquared=nonSquared)
    etLoss = diceLoss(et, etMask, nonSquared=nonSquared)
    return (wtLoss + tcLoss + etLoss) / 5

def bratsDiceLossOriginal5(outputs, labels, nonSquared=False):
    outputList = list(outputs.chunk(5, dim=1))
    labelsList = list(labels.chunk(5, dim=1))
    totalLoss = 0
    for pred, target in zip(outputList, labelsList):
        totalLoss = totalLoss + diceLoss(pred, target, nonSquared=nonSquared)
    return totalLoss


def sensitivity(pred, target):
    predBin = (pred > 0.5).float()
    intersection = (predBin * target).sum()
    allPositive = target.sum()

    # special case for zero positives
    if allPositive == 0:
        return 1.0
    return (intersection / allPositive).item()

def specificity(pred, target):
    predBinInv = (pred <= 0.5).float()
    targetInv = (target == 0).float()
    intersection = (predBinInv * targetInv).sum()
    allNegative = targetInv.sum()
    return (intersection / allNegative).item()

def getHd95(pred, target):
    pred = pred.cpu().numpy()
    # print("pred:",pred)
    target = target.cpu().numpy()
    # print("target:",target)
    if np.count_nonzero(pred) > 0 and np.count_nonzero(target) > 0:

        surDist1 = medpyMetrics.__surface_distances(pred, target)
        surDist2 = medpyMetrics.__surface_distances(target, pred)
        hd95 = np.percentile(np.hstack((surDist1, surDist2)), 95)

        # hd = binary.hd(pred, target, voxelspacing=voxelspacing)
        # hd95 = binary.hd95(pred, target)
        # print(hd95)

        return hd95
    else:
        # Edge cases that medpy cannot handle
        return -1

# Hausdorff and ASSD evaluation
# def get_edge_points(img):
#     """
#     get edge points of a binary segmentation result
#     """
#     dim = len(img.shape)
#     if (dim==2):
#         strt = ndimage.generate_binary_structure(2, 1)
#     else:
#         strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
#     ero = ndimage.morphology.binary_erosion(img, strt)
#     edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
#     return edge
#
# def getHd95(s, g, spacing=None):
#     """
#     get the hausdorff distance between a binary segmentation and the ground truth
#     inputs:
#         s: a 3D or 2D binary image for segmentation
#         g: a 2D or 2D binary image for ground truth
#         spacing: a list for image spacing, length should be 3 or 2
#     """
#     s_edge = get_edge_points(s)
#     g_edge = get_edge_points(g)
#     image_dim = len(s.shape)
#     assert (image_dim,len(g.shape))
#     if (spacing==None):
#         spacing = [1.0] * image_dim
#     else:
#         assert (image_dim ,len(spacing))
#     img = np.zeros_like(s)
#     if (image_dim==2):
#         s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
#         g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
#     elif (image_dim==3):
#         s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
#         g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)
#
#     dist_list1 = s_dis[g_edge > 0]
#     dist_list1 = sorted(dist_list1)
#     dist1 = dist_list1[int(len(dist_list1) * 0.95)]
#     dist_list2 = g_dis[s_edge > 0]
#     dist_list2 = sorted(dist_list2)
#     dist2 = dist_list2[int(len(dist_list2) * 0.95)]
#     return max(dist1, dist2)

def getWTMask(labels):
    return (labels != 0).float()

def getTCMask(labels):
    return ((labels != 0) * (labels != 2)).float() #We use multiplication as AND

def getETMask(labels):
    return (labels == 4).float()
