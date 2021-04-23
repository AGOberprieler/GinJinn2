'''Module for merging sliding-window cropped datasets
'''

import os
import copy
import shutil
import json
import itertools
from collections import defaultdict
from operator import itemgetter
from tempfile import TemporaryDirectory
from typing import Iterable, List, Tuple, Callable
import cv2
import imantics
import numpy as np
from scipy.sparse.csgraph import connected_components
from ginjinn.simulation import coco_utils
from .dataset_cropping import crop_annotations, crop_image
from .utils import load_coco_ann, get_obj_anns, coco_seg_to_mask, bbox_from_mask

# source: https://stackoverflow.com/a/52604722
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_bname_from_fname(file_name: str) -> str:
    '''get_bname_from_fname

    Get original image name from sliding-window cropped file name.

    Parameters
    ----------
    file_name : str
        sliding-window cropped image file name

    Returns
    -------
    str
        original image name sans extension
    '''
    fname, _ = os.path.splitext(file_name)
    fname_split = fname.split('_')
    return '_'.join(fname_split[:-3])

def get_coords_from_fname(file_name: str) -> np.ndarray:
    '''get_coords_from_fname

    Get bounding-box coordinates on the original image of a sliding-window
    crop from file name.

    Parameters
    ----------
    file_name : str
        sliding-window cropped image file name

    Returns
    -------
    np.ndarray
        numpy array of bounding box coordinates on original image (x0, x1, y0, y1).
    '''
    fname, _ = os.path.splitext(file_name)
    fname_split = fname.split('_')
    y0, y1 = [int(coord) for coord in fname_split[-1].rsplit('-', 1)]
    x0, x1 = [int(coord) for coord in fname_split[-2].rsplit('-', 1)]

    return np.array([x0, x1, y0, y1])

def get_size_from_fname(file_name: str) -> np.ndarray:
    '''get_size_from_fname

    Get size of the original image from the file name of a sliding-window crop.

    Parameters
    ----------
    file_name : str
        Sliding-window cropped image file name

    Returns
    -------
    np.ndarray
        Image size of original image (width, height)
    '''
    fname, _ = os.path.splitext(file_name)
    fname_split = fname.split('_')
    width, height = [int(x) for x in fname_split[-3].split('x')]

    return np.array([width, height])

def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    '''xywh_to_xyxy

    Translate bounding box format from x0y0wh to x0y0x1y1

    Parameters
    ----------
    xywh : np.ndarray
        bbox in x0y0wh format

    Returns
    -------
    np.ndarray
        bbox in x0y0x1y1 format
    '''
    xyxy = np.array(xywh)
    xyxy[2:4] = xyxy[0:2] + xyxy[2:4]
    return xyxy

def xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    '''xywh_to_xyxy

    Translate bounding box format from x0y0x1y1 to x0y0wh

    Parameters
    ----------
    xyxy : np.ndarray
        bbox in x0y0x1y1 format

    Returns
    -------
    np.ndarray
        bbox in x0y0wh format
    '''
    xywh = np.array(xyxy)
    xywh[2:4] = xywh[2:4] - xywh[0:2]
    return xywh

def intersection_bboxes(
    a: np.ndarray,
    b: np.ndarray,
    intersection_type: str='iou'
) -> float:
    '''intersection_bboxes

    Calculate intersection of two bounding-boxes a and b.

    Parameters
    ----------
    a : np.ndarray
        Bounding-box in x0y0x1y1 format.
    b : np.ndarray
        Bounding-box in x0y0x1y1 format.
    intersection_type : str, optional
        Type or intersection to calculate, by default 'iou'.

        Possible types are:
        - "simple": absolute intersection area
        - "iou": intersection over union (intersection area / union area)
        - "ios": intersection over smaller (intersection area / smaller bbox area)

    Returns
    -------
    float
        Intersection

    Raises
    ------
    Exception
        Raised when an invalid intersection type is passed.
    '''
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])

    if (dx>=0) and (dy>=0):
        intersection = dx*dy
        if intersection_type == 'simple':
            return intersection
        elif intersection_type == 'iou':
            w = max(a[2], b[2]) - min(a[0], b[0])
            h = max(a[3], b[3]) - min(a[1], b[1])

            return intersection / (w * h)
        elif intersection_type == 'ios':
            w = min(a[2]-a[0], b[2]-b[0])
            h = min(a[3]-a[1], b[3]-b[1])

            return intersection / (w * h)
        else:
            msg = f'Invalid intersection_type argument "{intersection_type}". ' +\
                'Available arguments are "simple", "iou", "ios".'
            raise Exception(msg)

    return 0.0

def intersection_bboxes_coco(
    a: np.ndarray,
    b: np.ndarray,
    intersection_type: str='iou',
):
    '''intersection_bboxes_coco

    Calculate intersection of two COCO bounding-boxes a and b.

    Parameters
    ----------
    a : np.ndarray
        Bounding-box in x0y0wh format.
    b : np.ndarray
        Bounding-box in x0y0wh format.
    intersection_type : str, optional
        Type or intersection to calculate, by default 'iou'.

        Possible types are:
        - "simple": absolute intersection area
        - "iou": intersection over union (intersection area / union area)
        - "ios": intersection over smaller (intersection area / smaller bbox area)

    Returns
    -------
    float
        Intersection
    '''
    # a = [a[0], a[1], a[0]+a[2], a[1]+a[3]]
    # b = [b[0], b[1], b[0]+b[2], b[1]+b[3]]
    a = xywh_to_xyxy(a)
    b = xywh_to_xyxy(b)

    return intersection_bboxes(a, b, intersection_type=intersection_type)

def calc_intersection_matrix(
    bboxes: np.ndarray,
    intersection_type: str='iou'
) -> np.ndarray:
    '''calc_intersection_matrix

    Calculate pair-wise intersection matrix for bounding-boxes.

    Parameters
    ----------
    bboxes : np.ndarray
        n * 4 np.array of bounding boxes in x0y0x1y1 format.
        Each row represents a single bounding box.
    intersection_type : str, optional
        Type or intersection to calculate, by default 'iou'.

        Possible types are:
        - "simple": absolute intersection area
        - "iou": intersection over union (intersection area / union area)
        - "ios": intersection over smaller (intersection area / smaller bbox area)

    Returns
    -------
    np.ndarray
        n * n matrix of pairwise intersections.
    '''
    intersection_matrix = np.ones((bboxes.shape[0], bboxes.shape[0]))
    for i in range(0, bboxes.shape[0]-1):
        for j in range(i + 1, bboxes.shape[0]):
            intersection_matrix[i, j] = intersection_bboxes(
                bboxes[i], bboxes[j],
                intersection_type=intersection_type
            )
            intersection_matrix[j, i] = intersection_matrix[i, j]

    return intersection_matrix

def reconstruct_original_image(
    img_anns: List[dict],
    img_dir: str,
) -> np.ndarray:
    '''reconstruct_original_image

    Reconstruct the original image from cropped sub images.

    Parameters
    ----------
    img_anns : List[dict]
        List of COCO image annotations as dictionary.
    img_dir : str
        Directory containing the images corresponding to img_anns.

    Returns
    -------
    np.ndarray
        RGB image as numpy array (h * w * 3)
    '''
    orig_w, orig_h = get_size_from_fname(img_anns[0]['file_name'])
    orig_img = np.zeros((orig_h, orig_w, 3), dtype="uint8")

    for img_ann in img_anns:
        sub_img = cv2.imread(os.path.join(img_dir, img_ann['file_name']))

        xxyy = get_coords_from_fname(img_ann['file_name'])
        orig_img[xxyy[2]:xxyy[3], xxyy[0]:xxyy[1], :] = sub_img

    return orig_img

def merge_segmentations(
    img_anns: List[dict],
    obj_anns: List[dict]
):
    '''merge_segmentations

    Merge objects from sliding-window predictions such that the output annotations refer
    to the original image.

    Parameters
    ----------
    img_anns : list of dict
        List of COCO image annotations as dictionary.
    obj_anns : list of dict
        List of COCO object annotations as dictionary.

    Returns
    -------
    polygons : list of list of float
        Segmentation of merged object
    bbox : np.ndarray
        Corresponding bounding box (COCO format)
    '''
    dict_images = {ann["id"]: ann for ann in img_anns}

    orig_w, orig_h = get_size_from_fname(img_anns[0]['file_name'])
    mask = np.zeros((orig_h, orig_w), dtype=np.bool_)

    for obj_ann in obj_anns:
        img_ann = dict_images[obj_ann["image_id"]]
        sub_mask = coco_seg_to_mask(
            obj_ann["segmentation"],
            img_ann["width"],
            img_ann["height"]
        )
        xxyy = get_coords_from_fname(img_ann["file_name"])
        mask[xxyy[2]:xxyy[3], xxyy[0]:xxyy[1]] = np.logical_or(
            mask[xxyy[2]:xxyy[3], xxyy[0]:xxyy[1]],
            sub_mask
        )

    polygons = imantics.Mask(mask).polygons().segmentation
    polygons = [p for p in polygons if len(p) >= 6]
    bbox = bbox_from_mask(mask, fmt="xywh")
    return polygons, bbox

def reconstruct_annotations_on_original(
    img_anns: List[dict],
    obj_anns: List[dict],
) -> List[dict]:
    '''reconstruct_annotations_on_original

    Reconstruct object annotations in the coordinate system of
    the original, sliding-window croppped image.

    Parameters
    ----------
    img_anns : List[dict]
        List of COCO image annotations as dictionary.
    obj_anns : List[dict]
        List of COCO object annotations as dictionary.

    Returns
    -------
    List[dict]
        List of COCO image annotations in the coordinate system of
        the original image as dictionary.
    '''
    orig_obj_anns = []
    for img_ann in img_anns:
        xxyy = get_coords_from_fname(img_ann['file_name'])

        sub_obj_anns = [obj_ann for obj_ann in obj_anns if obj_ann['image_id'] == img_ann['id']]
        for obj_ann in sub_obj_anns:
            orig_obj_ann = copy.deepcopy(obj_ann)
            orig_obj_ann['bbox'][0] = obj_ann['bbox'][0] + xxyy[0]
            orig_obj_ann['bbox'][1] = obj_ann['bbox'][1] + xxyy[2]
            orig_obj_ann['image_id'] = img_anns[0]['id']

            orig_obj_anns.append(orig_obj_ann)

    return orig_obj_anns

def merge_bbox_annotations(
    obj_anns: List[dict],
    img_id: int,
    intersection_type: str='iou',
    intersection_th: float=0.6,
) -> List[dict]:
    '''merge_bbox_annotations

    Merge bounding-box annotations using single-linkage hierarchical
    clustering based on pairwise intersections.

    Parameters
    ----------
    obj_anns : List[dict]
        List of COCO object annotations as dictionary.
    img_id : int
        Image ID merged object annotations should refer to.
    intersection_type : str, optional
        Type or intersection to calculate, by default 'iou'.

        Possible types are:
        - "simple": absolute intersection area
        - "iou": intersection over union (intersection area / union area)
        - "ios": intersection over smaller (intersection area / smaller bbox area)
    intersection_th : float, optional
        Intersection threshold for the clustering cut-off, by default 0.6

    Returns
    -------
    List[dict]
        List of COCO object annotations as dictionary.
    '''
    from sklearn.cluster import AgglomerativeClustering

    if len(obj_anns) < 1:
        return []

    bboxes = xywh_to_xyxy([o_ann['bbox'] for o_ann in obj_anns])
    intersection_matrix = calc_intersection_matrix(bboxes, intersection_type=intersection_type)
    if intersection_matrix.shape[0] < 2:
        cl = np.array([0], dtype=int)
    else:
        ac = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1-intersection_th,
            affinity='precomputed',
            linkage='single'
        )
        cl = ac.fit_predict(1 - intersection_matrix)

    new_anns = []
    for cl_id in np.unique(cl):
        cl_idcs = np.argwhere(cl == cl_id).flatten()
        bboxes_cl = bboxes[cl == cl_id]
        bbox_merged = np.array([*bboxes_cl.min(0)[:2], *bboxes_cl.max(0)[2:]])

        new_ann = copy.deepcopy(obj_anns[cl_idcs[0]])
        bbox_xywh = xyxy_to_xywh(bbox_merged).flatten()
        new_ann['bbox'] = list(bbox_xywh)
        new_ann['area'] = float(bbox_xywh[2] * bbox_xywh[3])
        new_ann['image_id'] = int(img_id)
        new_anns.append(new_ann)

    return new_anns

# source: itertools recipes
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def merge_xyxy(boxes: Iterable):
    """merge_xyxy

    Merge bounding boxes in PascalVOC format.

    Parameters
    ----------
    boxes : [iterable of] iterable of int or float
        Bounding boxes in PascalVOC format, i.e., (x_min, y_min, x_max, y_max)

    Returns
    -------
    np.ndarray
        Bounding box enclosing the input boxes (PascalVOC format)
    """
    if np.isscalar(boxes[0]) and len(boxes) == 4:
        return np.array(boxes)
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    return np.array((x0, y0, x1, y1))

def merge_window_predictions_bbox(
    img_anns: List[dict],
    obj_anns: List[dict],
    img_dir: str,
    iou_threshold: float = 0,
    ios_threshold: float = 0,
    intersection_threshold: float = 0
) -> Tuple[np.ndarray, List[dict], List[dict]]:
    """merge_window_predictions_bbox

    Merge bounding boxes from sliding-window cropped COCO annotations. Two objects from different
    windows are only merged if their intersection satisfies the specified conditions or if further
    objects act as connectors. The three conditions are combined in this way:
    ((IoU >= iou_threshold) OR (IoS >= ios_threshold)) AND (intersection >= intersection_threshold)

    Parameters
    ----------
    img_anns : list of dict
        List of COCO image annotations as dictionary.
    obj_anns : list of dict
        List of COCO object annotations as dictionary.
    img_dir : str
        Directory containing the images, img_anns refer to.
    iou_threshold : float
        Min. intersection over union of two objects to be merged (0 = disabled).
        Note that the denominator (union) is only calculated within the overlapping region
        of two sliding windows.
    ios_threshold : float
        Min. intersection over smaller area (0 = disabled).
        The latter considers the object within the whole sliding window, not only within
        the window overlap.
    intersection_threshold : float
        Min. absolute intersection.

    Returns
    -------
    Tuple[np.ndarray, List[dict], List[dict]]
        Tuple containing the reconstructed original image as np.ndarray,
        a new COCO annotation dict for the reconstructed original image,
        and the merged annotations in the coordinate system of the original image,
        i.e.: (orig_img, orig_img_ann, merged_obj_anns)
    """
    # get position of each window on the original image
    coord_list = []
    for img_ann in img_anns:
        x0, x1, y0, y1 = get_coords_from_fname(img_ann["file_name"]).tolist()
        coord_list.append((x0, x1, y0, y1, img_ann["id"]))

    obj_map = {ann["id"]: i for i, ann in enumerate(obj_anns)}
    adj_mat = np.zeros((len(obj_anns), len(obj_anns)), dtype=np.bool_)

    # find horizontal overlaps
    coord_list.sort(key=itemgetter(2, 0))
    for y0, row in itertools.groupby(coord_list, key=itemgetter(2)):
        # process pairs of consecutive images
        for coords1, coords2 in pairwise(row):
            obj_anns_1 = [ann for ann in obj_anns if ann["image_id"] == coords1[4]]
            obj_anns_2 = [ann for ann in obj_anns if ann["image_id"] == coords2[4]]

            for ann1 in obj_anns_1:
                for ann2 in obj_anns_2:
                    if ann1["category_id"] != ann2["category_id"]:
                        continue

                    # bboxes -> coordinate system of orig. image
                    bbox1 = copy.deepcopy(ann1["bbox"])
                    orig_size1 = bbox1[2] * bbox1[3]
                    bbox1[0] += coords1[0]
                    bbox1[1] += coords1[2]
                    bbox1 = xywh_to_xyxy(bbox1)

                    bbox2 = copy.deepcopy(ann2["bbox"])
                    orig_size2 = bbox2[2] * bbox2[3]
                    bbox2[0] += coords2[0]
                    bbox2[1] += coords2[2]
                    bbox2 = xywh_to_xyxy(bbox2)

                    # clip x coordinates to window overlap
                    bbox1[0], bbox1[2] = np.clip((bbox1[0], bbox1[2]), coords2[0], coords1[1])
                    bbox2[0], bbox2[2] = np.clip((bbox2[0], bbox2[2]), coords2[0], coords1[1])

                    # evaluate object intersection
                    if bbox1[2] - bbox1[0] > 0 and bbox2[2] - bbox2[0] > 0:
                        IoU = intersection_bboxes(bbox1, bbox2, intersection_type="iou")
                        #IoS = intersection_bboxes(bbox1, bbox2, intersection_type="ios")
                        IoS = intersection_bboxes(bbox1, bbox2, intersection_type="simple") / min(orig_size1, orig_size2)
                        intersection = intersection_bboxes(bbox1, bbox2, intersection_type="simple")

                        # update adjacency matrix
                        if (
                            (
                                (iou_threshold and IoU >= iou_threshold)
                                or (ios_threshold and IoS >= ios_threshold)
                            )
                            and intersection >= intersection_threshold
                        ):
                            obj_ind_1 = obj_map[ann1["id"]]
                            obj_ind_2 = obj_map[ann2["id"]]
                            adj_mat[obj_ind_1, obj_ind_2] = 1
                            adj_mat[obj_ind_2, obj_ind_1] = 1 # unnecessary (in connected_components(), connection defaults to "weak")

    # find vertical overlaps
    coord_list.sort(key=itemgetter(0, 2))
    for x0, col in itertools.groupby(coord_list, key=itemgetter(0)):
        # process pairs of consecutive images
        for coords1, coords2 in pairwise(col):
            obj_anns_1 = [ann for ann in obj_anns if ann["image_id"] == coords1[4]]
            obj_anns_2 = [ann for ann in obj_anns if ann["image_id"] == coords2[4]]

            for ann1 in obj_anns_1:
                for ann2 in obj_anns_2:
                    if ann1["category_id"] != ann2["category_id"]:
                        continue

                    # bboxes -> coordinate system of orig. image
                    bbox1 = copy.deepcopy(ann1["bbox"])
                    orig_size1 = bbox1[2] * bbox1[3]
                    bbox1[0] += coords1[0]
                    bbox1[1] += coords1[2]
                    bbox1 = xywh_to_xyxy(bbox1)

                    bbox2 = copy.deepcopy(ann2["bbox"])
                    orig_size2 = bbox2[2] * bbox2[3]
                    bbox2[0] += coords2[0]
                    bbox2[1] += coords2[2]
                    bbox2 = xywh_to_xyxy(bbox2)

                    # clip y coordinates to window overlap
                    bbox1[1], bbox1[3] = np.clip((bbox1[1], bbox1[3]), coords2[2], coords1[3])
                    bbox2[1], bbox2[3] = np.clip((bbox2[1], bbox2[3]), coords2[2], coords1[3])

                    # evaluate object intersection
                    if bbox1[3] - bbox1[1] > 0 and bbox2[3] - bbox2[1] > 0:
                        IoU = intersection_bboxes(bbox1, bbox2, intersection_type="iou")
                        #IoS = intersection_bboxes(bbox1, bbox2, intersection_type="ios")
                        IoS = intersection_bboxes(bbox1, bbox2, intersection_type="simple") / min(orig_size1, orig_size2)
                        intersection = intersection_bboxes(bbox1, bbox2, intersection_type="simple")

                        # update adjacency matrix
                        if (
                            (
                                (iou_threshold and IoU >= iou_threshold)
                                or (ios_threshold and IoS >= ios_threshold)
                            )
                            and intersection >= intersection_threshold
                        ):
                            obj_ind_1 = obj_map[ann1["id"]]
                            obj_ind_2 = obj_map[ann2["id"]]
                            adj_mat[obj_ind_1, obj_ind_2] = 1
                            adj_mat[obj_ind_2, obj_ind_1] = 1 # unnecessary

    orig_img = reconstruct_original_image(img_anns, img_dir)
    orig_img_ann = coco_utils.build_coco_image(
        image_id = int(img_anns[0]["id"]),
        file_name = get_bname_from_fname(img_anns[0]["file_name"]) + ".jpg",
        width = int(orig_img.shape[1]),
        height = int(orig_img.shape[0]),
        coco_url = str(img_anns[0].get("coco_url", "")),
        date_captured = str(img_anns[0].get("date_captured", 0)),
        flickr_url = str(img_anns[0].get("flickr_url", "")),
    )

    # identify groups of objects to be merged
    n_comp, comp_labels = connected_components(adj_mat, directed=False)

    merged_obj_anns = []
    for i_comp in range(n_comp):
        inds_merge = np.where(comp_labels == i_comp)[0].tolist()
        obj_anns_merge = reconstruct_annotations_on_original(
            img_anns,
            [obj_anns[i] for i in inds_merge]
        )
        bbox = merge_xyxy([xywh_to_xyxy(ann["bbox"]) for ann in obj_anns_merge])
        bbox = xyxy_to_xywh(bbox)
        merged_obj_anns.append({
            "area": bbox[2] * bbox[3],
            "bbox": bbox,
            "segmentation": [],
            "iscrowd": 0,
            "image_id": orig_img_ann["id"],
            "id": obj_anns[inds_merge[0]]["id"],
            "category_id": obj_anns[inds_merge[0]]["category_id"]
        })

    return orig_img, orig_img_ann, merged_obj_anns

def merge_window_predictions_seg(
    img_anns: List[dict],
    obj_anns: List[dict],
    img_dir: str,
    iou_threshold: float = 0,
    ios_threshold: float = 0,
    intersection_threshold: float = 0
) -> Tuple[np.ndarray, List[dict], List[dict]]:
    """merge_window_predictions_seg

    Merge instance segmentations from sliding-window cropped COCO annotations. Two objects from
    different windows are only merged if their intersection satisfies the specified conditions or
    if further objects act as connectors. The three conditions are combined in this way:
    ((IoU >= iou_threshold) OR (IoS >= ios_threshold)) AND (intersection >= intersection_threshold)

    Parameters
    ----------
    img_anns : list of dict
        List of COCO image annotations as dictionary.
    obj_anns : list of dict
        List of COCO object annotations as dictionary.
    img_dir : str
        Directory containing the images, img_anns refer to.
    iou_threshold : float
        Min. intersection over union of two objects to be merged (0 = disabled).
        Note that the denominator (union) is only calculated within the overlapping region
        of two sliding windows.
    ios_threshold : float
        Min. intersection over smaller area (0 = disabled).
        The latter considers the object within the whole sliding window, not only within
        the window overlap.
    intersection_threshold : float
        Min. absolute intersection.

    Returns
    -------
    Tuple[np.ndarray, List[dict], List[dict]]
        Tuple containing the reconstructed original image as np.ndarray,
        a new COCO annotation dict for the reconstructed original image,
        and the merged annotations in the coordinate system of the original image,
        i.e.: (orig_img, orig_img_ann, merged_obj_anns)
    """
    # get position of each window on the original image
    coord_list = []
    for img_ann in img_anns:
        x0, x1, y0, y1 = get_coords_from_fname(img_ann["file_name"]).tolist()
        coord_list.append((x0, x1, y0, y1, img_ann["id"]))

    obj_map = {ann["id"]: i for i, ann in enumerate(obj_anns)}
    adj_mat = np.zeros((len(obj_anns), len(obj_anns)), dtype=np.bool_)

    # find horizontal overlaps
    coord_list.sort(key=itemgetter(2, 0))
    for y0, row in itertools.groupby(coord_list, key=itemgetter(2)):
        # process pairs of consecutive images
        for coords1, coords2 in pairwise(row):
            obj_anns_1 = [ann for ann in obj_anns if ann["image_id"] == coords1[4]]
            obj_anns_2 = [ann for ann in obj_anns if ann["image_id"] == coords2[4]]

            for ann1 in obj_anns_1:
                for ann2 in obj_anns_2:
                    if ann1["category_id"] != ann2["category_id"]:
                        continue

                    mask1 = coco_seg_to_mask(
                        ann1["segmentation"],
                        coords1[1] - coords1[0],
                        coords1[3] - coords1[2]
                    )
                    mask2 = coco_seg_to_mask(
                        ann2["segmentation"],
                        coords2[1] - coords2[0],
                        coords2[3] - coords2[2]
                    )
                    orig_size1 = mask1.sum()
                    orig_size2 = mask2.sum()

                    # extract overlap
                    mask1 = mask1[:, coords2[0] - coords1[0]:]
                    mask2 = mask2[:, :coords1[1] - coords2[0]]

                    # evaluate object intersection
                    if mask1.sum() > 0 and mask2.sum() > 0:
                        IoU = np.logical_and(mask1, mask2).sum() / np.logical_or(mask1, mask2).sum()
                        #IoS = np.logical_and(mask1, mask2).sum() / min(mask1.sum(), mask2.sum())
                        IoS = np.logical_and(mask1, mask2).sum() / min(orig_size1, orig_size2)
                        intersection = np.logical_and(mask1, mask2).sum()

                        # update adjacency matrix
                        if (
                            (
                                (iou_threshold and IoU >= iou_threshold)
                                or (ios_threshold and IoS >= ios_threshold)
                            )
                            and intersection >= intersection_threshold
                        ):
                            obj_ind_1 = obj_map[ann1["id"]]
                            obj_ind_2 = obj_map[ann2["id"]]
                            adj_mat[obj_ind_1, obj_ind_2] = 1
                            adj_mat[obj_ind_2, obj_ind_1] = 1

    # find vertical overlaps
    coord_list.sort(key=itemgetter(0, 2))
    for x0, col in itertools.groupby(coord_list, key=itemgetter(0)):
        # process pairs of consecutive images
        for coords1, coords2 in pairwise(col):
            obj_anns_1 = [ann for ann in obj_anns if ann["image_id"] == coords1[4]]
            obj_anns_2 = [ann for ann in obj_anns if ann["image_id"] == coords2[4]]

            for ann1 in obj_anns_1:
                for ann2 in obj_anns_2:
                    if ann1["category_id"] != ann2["category_id"]:
                        continue

                    mask1 = coco_seg_to_mask(
                        ann1["segmentation"],
                        coords1[1] - coords1[0],
                        coords1[3] - coords1[2]
                    )
                    mask2 = coco_seg_to_mask(
                        ann2["segmentation"],
                        coords2[1] - coords2[0],
                        coords2[3] - coords2[2]
                    )
                    orig_size1 = mask1.sum()
                    orig_size2 = mask2.sum()

                    # extract overlap
                    mask1 = mask1[coords2[2] - coords1[2]:, :]
                    mask2 = mask2[:coords1[3] - coords2[2], :]

                    # evaluate object intersection
                    if mask1.sum() > 0 and mask2.sum() > 0:
                        IoU = np.logical_and(mask1, mask2).sum() / np.logical_or(mask1, mask2).sum()
                        #IoS = np.logical_and(mask1, mask2).sum() / min(mask1.sum(), mask2.sum())
                        IoS = np.logical_and(mask1, mask2).sum() / min(orig_size1, orig_size2)
                        intersection = np.logical_and(mask1, mask2).sum()

                        # update adjacency matrix
                        if (
                            (
                                (iou_threshold and IoU >= iou_threshold)
                                or (ios_threshold and IoS >= ios_threshold)
                            )
                            and intersection >= intersection_threshold
                        ):
                            obj_ind_1 = obj_map[ann1["id"]]
                            obj_ind_2 = obj_map[ann2["id"]]
                            adj_mat[obj_ind_1, obj_ind_2] = 1
                            adj_mat[obj_ind_2, obj_ind_1] = 1 # unnecessary

    orig_img = reconstruct_original_image(img_anns, img_dir)
    orig_img_ann = coco_utils.build_coco_image(
        image_id = int(img_anns[0]["id"]),
        file_name = get_bname_from_fname(img_anns[0]["file_name"]) + ".jpg",
        width = int(orig_img.shape[1]),
        height = int(orig_img.shape[0]),
        coco_url = str(img_anns[0].get("coco_url", "")),
        date_captured = str(img_anns[0].get("date_captured", 0)),
        flickr_url = str(img_anns[0].get("flickr_url", "")),
    )

    # identify groups of objects to be merged
    n_comp, comp_labels = connected_components(adj_mat, directed=False)

    merged_obj_anns = []
    for i_comp in range(n_comp):
        inds_merge = np.where(comp_labels == i_comp)[0].tolist()
        polygons, bbox = merge_segmentations(img_anns, [obj_anns[i] for i in inds_merge])
        merged_obj_anns.append({
            "area": bbox[2] * bbox[3],
            "bbox": bbox.tolist(),
            "segmentation": polygons,
            "iscrowd": 0,
            "image_id": orig_img_ann["id"],
            "id": obj_anns[inds_merge[0]]["id"],
            "category_id": obj_anns[inds_merge[0]]["category_id"]
        })

    return orig_img, orig_img_ann, merged_obj_anns

def merge_sliding_window_predictions(
    img_dir: str,
    ann_path: str,
    out_dir: str,
    task: str,
    iou_threshold: float = 0.5,
    ios_threshold: float = 0,
    intersection_threshold: float = 100,
    on_out_dir_exists: Callable[[str], bool] = lambda out_dir: True,
    on_img_out_dir_exists: Callable[[str], bool] = lambda img_out_dir: True,
):
    '''merge_sliding_window_predictions

    Merge predicted annotations that were based on sliding-window cropped images.

    Parameters
    ----------
    img_dir : str
        Path to directory containing the images that the prediction was made for.
    ann_path : str
        Path to predicted annotation.
    out_dir : str
        Path to directory that the merged annotations and images should be written to.
    task : str
        Either 'bbox-detection' or 'instance-segmentation'.
    iou_threshold : float
        Min. intersection over union of two objects to be merged.
        Note that the denominator (union) is only calculated within the overlapping region
        of two sliding windows.
    ios_threshold : float
        Min. intersection over smaller area.
        The latter considers the object within the whole sliding window, not only within
        the window overlap.
    intersection_threshold : float
        Min. absolute intersection.
    on_out_dir_exists : Callable[[str], bool], optional
        A function that decides what should happen if out_dir already exists.
        If true is returned, out_dir will be overwritten.
        By default, out_dir will be overwritten.
    on_img_out_dir_exists : Callable[[str], bool], optional
        A function that decides what should happen if img_out_dir already exists.
        If true is returned, img_out_dir will be overwritten.
        By default, img_out_dir will be overwritten.
    '''
    ann = load_coco_ann(ann_path)

    bname_to_img_anns = defaultdict(list)
    for img_ann in ann["images"]:
        bname = get_bname_from_fname(img_ann["file_name"])
        bname_to_img_anns[bname].append(img_ann)

    img_id_to_obj_anns = defaultdict(list)
    for obj_ann in ann["annotations"]:
        img_id_to_obj_anns[obj_ann["image_id"]].append(obj_ann)

    if os.path.exists(out_dir):
        should_remove = on_out_dir_exists(out_dir)
        if should_remove:
            shutil.rmtree(out_dir)
            os.mkdir(out_dir)
    else:
        os.mkdir(out_dir)

    img_out_dir = os.path.join(out_dir, 'images')
    if os.path.exists(img_out_dir):
        should_remove = on_img_out_dir_exists(img_out_dir)
        if should_remove:
            shutil.rmtree(img_out_dir)
            os.mkdir(img_out_dir)
    else:
        os.mkdir(img_out_dir)

    ann_out_file = os.path.join(out_dir, 'annotations.json')

    new_img_anns = []
    new_obj_anns = []

    with TemporaryDirectory() as tmp_dir:
        for bname in bname_to_img_anns:
            img_anns = [] # without padding
            obj_anns = []

            for img_ann in bname_to_img_anns[bname]:
                img_ann_new = copy.deepcopy(img_ann)
                obj_anns_win = copy.deepcopy(img_id_to_obj_anns[img_ann["id"]])
                width_orig, height_orig = get_size_from_fname(img_ann['file_name'])
                x0, x1, y0, y1 = get_coords_from_fname(img_ann['file_name'])

                # remove padding
                if any((x0 < 0, y0 < 0, x1 > width_orig, y1 > height_orig)) :
                    X0 = -min(0, x0)
                    Y0 = -min(0, y0)
                    X1 = x1 - x0 - max(0, x1 - width_orig)
                    Y1 = y1 - y0 - max(0, y1 - height_orig)

                    _, obj_anns_win = crop_annotations(
                        annotations = obj_anns_win,
                        img_width = img_ann["width"],
                        img_height = img_ann["height"],
                        cropping_range = (X0, X1, Y0, Y1),
                        start_id = None,
                        task = task,
                        keep_incomplete = True
                    )
                    img_name_new = '{}_{}x{}_{}-{}_{}-{}.jpg'.format(
                        bname,
                        width_orig,
                        height_orig,
                        *np.clip((x0, x1), 0, width_orig).tolist(),
                        *np.clip((y0, y1), 0, height_orig).tolist()
                    )
                    img_padded = cv2.imread(os.path.join(img_dir, img_ann['file_name']))
                    cv2.imwrite(
                        os.path.join(tmp_dir, img_name_new),
                        crop_image(img_padded, (X0, X1, Y0, Y1))
                    )
                    img_ann_new["file_name"] = img_name_new
                    img_ann_new["width"] = X1 - X0
                    img_ann_new["height"] = Y1 - Y0
                else:
                    os.symlink(
                        os.path.abspath(os.path.join(img_dir, img_ann["file_name"])),
                        os.path.join(tmp_dir, img_ann["file_name"])
                    )

                img_anns.append(img_ann_new)
                obj_anns.extend(obj_anns_win)

            if task == "bbox-detection":
                orig_img, orig_img_ann, merged_obj_anns = merge_window_predictions_bbox(
                    img_anns,
                    obj_anns,
                    tmp_dir,
                    iou_threshold = iou_threshold,
                    ios_threshold = ios_threshold,
                    intersection_threshold = intersection_threshold
                )
            elif task == "instance-segmentation":
                orig_img, orig_img_ann, merged_obj_anns = merge_window_predictions_seg(
                    img_anns,
                    obj_anns,
                    tmp_dir,
                    iou_threshold = iou_threshold,
                    ios_threshold = ios_threshold,
                    intersection_threshold = intersection_threshold
                )

            img_f = os.path.join(img_out_dir, f'{bname}.jpg')
            cv2.imwrite(img_f, orig_img)

            new_img_anns.append(orig_img_ann)
            new_obj_anns.extend(merged_obj_anns)

    new_ann = coco_utils.build_coco_dataset(
        annotations=new_obj_anns,
        images=new_img_anns,
        categories=ann['categories'],
        licenses=ann['licenses'],
        info=ann['info']
    )

    with open(ann_out_file, 'w') as ann_f:
        json.dump(new_ann, ann_f, indent=2, cls=NumpyEncoder)
