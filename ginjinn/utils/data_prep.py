''' Module containing functionality for data set preprocessing.
'''

import glob
import json
import os
import shutil
from typing import List
import xml.etree.ElementTree as ET
import datetime
import cv2
import imantics
import pandas as pd
from .utils import bbox_from_mask, coco_seg_to_mask, confirmation

def flatten_coco(
    ann_file: str,
    img_root_dir: str,
    out_dir: str,
    sep: str = '~',
    custom_id: bool = False,
    annotated_only: bool = False,
    link_images: bool = True
):
    '''flatten_coco

    Flatten COCO data set in such a way that all images are located in the same
    directory.

    Parameters
    ----------
    ann_file : str
        Path to annotation (JSON) file.
    img_root_dir : str
        Root dir of the image directory. For COCO, this directory is often called
        "images".
    out_dir : str
        Output directory.
    sep : str, optional
        Seperator for path flattening, by default '~'
    custom_id : bool, optional
        Whether the new image name should be replaced with a custom id, by default False
    annotated_only : bool, optional
        Whether only annotated images should be kept in the data set.
    link_images : bool, optional
        If true, images won't be copied but hard-linked instead.
    '''
    with open(ann_file) as ann_f:
        annotations = json.load(ann_f)

    out_img_dir = os.path.join(out_dir, 'images')
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)

    if annotated_only:
        img_ids = {ann['image_id'] for ann in annotations['annotations']}
        annotations['images'] = [ann for ann in annotations['images'] if ann['id'] in img_ids]

    id_map = {}

    for i, img_ann in enumerate(annotations['images']):
        file_name = img_ann['file_name']

        if custom_id:
            new_file_name = f'{i}.jpg'
            id_map[i] = file_name
        else:
            new_file_name = file_name.replace('/', sep)

        img_ann['file_name'] = new_file_name
        if link_images:
            os.link(
                os.path.join(img_root_dir, file_name),
                os.path.join(out_img_dir, new_file_name)
            )
        else:
            shutil.copy(
                os.path.join(img_root_dir, file_name),
                os.path.join(out_img_dir, new_file_name)
            )

    out_ann_file = os.path.join(out_dir, 'annotations.json')
    with open(out_ann_file, 'w') as ann_f:
        json.dump(annotations, ann_f, indent=2)

    if custom_id:
        id_map_file = os.path.join(out_dir, 'id_map.csv')
        id_map_df = pd.DataFrame.from_records(
            list(id_map.items()),
            columns=['id', 'original_path']
        )
        id_map_df.to_csv(id_map_file, index=False)


def filter_categories_coco(
    ann_file: str,
    img_dir: str,
    out_dir: str,
    link_images: bool = True,
    drop: List[str] = None,
    keep: List[str] = None
):
    """filter_categories_coco

    This function allows to filter object annotations in a COCO dataset according to their
    assigned category. Therefore, either ``drop`` or ``keep`` has to be specified.
    If img_dir is specified, a new, filtered image directory is created as well.
    Note that the original IDs referring to annotations, categories and images are preserved,
    and may be non-contiguous in the output dataset.

    Parameters
    ----------
    ann_file : str
        Path to annotation (JSON) file
    img_dir: str
        Directory containing image files. If None, no filtered image directory is written.
    out_dir: str
        Output directory
    link_images : bool
        If true, images won't be copied but hard-linked instead.
    drop : list of str
        If specified, these categories are removed from the dataset.
    keep : list of str
        If specified, only these categories are preserved.

    Raises
    ------
    ValueError
        Raised for unsupported parameter settings.
    """
    if bool(drop) + bool(keep) == 0:
        raise ValueError(
            "Either ``drop`` or ``keep`` has to be specified as non-empty list."
        )

    if os.path.exists(out_dir):
        if confirmation(
            out_dir + ' already exists.\nDo you want do overwrite it?'
        ):
            shutil.rmtree(out_dir)
        else:
            return

    os.makedirs(out_dir)
    if img_dir:
        os.makedirs(os.path.join(out_dir, "images"))

    with open(ann_file, "rb") as f:
        anno = json.load(f)

    if keep:
        categories_left = {cat["id"] for cat in anno.get("categories") if cat["name"] in keep}
    else:
        categories_left = {cat["id"] for cat in anno.get("categories") if cat["name"] not in drop}

    anno["annotations"] = [ann for ann in anno.get("annotations") if ann["category_id"] in categories_left]
    anno["categories"] = [cat for cat in anno.get("categories") if cat["id"] in categories_left]

    images_left = {ann["image_id"] for ann in anno.get("annotations")}
    anno["images"] = [img for img in anno.get("images") if img["id"] in images_left]

    anno["info"] = {
        "contributor" : "",
        "date_created" : datetime.datetime.now().strftime("%Y/%m/%d"),
        "description" : "",
        "version" : "",
        "url" : "",
        "year" : ""
    }

    # write COCO json file
    with open(os.path.join(out_dir, "annotations.json"), 'w') as json_file:
        json.dump(
            anno,
            json_file,
            indent = 2,
            sort_keys = True
        )

    if img_dir:
        for img in anno.get("images"):
            if img["id"] in images_left:
                img_name = os.path.split(img["file_name"])[1]
                img_path = os.path.join(img_dir, img_name)

                if link_images:
                    os.link(img_path, os.path.join(out_dir, "images", img_name))
                else:
                    shutil.copy(img_path, os.path.join(out_dir, "images", img_name))


def filter_categories_pvoc(
    ann_dir: str,
    img_dir: str,
    out_dir: str,
    link_images: bool = True,
    drop: List[str] = None,
    keep: List[str] = None
):
    """filter_categories_pvoc

    This function allows to filter object annotations in a PascalVOC dataset according to their
    assigned category. Therefore, either ``drop`` or ``keep`` has to be specified.
    If img_dir is specified, a new, filtered image directory is created as well.

    Parameters
    ----------
    ann_dir: str
        Directory containing annotation files (XML)
    img_dir: str
        Directory containing image files. If None, no filtered image directory is written.
    out_dir: str
        Output directory
    link_images : bool
        If true, images won't be copied but hard-linked instead.
    drop : list of str
        If specified, these categories are removed from the dataset.
    keep : list of str
        If specified, only these categories are preserved.

    Raises
    ------
    ValueError
        Raised for unsupported parameter settings.
    """
    if bool(drop) + bool(keep) == 0:
        raise ValueError(
            "Either ``drop`` or ``keep`` has to be specified as non-empty list."
        )

    if os.path.exists(out_dir):
        if confirmation(
            out_dir + ' already exists.\nDo you want do overwrite it?'
        ):
            shutil.rmtree(out_dir)
        else:
            return

    os.makedirs(os.path.join(out_dir, "annotations"))
    if img_dir:
        os.makedirs(os.path.join(out_dir, "images"))

    for ann_path in glob.glob(os.path.join(ann_dir, "*.xml")):
        tree = ET.parse(ann_path)
        root = tree.getroot()

        for child in root.findall("object"):
            if keep and child.findtext("name") not in keep:
                root.remove(child)
            if drop and child.findtext("name") in drop:
                root.remove(child)

        if root.findall("object"):
            tree.write(
                os.path.join(
                    out_dir,
                    "annotations",
                    os.path.split(ann_path)[1]
                )
            )
            if img_dir:
                img_name = os.path.split(root.findtext("filename"))[1]
                img_path = os.path.join(img_dir, img_name)

                if link_images:
                    os.link(img_path, os.path.join(out_dir, "images", img_name))
                else:
                    shutil.copy(img_path, os.path.join(out_dir, "images", img_name))

def filter_objects_by_size(
    ann_file: str,
    out_file: str,
    task: str,
    min_width: int = 0,
    min_height: int = 0,
    min_area: int = 0,
    min_fragment_area: int = 0
):
    """filter_objects_by_size

    Filter (sub)objects from a COCO annotation file by size.

    Parameters
    ----------
    ann_file : str
        Path to annotation (JSON) file
    out_file : str
        Output file name
    task : str
        Either 'bbox-detection' or 'instance-segmentation', determines whether filter criteria
        are applied to bounding boxes or segmentations.
    min_width : int
        Min. object width (pixels)
    min_height : int
        Min. object height (pixels)
    min_area : int
        Min. object area (pixels)
    min_fragment_area : int
        Min. area of object parts (pixels).
        If a segmentation instance consists of multiple disjunct parts, this option allows
        to remove small subobjects without discarding the whole object.

    Raises
    ------
    ValueError
        Raised for invalid filter settings.
    """
    if (
        max(min_width, min_height, min_area, min_fragment_area) < 1
        or min(min_width, min_height, min_area, min_fragment_area) < 0
    ):
        raise ValueError(
            "\"min_width\", \"min_height\", \"min_area\", and \"min_fragment_area\" have to be "\
            "non-negative integers, at least one of which must be non-zero."
        )

    with open(ann_file, "rb") as f:
        ann = json.load(f)

    dict_images = {img_ann["id"]: img_ann for img_ann in ann.get("images")}
    annotations_filtered = []

    for annotation in ann.get("annotations"):
        img_ann = dict_images[annotation["image_id"]]
        img_width = img_ann["width"]
        img_height = img_ann["height"]

        if task == "bbox-detection":
            obj_width, obj_height = annotation["bbox"][2:]
            if (
                obj_width >= min_width
                and obj_height >= min_height
                and obj_width * obj_height >= min_area
            ):
                annotations_filtered.append(annotation)

        elif task == "instance-segmentation":
            seg = annotation.get("segmentation")
            if seg:
                seg_mask = coco_seg_to_mask(seg, img_width, img_height)
                if seg_mask.sum() < min_area:
                    continue
            else:
                continue

            # calculate width and height from segmentation
            *_, obj_width, obj_height = bbox_from_mask(seg_mask, fmt="xywh").tolist()
            if obj_width < min_width or obj_height < min_height:
                continue

            if min_fragment_area > 0:
                n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    seg_mask.astype("uint8"),
                    connectivity = 8
                )
                sizes = stats[-1]
                # 0th element: background
                for i in range(1, n_labels):
                    if sizes[i] < min_fragment_area:
                        seg_mask[labels == i] = False
                annotation["segmentation"] = imantics.Mask(seg_mask).polygons().segmentation
                annotation["iscrowd"] = 0

            if seg_mask.sum() > 0:
                annotations_filtered.append(annotation)

    ann["annotations"] = annotations_filtered
    ann["info"] = {
        "contributor" : "",
        "date_created" : datetime.datetime.now().strftime("%Y/%m/%d"),
        "description" : "",
        "version" : "",
        "url" : "",
        "year" : ""
    }

    # write COCO json file
    with open(out_file, 'w') as json_file:
        json.dump(
            ann,
            json_file,
            indent = 2,
            sort_keys = True
        )
