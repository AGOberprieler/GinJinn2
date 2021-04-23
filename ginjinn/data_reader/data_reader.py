"""
Convert input to Detectron2's default dictionary format.
"""

import glob
import os
from typing import List
import xml.etree.ElementTree as ET
from detectron2.structures import BoxMode
from pycocotools.coco import COCO
from .data_error import IncompatibleDatasetsError


def get_class_names_pvoc(ann_dirs: List[str]) -> List[str]:
    """Get all object class names contained in a number of Pascal VOC annotation files.

    Parameters
    ----------
    ann_dirs : list of str
        Directories containing annotation files, each xml file is scanned.

    Returns
    -------
    class_names : list of str
        Ordered list of object class names

    Raises
    ------
    IncompatibleDatasetsError
        If different datasets comprise different sets of class names.
    """
    class_names = set()

    for ann_dir in ann_dirs:
        if ann_dir is None:
            continue
        classes_dir = set()
        for ann_file in glob.glob(os.path.join(ann_dir, "*.xml")):
            tree = ET.parse(ann_file)
            root = tree.getroot()

            for obj in root.findall("object"):
                name = obj.findtext("name")
                classes_dir.add(name)

        class_names.add(tuple(sorted(classes_dir)))

    if len(class_names) > 1:
        raise IncompatibleDatasetsError(
            "All data sets must contain the same categories."
        )

    class_names = class_names.pop()

    return list(class_names)


def get_class_names_coco(ann_files: List[str]) -> List[str]:
    """Get all object class names contained in a number of COCO annotation files.

    Parameters
    ----------
    ann_files : list of str
        List of COCO json files

    Returns
    -------
    class_names : list of str
        Ordered list of object class names

    Raises
    ------
    IncompatibleDatasetsError
        If different datasets comprise different sets of class names.
    """
    class_names = set()

    for ann_file in ann_files:
        if ann_file is None:
            continue
        coco_api = COCO(ann_file)
        category_ids = coco_api.getCatIds()
        classes_file = [cat["name"] for cat in coco_api.loadCats(category_ids)]
        class_names.add(tuple(sorted(classes_file)))

    if len(class_names) > 1:
        raise IncompatibleDatasetsError(
            "All data sets must contain the same categories."
        )

    class_names = class_names.pop()

    return list(class_names)


def save_class_names(
    project_dir: str,
    class_names: List[str]
    ):
    """Save object class names within the project directory.

    Parameters
    ----------
    project_dir : str
        GinJinn project directory
    class_names : list of str
        Ordered list of object class names
    """
    path = os.path.join(project_dir, "class_names.txt")
    with open(path, "w") as fout:
        fout.write("\n".join(class_names))


def get_class_names(
    project_dir: str
    ):
    """Get object class names stored in the project directory.

    Parameters
    ----------
    project_dir : str
        GinJinn project directory

    Returns
    -------
    class_names : list of str
        Ordered list of object class names
    """
    path = os.path.join(project_dir, "class_names.txt")
    with open(path, "r") as fin:
        class_names = [s.strip() for s in fin.readlines()]
    return class_names


def set_category_ids_coco(dict_list: List[dict], ann_file: str):
    """Sort categories alphabetically and assign contiguous IDs starting from 0.

    Parameters
    ----------
    dict_list : list of dict
        Annotations in Detectron2 format
    ann_file : str
        COCO json file

    Returns
    -------
    class_names: list of str
        Ordered list of object class names
    """
    coco_api = COCO(ann_file)
    category_ids = coco_api.getCatIds()
    class_names = [cat["name"] for cat in coco_api.loadCats(category_ids)]
    mapping_orig = dict(zip(category_ids, class_names))
    class_names.sort()

    for record in dict_list:
        for obj in record["annotations"]:
            id_orig = obj["category_id"]
            id_new = class_names.index(mapping_orig[id_orig])
            obj["category_id"] = id_new


# unused
def get_img_ids(img_dirs: List[str]) -> dict:
    """Assign unique ID to each JPG image from multiple directories.

    Parameters
    ----------
    img_dirs : list of str
        Directories containing JPG images.

    Returns
    -------
    img_ids : dict
        key = image path, value = image ID
    """
    i_img = 0
    img_ids = {}

    for img_dir in img_dirs:
        for img_file in os.scandir(img_dir):
            if os.path.splitext(img_file.path)[1] in ["jpg", "jpeg", "JPG", "JPEG"]:
                img_ids[img_file.path] = i_img
                i_img += 1

    return img_ids


def get_dicts_pvoc(ann_dir: str, img_dir: str, class_names: List[str]) -> List[dict]:
    """Load Pascal VOC annotations to Detectron2 format.

    Parameters
    ----------
    ann_dir : str
        Directory containing xml files
    img_dir : str
        Directory containing JPG images
    class_names : list of str
        required to assign category IDs

    Notes
    -----
    Bounding boxes contained in the input annotations are assumed to represent zero-based,
    half open intervals (xmax and ymax don't belong to the object).

    Returns
    -------
    dict_list : list of dict
        Annotations in Detectron2 format
    """
    dict_list = []

    for i_img, ann_file in enumerate(glob.glob(os.path.join(ann_dir, "*.xml"))):

        tree = ET.parse(ann_file)
        root = tree.getroot()

        record = {}

        # folder = root.findtext("folder")
        filename = root.findtext("filename")
        record["file_name"] = os.path.join(img_dir, filename)
        record["image_id"] = i_img

        size = root.find("size")
        record["height"] = int(size.findtext("height"))
        record["width"] = int(size.findtext("width"))

        record["annotations"] = []

        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = float(bbox.findtext("xmin"))
            ymin = float(bbox.findtext("ymin"))
            xmax = float(bbox.findtext("xmax"))
            ymax = float(bbox.findtext("ymax"))
            name = obj.findtext("name")

            annotation = {
                'bbox': [xmin, ymin, xmax, ymax],
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': class_names.index(name)
            }

            record["annotations"].append(annotation)

        dict_list.append(record)

    return dict_list
