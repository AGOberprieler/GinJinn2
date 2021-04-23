"""
Functions for merging multiple datasets.
"""

import datetime
import glob
import hashlib
import json
import os
import shutil
from typing import List
import xml.etree.ElementTree as ET
from ginjinn.data_reader.data_error import IncompatibleDatasetsError


def merge_datasets_pvoc(
    ann_dirs: List[str],
    img_dirs: List[str],
    outdir: str,
    link_images: bool=True
):
    """Combine multiple datasets with annotations in PascalVOC format.

    Parameters
    ----------
    ann_dirs : list of str
        Directories containing PascalVOC XML files.
        Each of he latter must have a unique file name.
    img_dirs : list of str
        Directories containing JPG images.
        Each of he latter must have a unique file name.
    outdir : str
        Output directory
    link_images : bool
        If true, images won't be copied but hard-linked instead.
    """

    # create/clean new annotation directory
    ann_outdir = os.path.join(outdir, "annotations")
    os.makedirs(ann_outdir, exist_ok=True)
    for path in glob.iglob(os.path.join(ann_outdir, "*")):
        os.remove(path)

    # create/clean new image directory
    img_outdir = os.path.join(outdir, "images")
    os.makedirs(img_outdir, exist_ok=True)
    for path in glob.iglob(os.path.join(img_outdir, "*")):
        os.remove(path)

    # find images
    extensions = ("jpg", "jpeg", "JPG", "JPEG")
    img_paths = []
    for img_dir in img_dirs:
        for ext in extensions:
            img_paths.extend(glob.glob(os.path.join(img_dir, "*." + ext)))

    # find annotation files
    extensions = ("xml", "XML")
    ann_paths = []
    for ann_dir in ann_dirs:
        for ext in extensions:
            ann_paths.extend(glob.glob(os.path.join(ann_dir, "*." + ext)))

    # check for duplicate images
    hashes = dict()
    for img_path in img_paths:
        with open(img_path, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
            if h in hashes:
                raise IncompatibleDatasetsError(
                    f"Identical images files detected:\n{hashes[h]}\n{img_path}"
                )
            else:
                hashes[h] = img_path

    # check for colliding image file names
    images = dict()
    for img_path in img_paths:
        img_name = os.path.split(img_path)[1]
        if img_name in images:
            raise IncompatibleDatasetsError(
                f"Identical file names detected:\n{images[img_name]}\n{img_path}"
            )
        else:
            images[img_name] = img_path

    # check for colliding annotation file names
    annotations = dict()
    for ann_path in ann_paths:
        ann_name = os.path.split(ann_path)[1]
        if ann_name in annotations:
            raise IncompatibleDatasetsError(
                f"Identical file names detected:\n{annotations[ann_name]}\n{ann_path}"
            )
        else:
            annotations[ann_name] = ann_path

    # write annotations
    for ann_dir in ann_dirs:
        for ann_file in glob.glob(os.path.join(ann_dir, "*.xml")):
            tree = ET.parse(ann_file)
            root = tree.getroot()

            filename = root.findtext("filename")
            for folder in root.iter("folder"):
                folder.text = img_outdir
            for path in root.iter("path"):
                path.text = os.path.join(img_outdir, filename)

            tree.write(os.path.join(ann_outdir, ann_file))

    # copy/link images
    for img_path in img_paths:
        img_name = os.path.split(img_path)[1]
        img_path_new = os.path.join(img_outdir, img_name)
        if link_images:
            os.link(img_path, img_path_new)
        else:
            shutil.copy(img_path, img_path_new)


def merge_datasets_coco(
    ann_files: List[str],
    img_dirs: List[str],
    outdir: str,
    link_images: bool=True
):
    """Combine multiple datasets with annotations in COCO format.

    Parameters
    ----------
    ann_files : list of str
        List of COCO json files
    img_dirs : list of str
        Directories containing JPG images.
        Each of he latter must have a unique file name.
    outdir : str
        Output directory
    link_images : bool
        If true, images won't be copied but hard-linked instead.
    """

    # create/clean new image directory
    img_outdir = os.path.join(outdir, "images")
    os.makedirs(img_outdir, exist_ok=True)
    for path in glob.iglob(os.path.join(img_outdir, "*")):
        os.remove(path)

    # find images
    extensions = ("jpg", "jpeg", "JPG", "JPEG")
    img_paths = []
    for img_dir in img_dirs:
        for ext in extensions:
            img_paths.extend(glob.glob(os.path.join(img_dir, "*." + ext)))

    # check for duplicate images
    hashes = dict()
    for img_path in img_paths:
        with open(img_path, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
            if h in hashes:
                raise IncompatibleDatasetsError(
                    f"Identical images files detected:\n{hashes[h]}\n{img_path}"
                )
            else:
                hashes[h] = img_path

    # check for colliding image file names
    images = dict()
    for img_path in img_paths:
        img_name = os.path.split(img_path)[1]
        if img_name in images:
            raise IncompatibleDatasetsError(
                f"Identical file names detected:\n{images[img_name]}\n{img_path}"
            )
        else:
            images[img_name] = img_path

    # copy/link images
    for img_path in img_paths:
        img_name = os.path.split(img_path)[1]
        img_path_new = os.path.join(img_outdir, img_name)
        if link_images:
            os.link(img_path, img_path_new)
        else:
            shutil.copy(img_path, img_path_new)

    # combine json files:

    info = {
        "contributor" : "",
        "date_created" : datetime.datetime.now().strftime("%Y/%m/%d"),
        "description" : "",
        "version" : "",
        "url" : "",
        "year" : ""
    }

    # name -> COCO entry
    dict_licenses = dict()
    dict_images = dict()
    dict_categories = dict()

    annotations = []

    for ann_file in ann_files:
        with open(ann_file, "rb") as f:
            ann = json.load(f)

            id_to_lic = dict() # old mapping
            for license in ann.get("licenses"):
                id_to_lic[license["id"]] = license["name"]
                if license["name"] not in dict_licenses:
                    dict_licenses[license["name"]] = license
                    dict_licenses[license["name"]]["id"] = len(dict_licenses)

            id_to_img = dict() # old mapping
            for image in ann.get("images"):
                id_to_img[image["id"]] = image["file_name"]
                dict_images[image["file_name"]] = image
                dict_images[image["file_name"]]["id"] = len(dict_images)
                license = id_to_lic[image["license"]]
                dict_images[image["file_name"]]["license"] = dict_licenses[license]["id"]

            id_to_cat = dict() # old mapping
            for category in ann.get("categories"):
                id_to_cat[category["id"]] = category["name"]
                if category["name"] not in dict_categories:
                    dict_categories[category["name"]] = category
                    dict_categories[category["name"]]["id"] = len(dict_categories)

            for annotation in ann.get("annotations"):
                annotation["id"] = len(annotations) + 1

                img_file = id_to_img[annotation["image_id"]]
                annotation["image_id"] = dict_images[img_file]["id"]

                category = id_to_cat[annotation["category_id"]]
                annotation["category_id"] = dict_categories[category]["id"]

                annotations.append(annotation)

    licenses = sorted(list(dict_licenses.values()), key=lambda d:d["id"])
    images = sorted(list(dict_images.values()), key=lambda d:d["id"])
    categories = sorted(list(dict_categories.values()), key=lambda d:d["id"])

    # write COCO annotation file
    json_new = os.path.join(outdir, "annotations.json")
    with open(json_new, 'w') as json_file:
        json.dump({
            'info': info,
            'licenses': licenses,
            'images': images,
            'annotations': annotations,
            'categories': categories
            },
            json_file,
            indent = 2,
            sort_keys = True
        )
