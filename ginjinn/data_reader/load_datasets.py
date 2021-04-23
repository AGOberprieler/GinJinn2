"""
Load and register datasets as specified by a GinjinnConfiguration object.
"""

from typing import List
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from ginjinn.ginjinn_config.ginjinn_config import GinjinnConfiguration
from .data_error import IncompatibleDatasetsError
from .data_reader import get_class_names_coco, get_class_names_pvoc
from .data_reader import get_dicts_pvoc, set_category_ids_coco
from .data_reader import get_class_names, save_class_names


def register_dicts(
    dict_list: List[dict],
    dataset_name: str,
    class_names: List[str]
    ):
    """Register list of dictionaries as Detectron2 dataset.

    Parameters
    ----------
    dict_list : list of dict
        Detectron2 default dictionary format
    dataset_name : str
        Name of the dataset to be registered
    class_names : list of str
        Ordered list of object class names
    """
    if dict_list:
        DatasetCatalog.register(dataset_name, lambda: dict_list)
        MetadataCatalog.get(dataset_name).thing_classes = class_names


def register_coco(
    ann_file: str,
    img_dir: str,
    dataset_name: str,
    class_names: List[str]
    ):
    """Register dataset with annotations in COCO format in Detectron2.

    Parameters
    ----------
    ann_file : str
        Path of annotation json file
    img_dir : str
        Directory containing JPG images
    dataset_name : str
        Name of the dataset to be registered
    class_names : list of str
        Ordered list of object class names
    """
    if ann_file and img_dir:
        dict_list = load_coco_json(
            ann_file,
            img_dir
        )
        set_category_ids_coco(
            dict_list,
            ann_file
        )
        DatasetCatalog.register(dataset_name, lambda: dict_list)
        MetadataCatalog.get(dataset_name).thing_classes = class_names


def register_pvoc(
    ann_dir: str,
    img_dir: str,
    dataset_name: str,
    class_names: List[str]
    ):
    """Register dataset with annotations in PascalVOC format in Detectron2.

    Parameters
    ----------
    ann_dir : str
        Directory containing annotation xml files
    img_dir : str
        Directory containing JPG images
    dataset_name : str
        Name of the dataset to be registered
    class_names : list of str
        Ordered list of object class names
    """
    if ann_dir and img_dir:
        dict_list = get_dicts_pvoc(
            ann_dir,
            img_dir,
            class_names
        )
        DatasetCatalog.register(dataset_name, lambda: dict_list)
        MetadataCatalog.get(dataset_name).thing_classes = class_names


def load_test_set(
        cfg: GinjinnConfiguration
    ):
    """Read and register test dataset.

    Parameters
    ----------
    cfg : GinjinnConfiguration

    Raises
    ------
    IncompatibleDatasetsError
        If the class names contained in the test set do not match those of the training set.
    """
    class_names_project = get_class_names(cfg.project_dir)
    ann_path = cfg.input.test.annotation_path
    img_path = cfg.input.test.image_path

    if cfg.input.type == "COCO":
        class_names_test = get_class_names_coco([ann_path])
        if class_names_test != class_names_project:
            raise IncompatibleDatasetsError(
                "The test set must contain the same categories as the training set."
            )
        register_coco(ann_path, img_path, "test", class_names_test)

    elif cfg.input.type == "PVOC":
        class_names_test = get_class_names_pvoc([ann_path])
        if class_names_test != class_names_project:
            raise IncompatibleDatasetsError(
                "The test set must contain the same categories as the training set."
            )
        register_pvoc(ann_path, img_path, "test", class_names_test)

def load_train_val_sets (
        cfg: GinjinnConfiguration
    ):
    """Read and register datasets for training and, optionally, validation.

    Parameters
    ----------
    cfg : GinjinnConfiguration

    Raises
    ------
    IncompatibleDatasetsError
        If datasets for training and validation comprise different sets of class names.
    """
    ann_path_train = cfg.input.train.annotation_path
    img_path_train = cfg.input.train.image_path

    if not cfg.input.val is None:
        ann_path_val = cfg.input.val.annotation_path
        img_path_val = cfg.input.val.image_path

    if cfg.input.type == "COCO":
        if not cfg.input.val is None:
            class_names = get_class_names_coco([ann_path_train, ann_path_val])
            register_coco(ann_path_train, img_path_train, "train", class_names)
            register_coco(ann_path_val, img_path_val, "val", class_names)
        else:
            class_names = get_class_names_coco([ann_path_train])
            register_coco(ann_path_train, img_path_train, "train", class_names)

    elif cfg.input.type == "PVOC":
        if not cfg.input.val is None:
            class_names = get_class_names_pvoc([ann_path_train, ann_path_val])
            register_pvoc(ann_path_train, img_path_train, "train", class_names)
            register_pvoc(ann_path_val, img_path_val, "val", class_names)
        else:
            class_names = get_class_names_pvoc([ann_path_train])
            register_pvoc(ann_path_train, img_path_train, "train", class_names)

    save_class_names(cfg.project_dir, class_names)

def load_vis_set(
        ann_path: str,
        img_dir: str,
        ann_type: str,
    ):
    """Read and register a visualization ("vis") data set.
    The registered data set can be accessed via DatasetCatalog.get('vis').

    Parameters
    ----------
    ann_path: str
        Path to annotations JSON file for a COCO data set or path to a directory
        containing XML annotations files for a PVOC data set.
    img_dir: str
        Path to a directory containing images corresponding to annotations in
        ann_path.
    ann_type: str
        Type of annotation. Either "COCO" or "PVOC".

    Raises
    ------
    Exception
        Raised if an invalid annotation type is passed.
    """

    if ann_type == "COCO":
        class_names = get_class_names_coco([ann_path])
        register_coco(ann_path, img_dir, "vis", class_names)
    elif ann_type == "PVOC":
        class_names = get_class_names_pvoc([ann_path])
        register_pvoc(ann_path, img_dir, "vis", class_names)
    else:
        msg = f'Unknown annotation type "{ann_type}".'
        raise Exception(msg)
