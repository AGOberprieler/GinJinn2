"""
Module for generic helper functions.
"""

from collections import defaultdict
import sys
import json
import glob
import os
import xml
import xml.etree.ElementTree as et
from typing import List, Sequence, Tuple
import imantics
import numpy as np
from pycocotools import mask as pmask

def coco_seg_to_mask(seg, width, height):
    """coco_seg_to_mask

    Convert segmentation annotation (either list of polygons or COCO's compressed RLE)
    to binary mask.

    Parameters
    ----------
    seg : dict or list
        Segmentation annotation
    width : int
        Image/mask width
    height : int
        Image/mask height

    Returns
    -------
    seg_mask : np.ndarray
        Boolean segmentation mask

    Raises
    ------
    TypeError
        Raised for unsupported annotation formats.
    """
    if isinstance(seg, dict):
        # compressed rle to mask
        seg_mask = pmask.decode(seg).astype("bool")
    elif isinstance(seg, list):
        # polygon to mask
        polygons = imantics.Polygons(seg)
        seg_mask = polygons.mask(width, height).array
    else:
        raise TypeError(
            "Unknown segmentation format, polygons or compressed RLE expected"
        )
    return seg_mask

def bbox_from_mask(mask: np.ndarray, fmt: str):
    """Calculate bounding box from segmentation mask.

    Parameters
    ----------
    mask : np.ndarray
        Segmentation mask
    fmt : str
        Output format, either "xywh" (COCO-like) or "xyxy" (PascalVoc-like)

    Returns
    -------
    np.ndarray
        Bounding box

    Raises
    ------
    ValueError
        Raised for unsupported output formats.
    """
    x_any = mask.any(axis=0)
    y_any = mask.any(axis=1)
    x = np.where(x_any)[0].tolist()
    y = np.where(y_any)[0].tolist()

    if len(x) > 0 and len(y) > 0:
        x1, y1, x2, y2 = (x[0], y[0], x[-1] + 1, y[-1] + 1)
    else:
        x1, y1, x2, y2 = (0, 0, 0, 0)

    if fmt == "xywh":
        bbox = ( x1, y1, x2 - x1, y2 - y1 )
    elif fmt == "xyxy":
        bbox = ( x1, y1, x2, y2 )
    else:
        raise ValueError(
            f"Unknown bounding box format \"{fmt}\"."
        )
    return np.array(bbox).astype("int")

def bbox_from_polygons(polygons: List[List[float]], fmt: str):
    """Calculate bounding box from polygons.

    Parameters
    ----------
    polygons : list of list of float
        List of polygons, i.e. [[x0, y0, x1, y1, x2, y2 ...], ...]
    fmt : str
        Output format, either "xywh" (COCO-like) or "xyxy" (PascalVoc-like)

    Returns
    -------
    np.ndarray
        Bounding box

    Raises
    ------
    ValueError
        Raised for unsupported output formats.
    """
    if any(len(p) for p in polygons):
        x = np.concatenate([p[0::2] for p in polygons])
        y = np.concatenate([p[1::2] for p in polygons])
        x0 = np.min(x)
        x1 = np.max(x) + 1
        y0 = np.min(y)
        y1 = np.max(y) + 1
    else:
        x0, y0, x1, y1 = (0, 0, 0, 0)

    if fmt == "xywh":
        bbox = ( x0, y0, x1 - x0, y1 - y0 )
    elif fmt == "xyxy":
        bbox = ( x0, y0, x1, y1 )
    else:
        raise ValueError(
            f"Unknown bounding box format \"{fmt}\"."
        )
    return np.array(bbox).round().astype("int")

def confirmation(question: str) -> bool:
    """Ask question expecting "yes" or "no".

    Parameters
    ----------
    question : str
        Question to be printed

    Returns
    -------
    bool
        True or False for "yes" or "no", respectively
    """
    valid = {"yes": True, "y": True, "no": False, "n": False}

    while True:
        choice = input(question + " [y/n]\n").strip().lower()
        if choice in valid.keys():
            return valid[choice]
        print("Please type 'yes' or 'no'\n")

def confirmation_cancel(question: str) -> bool:
    '''Ask question expecting 'yes' or 'no'.

    Parameters
    ----------
    question : str
        Question to be printed

    Returns
    -------
    bool
        True or False for 'yes' or 'no', respectively
    '''
    valid = {'yes': True, 'y': True, 'no': False, 'n': False}
    cancel = ['c', 'cancel', 'quit', 'q']

    while True:
        choice = input(question + ' [y(es)/n(o)/c(ancel)]\n').strip().lower()
        if choice in valid.keys():
            return valid[choice]
        elif choice in cancel:
            sys.exit()
        print('Please type "yes" or "no" (or "cancel")\n')

def get_image_files(
    img_dir: str,
    img_file_extensions: List[str] = [
        '*.jpg', '*.JPG', '*.jpeg', '*.JPEG',
        '*.png', '*.PNG',
    ],
) -> List[str]:
    '''get_image_files

    Get paths of image files in img_dir.

    Parameters
    ----------
    img_dir : str
        Directory containing images.
    img_file_extensions : List[str], optional
        Image file extensions,
        by default [ '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', ]

    Returns
    -------
    List[str]
        List of image file paths.
    '''
    img_files = []
    for ext in img_file_extensions:
        img_files.extend(glob.glob(os.path.join(img_dir, ext)))
    return img_files

def load_coco_ann(ann_path: str) -> dict:
    '''load_coco_ann

    Load coco JSON annotation into dict.

    Parameters
    ----------
    ann_path : str
        Path to annotation JSON file.

    Returns
    -------
    dict
        [COCO annotation as dict.
    '''
    with open(ann_path) as ann_f:
        ann = json.load(ann_f)
    return ann

def get_obj_anns(img_ann, ann: dict) -> List[dict]:
    '''get_obj_anns

    Get object annotations for image from COCO annotation dict.

    Parameters
    ----------
    img_ann
        Image id or image COCO dict.
    ann : dict
        COCO annotation ("dataset").

    Returns
    -------
    List[dict]
        List of COCO object annotations for img_ann.
    '''
    if isinstance(img_ann, int):
        img_id = img_ann
    else:
        img_id = img_ann['id']

    obj_anns = [obj_ann for obj_ann in ann['annotations'] if obj_ann['image_id'] == img_id]
    return obj_anns

def plot_coco_annotated_img(img, obj_anns: List[dict], ax=None):
    '''plot_coco_annotated_img

    Plot COCO annotations on image.

    Parameters
    ----------
    img
        Image, numpy array.
    obj_anns : List[dict]
        List of COCO object annotation dicts.
    ax
        pyplot axis to plot on, by default None.

    Returns
    -------
    pyplot axis
    '''

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(img)
    overlay_coco_annotations(obj_anns, ax)

    return ax

def overlay_coco_annotations(
    obj_anns: List[dict],
    ax
):
    '''overlay_coco_annotations

    Overlay coco annotations.

    Parameters
    ----------
    obj_anns : List[dict]
        List of COCO object annotation dicts.
    ax
        pyplot axis to plot on.

    Returns
    -------
    pyplot axis
    '''

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon

    for ann in obj_anns:
        print(ann)
        seg = ann.get('segmentation', None)
        if seg:
            poly = Polygon(
                np.array(seg).reshape(-1, 2),
                fill='orange',
                edgecolor='orange',
                alpha=0.5,
            )
            ax.add_patch(poly)

        bbox = ann.get('bbox', None)
        if bbox:
            rect = Rectangle(
                bbox[0:2], bbox[2], bbox[3],
                facecolor=None,
                fill=False,
                edgecolor='orange'
            )
            ax.add_patch(rect)

    return ax

def plot_pvoc_annotated_img(
    img: np.ndarray,
    ann: xml.etree.ElementTree.ElementTree,
    ax=None,
):
    '''plot_pvoc_annotated_img

    Plot PVOC annotated image.

    Parameters
    ----------
    img : np.ndarray
        Image as numpy array
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree
    ax
        matplotlib axis, by default None

    Returns
    -------
    matplotlib axis
    '''
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(img)
    overlay_pvoc_ann(ann, ax)

    return ax

def overlay_pvoc_ann(
    ann: xml.etree.ElementTree.ElementTree,
    ax,
):
    '''overlay_pvoc_ann

    Plot PVOC annotation on ax.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree
    ax
        matplotlib axis

    Returns
    -------
    matplotlib axis
    '''
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    for obj in get_pvoc_objects(ann):
        bbox = get_pvoc_obj_bbox(obj)
        w, h = bbox_size(bbox)
        rect = Rectangle(
            [bbox[0], bbox[1]], w, h,
            facecolor=None,
            fill=False,
            edgecolor='orange'
        )
        ax.add_patch(rect)

    return ax

def load_pvoc_annotation(
    ann_path: str,
) -> xml.etree.ElementTree.ElementTree:
    '''load_pvoc_annotation

    Load PVOC annotations from file.

    Parameters
    ----------
    ann_path : str
        PVOC annotation XML file path

    Returns
    -------
    xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree
    '''
    return et.parse(ann_path)

def write_pvoc_annotation(
    ann: xml.etree.ElementTree.ElementTree,
    ann_file: str,
):
    '''write_pvoc_annotation

    Write PVOC annotation in ElementTree format to XML file.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree
    ann_file : str
        Path to annotation XML file
    '''
    import xml.dom.minidom as minidom

    root = ann.getroot()
    xmlstr = minidom.parseString(et.tostring(root)).toprettyxml(indent='  ')
    with open(ann_file, 'w') as ann_f:
        ann_f.write(xmlstr)

def clip_bbox(
    bbox: Sequence[float],
    clipping_range: Sequence[float],
) -> Sequence[float]:
    '''clip_bbox

    Clip bounding box.

    Parameters
    ----------
    bbox : Sequence[float]
        Bounding-box in x0y0x1y1 format.
    clipping_range : Sequence[float]
        Clipping range in x0x1y0y1 format.

    Returns
    -------
    Sequence[float]
        Clipped bounding box in x0y0x1y1 format.
    '''
    xmn, xmx, ymn, ymx = clipping_range
    bbox_clipped = np.clip(
        bbox,
        [xmn, ymn, xmn, ymn],
        [xmx, ymx, xmx, ymx],
    )
    return bbox_clipped

def crop_bbox(
    bbox: Sequence[float],
    cropping_range: Sequence[float],
) -> Sequence[float]:
    '''crop_bbox

    Crop bounding box. Clips bbox and converts coordinates
    to local coordinates in cropping range.

    Parameters
    ----------
    bbox : Sequence[float]
        Bounding-box in x0y0x1y1 format.
    cropping_range : Sequence[float]
        Cropping range in x0x1y0y1 format.

    Returns
    -------
    Sequence[float]
        Cropped bounding box in x0y0x1y1 format.
    '''

    bbox_clipped = clip_bbox(bbox, cropping_range)
    bbox_cropped = [
        bbox_clipped[0] - cropping_range[0],
        bbox_clipped[1] - cropping_range[2],
        bbox_clipped[2] - cropping_range[0],
        bbox_clipped[3] - cropping_range[2],
    ]

    return bbox_cropped

def bbox_size(
    bbox: Sequence[float],
) -> Tuple[float, float]:
    '''bbox_size

    Calculate bounding box size (width, height).

    Parameters
    ----------
    bbox : Sequence[float]
        Bounding-box in x0y0x1y1 format.

    Returns
    -------
    Tuple[float, float]
        Tuple of (width, height)
    '''
    return (
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    )

def bbox_area(
    bbox: Sequence[float],
) -> float:
    '''bbox_area

    Calculate bounding-box area.

    Parameters
    ----------
    bbox : Sequence[float]
        Bounding-box in x0y0x1y1 format.

    Returns
    -------
    float
        Area of the bounding-box.
    '''
    w, h = bbox_size(bbox)
    return w * h

def get_pvoc_filename(
    ann: xml.etree.ElementTree.ElementTree,
) -> str:
    '''get_pvoc_filename

    Get image file name from PVOC annotation.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree

    Returns
    -------
    str
        Image file name.
    '''
    return ann.find('filename').text

def set_pvoc_filename(
    ann: xml.etree.ElementTree.ElementTree,
    filename: str,
):
    '''set_pvoc_filename

    Set image file name for PVCO annotation.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree
    filename : str
        Image file name
    '''
    ann.find('filename').text = filename

def get_pvoc_size(
    ann: xml.etree.ElementTree.ElementTree,
) -> Sequence[int]:
    '''get_pvoc_size

    Get size of annotated image from PVOC annotation.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree

    Returns
    -------
    Sequence[int]
        Image size as Tuple (width, height, depth)
    '''
    size_node = ann.find('size')

    # necessary to deal with PVOC exported by CVAT
    try:
        depth = int(size_node.find('depth').text)
    except TypeError:
        depth = 3

    return [
        int(size_node.find('width').text),
        int(size_node.find('height').text),
        depth,
    ]

def set_pvoc_size(
    ann: xml.etree.ElementTree.ElementTree,
    size: Sequence[int],
):
    '''set_pvoc_size

    Set size value for PVOC annotation.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree
    size : Sequence[int]
        Size as sequence of width, height, depth.
    '''
    size_node = ann.find('size')
    size_node.find('width').text = str(size[0])
    size_node.find('height').text = str(size[1])
    size_node.find('depth').text = str(size[2])

def get_pvoc_objects(
    ann: xml.etree.ElementTree.ElementTree,
) -> List[xml.etree.ElementTree.ElementTree]:
    '''get_pvoc_objects

    Get a list of PVCO annotation objects in ElementTree format.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree

    Returns
    -------
    List[xml.etree.ElementTree.ElementTree]
        List of PVOC objects as ElementTree
    '''
    return ann.findall('object')

def add_pvoc_object(
    ann: xml.etree.ElementTree.ElementTree,
    obj: xml.etree.ElementTree.ElementTree,
):
    '''add_pvoc_object

    Add PVOC object to PVOC annotation.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree
    obj : xml.etree.ElementTree.ElementTree
        PVOC object as ElementTree
    '''
    r = ann.getroot()
    r.append(obj)

def drop_pvoc_objects(
    ann: xml.etree.ElementTree.ElementTree,
):
    '''drop_pvoc_objects

    Remove all objects from PVOC annotation.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree
    '''
    r = ann.getroot()
    for o in get_pvoc_objects(ann):
        r.remove(o)

def get_pvoc_obj_bbox(
    obj: xml.etree.ElementTree.ElementTree,
) -> Sequence[float]:
    '''get_pvoc_obj_bbox

    Get bounding-box from PVOC object.

    Parameters
    ----------
    obj : xml.etree.ElementTree.ElementTree
        PVOC object as ElementTree

    Returns
    -------
    Sequence[float]
        Bounding-box in x0y0x1y1 format.
    '''
    bbox_node = obj.find('bndbox')
    bbox = [
        float(bbox_node.find('xmin').text),
        float(bbox_node.find('ymin').text),
        float(bbox_node.find('xmax').text),
        float(bbox_node.find('ymax').text),
    ]
    return bbox

def get_pvoc_obj_name(
    obj: xml.etree.ElementTree.ElementTree,
) -> str:
    '''get_pvoc_obj_name

    Get name from PVOC object.

    Parameters
    ----------
    obj : xml.etree.ElementTree.ElementTree
        PVOC object as ElementTree

    Returns
    -------
    str
        Name of the PVOC object.
    '''

    return obj.find('name').text

def set_pvoc_obj_bbox(
    obj: xml.etree.ElementTree.ElementTree,
    bbox: Sequence[float],
):
    '''set_pvoc_obj_bbox

    Set bounding-box for PVOC object.

    Parameters
    ----------
    obj : xml.etree.ElementTree.ElementTree
        PVOC object as ElementTree
    bbox : Sequence[int]
        Bounding-box in x0y0x1y1 format.
    '''
    bbox_node = obj.find('bndbox')
    bbox_node.find('xmin').text = str(bbox[0])
    bbox_node.find('ymin').text = str(bbox[1])
    bbox_node.find('xmax').text = str(bbox[2])
    bbox_node.find('ymax').text = str(bbox[3])

def visualize_annotations(
    ann_path: str,
    img_dir: str,
    out_dir: str,
    ann_type: str,
    vis_type: str,
):
    '''visualize_annotations

    Visualize COCO and PVOC object annotations for PVOC and COCO annotated images.

    Parameters
    ----------
    ann_path: str
        Path to annotations JSON file for a COCO data set or path to a directory
        containing XML annotations files for a PVOC data set.
    img_dir: str
        Path to a directory containing images corresponding to annotations in
        ann_path.
    out_dir : str
        Path to an existing directory, which the visualizations should be written to.
    ann_type: str
        Type of annotation. Either "COCO" or "PVOC".
    vis_type : str
        Type of visualization. Either "segmentation" or "bbox". For PVOC annotations,
        only "bbox" is allowed.

    Raises
    ------
    Exception
        Raised if an unknown visualization type is passed.
    Exception
        Raised if ann_type is "PVOC" and vis_type is "segmentation".
    Exception
        Raised if an unknown annotation type is passed.
    '''

    import cv2
    from ginjinn.data_reader.load_datasets import \
        MetadataCatalog, DatasetCatalog, load_vis_set, \
        get_class_names_coco, get_class_names_pvoc
    from ginjinn.predictor.predictors import GJ_Visualizer, ColorMode
    from detectron2.structures.boxes import BoxMode
    from .sliding_window_merging import xywh_to_xyxy

    # input sanity checks
    if not vis_type in ['bbox', 'segmentation']:
        msg = f'Unknown visualization type "{vis_type}".'
        raise Exception(msg)

    if vis_type == 'segmentation' and ann_type == 'PVOC':
        msg = 'Visualization type "segmentation" is incompatible with annotation type "PVOC".'
        raise Exception(msg)

    # get class names
    if ann_type == 'COCO':
        class_names = get_class_names_coco([ann_path])
    elif ann_type == 'PVOC':
        class_names = get_class_names_pvoc([ann_path])
    else:
        msg = f'Unknown annotation type "{ann_type}".'
        raise Exception(msg)

    # load data set for visualization
    load_vis_set(ann_path, img_dir, ann_type)
    # filter annotations for images in img_path
    vis_set = [ann for ann in DatasetCatalog.get('vis') if os.path.isfile(ann['file_name'])]
    metadata = MetadataCatalog.get('vis')

    for img_ann in vis_set:
        img = cv2.imread(img_ann['file_name'])
        classes = np.array([ann['category_id'] for ann in img_ann['annotations']], dtype=int)

        # get bboxes
        boxes = [ann['bbox'] for ann in img_ann['annotations']]
        box_modes = [ann['bbox_mode'] for ann in img_ann['annotations']]
        boxes = np.array([
            (xywh_to_xyxy(box) if b_mode == BoxMode.XYWH_ABS else box)
            for box, b_mode in zip(boxes, box_modes)
        ])

        # get segmentation masks if available
        if vis_type == 'segmentation':
            segmentations = [ann['segmentation'] for ann in img_ann['annotations']]
            masks = np.array([
                imantics.Polygons(seg).mask(img.shape[1], img.shape[0]).array
                for seg in segmentations
            ])
        else:
            masks = None

        # visualize bboxes and segmentation masks
        gj_vis = GJ_Visualizer(
            img[:, :, ::-1],
            metadata=metadata,
            instance_mode=ColorMode.IMAGE_BW,
        )
        vis_img = gj_vis.draw_instance_predictions_gj(
            None,
            classes,
            boxes,
            class_names,
            masks = masks
        )

        vis_img.save(
            os.path.abspath(os.path.join(out_dir, os.path.basename(img_ann['file_name'])))
        )

def dataset_info(
    ann_path: str,
    img_dir: str,
    ann_type: str,
):
    '''dataset_info

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
    '''
    from ginjinn.data_reader.load_datasets import \
        MetadataCatalog, DatasetCatalog, load_vis_set
    import pandas as pd

    # load data set
    load_vis_set(ann_path, img_dir, ann_type)
    metadata = MetadataCatalog.get('vis')

    count_df = pd.DataFrame(
        columns=['#seg', '#bbox'],
        index=[*metadata.thing_classes],
        data=0
    )

    img_count = 0
    for ann in DatasetCatalog.get('vis'):
        img_count += 1
        objs = ann['annotations']
        for obj in objs:
            if obj.get('segmentation'):
                count_df.iloc[obj['category_id'], 0] += 1
            elif obj.get('bbox'):
                count_df.iloc[obj['category_id'], 1] += 1
            else:
                print(f'Invalid object annotation: {obj}')

    count_df.loc['total']= count_df.sum(numeric_only=True, axis=0)
    count_df.loc[:,'total'] = count_df.sum(numeric_only=True, axis=1)

    empty_cls_idc = count_df['total'] == 0

    print()
    print('Dataset info for dataset')
    print(f'\tann_path: {ann_path}')
    print(f'\timg_dir: {img_dir}')

    print()
    print('# images:', img_count)
    print()
    print('category distribution:')
    print(count_df)

    if empty_cls_idc.any():
        print('')
        print('<WARNING> Found categories without annotation')
        for cls_nm in count_df.index[empty_cls_idc]:
            print(f'\t- "{cls_nm}"')
        print()
        print(
            'Please remove empty categories (ginjinn utils filter) ' +\
            'if you are planning to use this dataset for model training.'
        )

def count_categories(
    ann_path: str,
) -> "pandas.DataFrame":
    '''count_categories

    Parameters
    ----------
    ann_path: str
        Path to COCO JSON annotation file.

    Returns
    -------
    "pandas.DataFrame"
        Count dataframe.
    '''
    import pandas as pd

    ann = load_coco_ann(ann_path)

    categories = sorted(
        [(cat['name'], cat['id']) for cat in ann['categories']],
        key = lambda x: x[1],
    )
    cat_map = {cat[1]: cat[0] for cat in categories}

    columns = {cat[0]: [] for cat in categories}
    index = []

    for img_ann in ann['images']:
        index.append(img_ann['file_name'])

        counts = {key: 0 for key in columns.keys()}
        for obj_ann in get_obj_anns(img_ann, ann):
            counts[cat_map[obj_ann['category_id']]] += 1

        for key, count in counts.items():
            columns[key].append(count)

    count_df = pd.DataFrame(
        data=columns,
        index=index,
    )
    count_df.index.set_names(['image'])

    return count_df

def count_categories_pvoc(
    ann_path: str,
) -> "pandas.DataFrame":
    '''count_categories_pvoc

    Parameters
    ----------
    ann_path: str
        Path to PVOC annotation folder.

    Returns
    -------
    "pandas.DataFrame"
        Count dataframe.
    '''

    import pandas as pd
    from glob import glob
    ann_files = glob(os.path.join(ann_path, '*.xml'))

    index = []
    rows = []
    for ann_file in ann_files:
        ann = load_pvoc_annotation(ann_file)
        img_name = get_pvoc_filename(ann)
        objs = get_pvoc_objects(ann)
        counts = defaultdict(int)
        for obj in objs:
            obj_name = get_pvoc_obj_name(obj)
            counts[obj_name] += 1
        rows.append(counts)
        index.append(img_name)

    count_df = pd.DataFrame(rows, index, dtype='Int64').fillna(0)
    count_df.index.set_names(['image'])

    return count_df


class InvalidAnnotationPath(Exception):
    pass

class InvalidDatasetDir(Exception):
    pass

class ImageDirNotFound(Exception):
    pass

def get_anntype(ann_path: str) -> str:
    '''get_anntype

    Get annotation type (COCO or PVOC) of ann_path.

    Parameters
    ----------
    ann_path : str
        Path to JSON file for COCO or a folder for PVOC.

    Returns
    -------
    str
        Annotation type. Either "COCO" or "PVOC".

    Raises
    ------
    InvalidAnnotationPath
        Raised if ann_path is not a valid annotation path.
    '''
    if os.path.isfile(ann_path):
        return 'COCO'
    elif os.path.isdir(ann_path):
        return 'PVOC'
    else:
        msg = f'"{ann_path}" is not a valid annotation path.'
        raise InvalidAnnotationPath(msg)

def find_img_dir(ann_path: str) -> str:
    '''find_img_dir

    Find images directory for ann_path.

    Parameters
    ----------
    ann_path : str
        Path to JSON annotation file (COCO) or folder (PVOC).

    Returns
    -------
    str
        Path to images directory.

    Raises
    ------
    ImageDirNotFound
        Raised if image directory could not be found.
    '''
    ds_dir = os.path.dirname(os.path.abspath(ann_path))
    img_dir = os.path.join(ds_dir, 'images')
    if not os.path.isdir(img_dir):
        msg = f'Could not find "images" folder as a sibling to "{ann_path}".'
        raise ImageDirNotFound(msg)

    return img_dir

def get_dstype(data_dir: str) -> str:
    '''get_dstype

    Get annotation type (COCO or PVOC) of dataset in data_dir.

    Parameters
    ----------
    data_dir : str
        Dataset directory. Must contain a folder named "images", and
        a file named "annotations.json" for a COCO dataset or
        a folder named "annotations" for a PVOC dataset.

    Returns
    -------
    str
        Dataset type. Either "COCO" or "PVOC".

    Raises
    ------
    InvalidDatasetDir
        Raised if data_dir is not a valid dataset directory.
    '''

    dir_content = os.listdir(data_dir)
    if not 'images' in dir_content:
        msg = f'Could not find "images" folder in "{data_dir}".'
        raise InvalidDatasetDir(msg)
    images_path = os.path.join(data_dir, 'images')
    if not os.path.isdir(images_path):
        msg = f'"{images_path}" is not a folder.'
        raise InvalidDatasetDir(msg)

    # COCO
    if 'annotations.json' in dir_content:
        ann_path = os.path.join(data_dir, 'annotations.json')
        if not os.path.isfile(ann_path):
            msg = f'"{ann_path}" is not a file.'
            raise InvalidDatasetDir(msg)
        ds_type = 'COCO'
    # PVOC
    elif 'annotations' in dir_content:
        ann_path = os.path.join(data_dir, 'annotations')
        if not os.path.isdir(ann_path):
            msg = f'"{ann_path}" is not a folder.'
            raise InvalidDatasetDir(msg)
        ds_type = 'PVOC'
    else:
        msg = f'Could not find annotations ("annotations" or "annotations.json") in {data_dir}.'
        raise InvalidDatasetDir(msg)

    return ds_type
