'''Utility functions for working with PVOC data sets.
'''

from typing import List
import xml.etree.ElementTree as ET

def build_pvoc_object( #pylint: disable=too-many-arguments
    category: str,
    bbox: List[int],
    truncated: int = 0,
    difficult: int = 0,
    pose: str = 'Unspecified',
):
    '''build_pvoc_object

    Construct a PVOC object XML (xml.etree.ElementTree.ElementTree).

    Parameters
    ----------
    category : str
        Object category.
    bbox : List[int]
        List (or np.array) of [xmin, ymin, xmax, ymax].
    truncated : int, optional
        Whether the object is truncated, by default 0. Ignored by GinJinn.
    difficult : int, optional
        Whether the object is difficult, by default 0.
    pose : str, optional
        Object pose, by default 'Unspecified'. Ignore by GinJinn

    Returns
    -------
    PVOC_XML
        XML (xml.etree.ElementTree.ElementTree) representation of the
        PVOC object.
    '''

    ann_object = ET.Element('object')
    obj_name = ET.SubElement(ann_object, 'name')
    obj_name.text = str(category)
    obj_pose = ET.SubElement(ann_object, 'pose')
    obj_pose.text = str(pose)
    obj_truncated = ET.SubElement(ann_object, 'truncated')
    obj_truncated.text = str(truncated)
    obj_difficult = ET.SubElement(ann_object, 'difficult')
    obj_difficult.text = str(difficult)
    obj_bbox = ET.SubElement(ann_object, 'bndbox')
    bbox_xmin = ET.SubElement(obj_bbox, 'xmin')
    bbox_xmin.text = str(bbox[0])
    bbox_ymin = ET.SubElement(obj_bbox, 'ymin')
    bbox_ymin.text = str(bbox[1])
    bbox_xmax = ET.SubElement(obj_bbox, 'xmax')
    bbox_xmax.text = str(bbox[2])
    bbox_ymax = ET.SubElement(obj_bbox, 'ymax')
    bbox_ymax.text = str(bbox[3])

    return ann_object

def build_pvoc_annotation( #pylint: disable=too-many-arguments
    folder: str,
    file_name: str,
    path: str,
    img_size: List[int],
    objects: List,
    segmented: int = 0,
    database_source: str = 'Unknown',
    verified: str = 'yes',
):
    '''build_pvoc_annotation

    Construct a PVOC annotation XML (xml.etree.ElementTree.ElementTree).

    Parameters
    ----------
    folder : str
        Image folder relative to PVOC project.
    file_name : str
        Image file name.
    path : str
        Image file path. Ignored by GinJinn.
    img_size : List[int]
        List (or np.array) of [width, height, depth].
    objects : List
        PVOC objects as XML (xml.etree.ElementTree.ElementTree).
    segmented : int, optional
        Whether the object is segmented, by default 0. Ignored by GinJinn.
    database_source : str, optional
        Database source string, by default 'Unknown'. Ignored by GinJinn
    verified : str, optional
        Whether the annotation was verified, by default 'yes'. Ignored by GinJinn.

    Returns
    -------
    PVOC_XML
        XML (xml.etree.ElementTree.ElementTree) representation of the
        PVOC annotation.
    '''

    ann = ET.Element('annotation', {'verified': verified})

    img_folder = ET.SubElement(ann, 'folder')
    img_folder.text = str(folder)
    img_file_name = ET.SubElement(ann, 'filename')
    img_file_name.text = str(file_name)
    img_path = ET.SubElement(ann, 'path')
    img_path.text = str(path)

    source = ET.SubElement(ann, 'source')
    database = ET.SubElement(source, 'database')
    database.text = str(database_source)

    size = ET.SubElement(ann, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(img_size[0])
    height = ET.SubElement(size, 'height')
    height.text = str(img_size[1])
    depth = ET.SubElement(size, 'depth')
    depth.text = str(img_size[2])

    obj_segmented = ET.SubElement(ann, 'segmented')
    obj_segmented.text = str(segmented)

    for ann_object in objects:
        ann.append(ann_object)

    return ann
