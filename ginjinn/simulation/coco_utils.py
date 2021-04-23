'''Utility functions for working with COCO data sets.
'''

from typing import List

def build_coco_dataset(
    annotations: List[dict],
    images: List[dict],
    categories: List[dict],
    licenses: List[dict],
    info: dict
):
    '''Construct COCO data set dictionary.

    Parameters
    ----------
    annotations : List[dict]
        List of annotations
    images : List[dict]
        List of images
    categories : List[dict]
        List of categories
    licenses : List[dict]
        List of licenses
    info : dict
        Info dictionary

    Returns
    -------
    dict
        COCO data set dictionary.
    '''
    return {
        'annotations': annotations,
        'images': images,
        'categories': categories,
        'licenses': licenses,
        'info': info,
    }

def build_coco_annotation( #pylint: disable=too-many-arguments
    ann_id: int,
    image_id: int,
    category_id: int,
    bbox: List[float],
    segmentation: List[float],
    area: float,
    iscrowd: int = 0
):
    '''Construct COCO annotation dictionary.

    Parameters
    ----------
    ann_id : int
        Annotation ID
    image_id : int
        Image ID
    category_id : int
        Category ID
    bbox : List[float]
        xmin, ymin, width, height
    segmentation : List[float]
        x1, y1, x2, y2, ..., xn, yn
    area : float
        Annotated area.
    iscrowd : int, optional
        Whether the annotation is a crowd, by default 0

    Returns
    -------
    dict
        COCO annotation dictionary.
    '''
    return {
        'id': int(ann_id),
        'bbox': bbox,
        'image_id': int(image_id),
        'segmentation': segmentation,
        'category_id': int(category_id),
        'area': float(area),
        'iscrowd': int(iscrowd)
    }

def build_coco_category(
    category_id: int,
    name: str,
    supercategory: str = '',
):
    '''Construct COCO category dictionary.

    Parameters
    ----------
    category_id : int
        Category ID
    name : str
        Category name
    supercategory : str, optional
        Name of the supercategory, by default ''

    Returns
    -------
    dict
        COCO category dictionary.
    '''
    return {
        'id': category_id,
        'supercategory': supercategory,
        'name': name,
    }

def build_coco_image( #pylint: disable=too-many-arguments,redefined-builtin
    image_id: int,
    file_name: str,
    width: int,
    height: int,
    license: int = 0,
    coco_url: str = '',
    date_captured: int = 0,
    flickr_url: str = ''
):
    '''Construct COCO image dictionary

    Parameters
    ----------
    image_id : int
        Image ID
    file_name : str
        Image file name/path
    width : int
        Image width in pixel
    height : int
        Image height in pixel
    license : int, optional
        License ID, by default 0
    coco_url : str, optional
        COCO URL, by default ''
    date_captured : int, optional
        Date, by default 0
    flickr_url : str, optional
        Flickr URL, by default ''

    Returns
    -------
    dict
        COCO image dictionary
    '''
    return {
        'id': int(image_id),
        'file_name': file_name,
        'width': width,
        'height': height,
        'license': license,
        'date_captured': date_captured,
        'coco_url': coco_url,
        'flickr_url': flickr_url
    }

def build_coco_license(
    license_id: int,
    name: str = '',
    url: str = '',
):
    '''Construct COCO license dictionary.

    Parameters
    ----------
    license_id : int
        License ID
    name : str, optional
        License name, by default ''
    url : str, optional
        License URL, by default ''

    Returns
    -------
    dict
        COCO license dictionary
    '''
    return {
        'name': name,
        'url': url,
        'id': license_id,
    }

def build_coco_info( #pylint: disable=too-many-arguments
    version: str = '',
    year: str = '',
    description: str = '',
    url: str = '',
    date_created: str = '',
    contributor: str = '',
):
    '''Construct COCO info dictionary

    Parameters
    ----------
    version: str, optional
        version, by default '
    year : str, optional
        year, by default ''
    description : str, optional
        description, by default ''
    url : str, optional
        url, by default ''
    date_created : str, optional
        date_created, by default ''
    contributor : str, optional
        contributor, by default ''

    Returns
    -------
    dict
        COCO description dictionary
    '''
    return {
        'version': version,
        'year': year,
        'description': description,
        'url': url,
        'date_created': date_created,
        'contributor': contributor
    }
