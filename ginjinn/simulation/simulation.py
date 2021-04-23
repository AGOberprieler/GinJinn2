'''Module for simulating simple data sets
'''

import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import skimage.filters
import skimage.util
import skimage.io
import numpy as np

from .coco_utils import \
    build_coco_annotation, build_coco_category, build_coco_dataset,\
    build_coco_image, build_coco_info, build_coco_license
from .pvoc_utils import build_pvoc_annotation, build_pvoc_object
from .shapes import add_random_circle, add_random_triangle
from .utils import polygon_area

def generate_simple_shapes_coco( #pylint: disable=too-many-arguments,too-many-locals
    img_dir: str,
    ann_file: str,
    n_images: int = 100,
    min_w: int = 400,
    max_w: int = 800,
    min_h: int = 400,
    max_h: int = 800,
    min_n_shapes: int = 1,
    max_n_shapes: int = 4,
    circle_col: np.ndarray = np.array([0.8, 0.5, 0.5]),
    triangle_col: np.ndarray = np.array([0.5, 0.8, 0.5]),
    col_var: float = 0.15,
    min_r: float = 25,
    max_r: float = 75,
    min_rot: float = 0.0,
    max_rot: float = 60.0,
    noise: float = 0.005,
    triangle_prob: float = 0.5,
):
    '''Generate a simple COCO data set.

    Parameters
    ----------
    img_dir : str
        Path to directory, where images should be stored. Must exist.
    ann_file : str
        Path to annotation file (output).
    n_images : int, optional
        Number of images to generate, by default 100
    min_w : int, optional
        Minimum image width, by default 400
    max_w : int, optional
        Maximum image width, by default 800
    min_h : int, optional
        Minimum image height, by default 400
    max_h : int, optional
        Maximum image height, by default 800
    min_n_shapes : int, optional
        Minumum number of shapes per image, by default 1
    max_n_shapes : int, optional
        Maximum number of shapes per image, by default 4
    circle_col : np.ndarray, optional
        Mean circle color, by default np.array([0.8, 0.5, 0.5])
    triangle_col : np.ndarray, optional
        Mean triangle color, by default np.array([0.5, 0.8, 0.5])
    col_var : float, optional
        Variance of colors, by default 0.15
    min_r : float, optional
        Minimum shape radius, by default 25
    max_r : float, optional
        Maximum shape radius, by default 75
    min_rot : float, optional
        Minimum shape rotation, by default 0.0
    max_rot : float, optional
        Maximum shape rotation, by default 60.0
    noise : float, optional
        Amount of gaussian noise to add to image, by default 0.005
    triangle_prob : float, optional
        Probability of drawin a triangle, by default 0.5
    '''
    category_map = {
        'circle': 1,
        'triangle': 2,
    }

    annotations = []
    images = []
    categories = [
        build_coco_category(_id, name, '') for name, _id in category_map.items()
    ]
    licenses = [build_coco_license(0)]
    info =  build_coco_info()

    ann_id = 0

    for i in range(n_images):
        w, h = np.random.randint(min_w, max_w + 1), np.random.randint(min_h, max_h + 1)
        img = np.full((h, w, 3), (0.8, 0.8, 0.8))

        img_id = i + 1
        file_name = os.path.join(img_dir, f'img_{img_id}.jpg')
        images.append(build_coco_image(
            img_id, os.path.basename(file_name), w, h
        ))

        for _ in range(np.random.randint(min_n_shapes, max_n_shapes)):
            ann_id +=1

            if np.random.uniform() > triangle_prob:
                col = np.clip(np.random.normal(circle_col, scale=col_var), 0, 1)
                contour, *_ = add_random_circle(img, min_r, max_r, col=col)
                category = 'circle'
            else:
                col = np.clip(np.random.normal(triangle_col, scale=col_var), 0, 1)
                contour, *_ = add_random_triangle(
                    img, min_r, max_r, rot_min=min_rot, rot_max=max_rot, col=col
                )
                category = 'triangle'
            bbox = np.array([*np.min(contour, 0), *np.max(contour, 0)])
            bbox = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
            area = polygon_area(contour[:,0], contour[:,1])
            cat_id = category_map[category]

            annotations.append(build_coco_annotation(
                ann_id, img_id, cat_id,
                list(bbox),
                [list(contour.flatten())],
                area,
                0
            ))

        # add noise
        img = skimage.filters.gaussian(img, sigma=1, multichannel=True)
        img = skimage.util.random_noise(img, var=noise)
        img = skimage.filters.gaussian(img, sigma=0.25, multichannel=True)
        img *= 255
        img = img.astype(np.uint8)
        skimage.io.imsave(file_name, img)

    dataset = build_coco_dataset(
        annotations,
        images,
        categories,
        licenses,
        info
    )

    with open(ann_file, 'w') as ann_f:
        json.dump(dataset, ann_f)

def generate_simple_shapes_pvoc( #pylint: disable=too-many-arguments,too-many-locals
    img_dir: str,
    ann_dir: str,
    n_images: int = 100,
    min_w: int = 400,
    max_w: int = 800,
    min_h: int = 400,
    max_h: int = 800,
    min_n_shapes: int = 1,
    max_n_shapes: int = 4,
    circle_col: np.ndarray = np.array([0.8, 0.5, 0.5]),
    triangle_col: np.ndarray = np.array([0.5, 0.8, 0.5]),
    col_var: float = 0.15,
    min_r: float = 25,
    max_r: float = 75,
    min_rot: float = 0.0,
    max_rot: float = 60.0,
    noise: float = 0.005,
    triangle_prob: float = 0.5,
    encoding: str = 'utf-8',
):
    '''Generate a simple PVOC data set.

    Parameters
    ----------
    img_dir : str
        Path to directory, where images should be stored. Must exist.
    ann_dir : str
        Path to directory, where annotations should be stored. Must exist.
    n_images : int, optional
        Number of images to generate, by default 100
    min_w : int, optional
        Minimum image width, by default 400
    max_w : int, optional
        Maximum image width, by default 800
    min_h : int, optional
        Minimum image height, by default 400
    max_h : int, optional
        Maximum image height, by default 800
    min_n_shapes : int, optional
        Minumum number of shapes per image, by default 1
    max_n_shapes : int, optional
        Maximum number of shapes per image, by default 4
    circle_col : np.ndarray, optional
        Mean circle color, by default np.array([0.8, 0.5, 0.5])
    triangle_col : np.ndarray, optional
        Mean triangle color, by default np.array([0.5, 0.8, 0.5])
    col_var : float, optional
        Variance of colors, by default 0.15
    min_r : float, optional
        Minimum shape radius, by default 25
    max_r : float, optional
        Maximum shape radius, by default 75
    min_rot : float, optional
        Minimum shape rotation, by default 0.0
    max_rot : float, optional
        Maximum shape rotation, by default 60.0
    noise : float, optional
        Amount of gaussian noise to add to image, by default 0.005
    triangle_prob : float, optional
        Probability of drawin a triangle, by default 0.5
    encoding : str, optional
        XML encoding, by default 'utf-8'.
    '''
    annotations = []
    images = []

    for i in range(n_images):
        objects = []

        w, h = np.random.randint(min_w, max_w + 1), np.random.randint(min_h, max_h + 1)
        img = np.full((h, w, 3), (0.8, 0.8, 0.8))

        img_id = i + 1
        file_name = os.path.join(img_dir, f'img_{img_id}.jpg')
        images.append(file_name)

        for _ in range(np.random.randint(min_n_shapes, max_n_shapes)):
            if np.random.uniform() > triangle_prob:
                col = np.clip(np.random.normal(circle_col, scale=col_var), 0, 1)
                contour, *_ = add_random_circle(img, min_r, max_r, col=col)
                category = 'circle'
            else:
                col = np.clip(np.random.normal(triangle_col, scale=col_var), 0, 1)
                contour, *_ = add_random_triangle(
                    img, min_r, max_r, rot_min=min_rot, rot_max=max_rot, col=col
                )
                category = 'triangle'
            bbox = np.array(
                [*np.clip(np.min(contour, 0), 0, w-1), *np.clip(np.max(contour, 0), 0, h-1)]
            ).astype(np.int)
            bbox = np.array(bbox)

            objects.append(build_pvoc_object(
                category=category,
                bbox=bbox,
                truncated=0,
                difficult=0,
                pose='unspecified'
            ))

        annotations.append(build_pvoc_annotation(
            folder='images',
            file_name=os.path.basename(file_name),
            path=os.path.abspath(file_name),
            img_size=np.array([w, h, 3]),
            objects=objects,
            segmented=0,
            database_source='Unknown',
            verified='yes',
        ))

        # add noise
        img = skimage.filters.gaussian(img, sigma=1, multichannel=True)
        img = skimage.util.random_noise(img, var=noise)
        img = skimage.filters.gaussian(img, sigma=0.25, multichannel=True)
        img *= 255
        img = img.astype(np.uint8)
        skimage.io.imsave(file_name, img)

    for ann_id, ann in enumerate(annotations):
        file_name = os.path.join(ann_dir, f'img_{ann_id + 1}.xml')

        with open(file_name, 'w') as xml_f:
            xml_f.write(
                minidom.parseString(
                    ET.tostring(ann, encoding=encoding)
                ).toprettyxml(indent='  ')
            )
