'''Module for generating random shapes.
'''

# from typing import Optional, Tuple

import numpy as np
import skimage.draw
import skimage.measure
import skimage.filters

try:
    draw_circle = skimage.draw.disk
except: # pylint: disable=bare-except
    def draw_circle(center, radius, shape=None):
        '''Wrapper around skimage.draw.circle for backward compatibility,
        when skimage.draw.disk is not available

        Parameters
        ----------
        center
            center of the circle
        radius
            radius of the circle
        shape, optional
            shape of "image" to draw to, by default None

        Returns
        -------
        Circle
            Numpy "image" containing a circle
        '''
        return skimage.draw.circle(center[0], center[1], radius, shape)

# Should probably find/clip contours using
# https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm
# instead of using skimage.

def add_circle(
    img: np.ndarray,
    c_xy: np.ndarray,
    r,
    col=(1,1,1)
):
    '''Add circle to image.

    This function modfies img in place.

    Parameters
    ----------
    img : np.ndarray
        Image to be modified as numpy array.
    c_xy : np.ndarray
        Center coordinates of the circle
    r : float
        Radius of the circle
    col : tuple, optional
        Color of the circle, by default white: (1,1,1)

    Returns
    -------
    Tuple
        Tuple of circle data: (contour, center_xy, radius)
    '''

    # circle_coords = skimage.draw.disk(c_xy[[1,0]], radius=r, shape=img.shape)
    circle_coords = draw_circle(c_xy[[1,0]], radius=r, shape=img.shape)


    img[circle_coords] = col

    # need to add border for find_contours to work at the corners of the image
    tmp_img = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.int32)
    tmp_img[(circle_coords[0] + 1, circle_coords[1] + 1)] = 1

    circle_contour = sorted(skimage.measure.find_contours(tmp_img, level=0), key=len)[0] - 1

    return circle_contour[:,[1,0]], c_xy, r

def add_random_circle(
    img: np.ndarray,
    r_min,
    r_max,
    col=(1,1,1)
):
    '''Add random circle to image

    Parameters
    ----------
    img : np.ndarray
        Image to be modified as numpy array.
    r_min : float
        Minimum radius of the circle
    r_max : float
        Maximum radius of the circle
    col : tuple, optional
        Color of the circle, by default white: (1,1,1)

    Returns
    -------
    Tuple
        Tuple of circle data: (contour, center_xy, radius)
    '''
    h, w, _ = img.shape
    r = np.random.randint(r_min, r_max)
    c_x = np.random.randint(r, w)
    c_y = np.random.randint(r, h)
    c_xy = np.array([c_x, c_y])

    return add_circle(img, c_xy, r, col)

def add_triangle(
    img: np.ndarray,
    c_xy: np.ndarray,
    r,
    rot=0.0,
    col=(1,1,1)
):
    '''Add equilateral triangle to image

    Parameters
    ----------
    img : np.ndarray
        Image to be modified as numpy array.
    c_xy : np.ndarray
        Center coordinates of the triangle
    r : float
        Radius of the triangle
    rot : float, optional
        Rotation of the triangle in degrees, by default 0.0
    col : tuple, optional
        Color of the triangle, by default white: (1,1,1)

    Returns
    -------
    Tuple
        Tuple of triangle data: (contour, center_xy, radius, rotation)
    '''
    rot_mat_120 = np.array([
        [-1/2, -np.sqrt(3)/2],
        [np.sqrt(3)/2, -1/2]
    ])

    rot = rot * np.pi / 180
    rot_mat = np.array([
        [np.cos(rot), -np.sin(rot)],
        [np.sin(rot), np.cos(rot)]
    ])

    # vertices a, b, c of the triangle
    a = np.dot(rot_mat, np.array([0, r]))
    b = np.dot(rot_mat_120, a)
    c = np.dot(rot_mat_120, b)

    rows = np.array([a[1], b[1], c[1]]) + c_xy[1]
    cols = np.array([a[0], b[0], c[0]]) + c_xy[0]

    triangle_coords = skimage.draw.polygon(rows, cols, shape=img.shape)
    img[triangle_coords] = col

    # need to add border for find_contours to work at the corners of the image
    tmp_img = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    tmp_img[(triangle_coords[0] + 1, triangle_coords[1] + 1)] = 1
    tmp_img = skimage.filters.gaussian(tmp_img, 0.25)
    triangle_contour = sorted(skimage.measure.find_contours(tmp_img, level=0), key=len)[0] - 1


    return triangle_contour[:,[1,0]], c_xy, r, rot


def add_random_triangle( #pylint: disable=too-many-arguments
    img: np.ndarray,
    r_min,
    r_max,
    rot_min=0.0,
    rot_max=0.0,
    col=(1,1,1)
):
    '''Add random triangle to image

    Parameters
    ----------
    img : np.ndarray
        Image to be modified as numpy array.
    r_min : float
        Minimum radius of the triangle
    r_max : float
        Maximum radius of the triangle
    rot_min : float, optional
        Minimum rotation of the triangle in degrees, by default 0.0
    rot_max : float, optional
        Maximum rotation of the triangle in degrees, by default 0.0
    col : tuple, optional
        Color of the triangle, by default white: (1,1,1)

    Returns
    -------
    Tuple
        Tuple of triangle data: (contour, center_xy, radius, rotation)
    '''
    h, w, _ = img.shape
    r = np.random.randint(r_min, r_max)
    c_x = np.random.randint(r, w)
    c_y = np.random.randint(r, h)
    c_xy = np.array([c_x, c_y])

    if rot_min == rot_max:
        rot = rot_min
    else:
        rot = np.random.randint(rot_min, rot_max)

    return add_triangle(img, c_xy, r, rot, col)
