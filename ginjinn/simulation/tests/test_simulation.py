''' Tests for data simulation
'''

import os
import tempfile
import pytest

from ginjinn.simulation import generate_simple_shapes_coco, generate_simple_shapes_pvoc

@pytest.fixture(scope='module', autouse=True)
def tmp_dir():
    tmpdir = tempfile.TemporaryDirectory()

    yield tmpdir.name

    tmpdir.cleanup()

def test_simple_shapes_coco(tmp_dir):
    test_dir = os.path.join(tmp_dir, 'test_simple_shapes_coco')
    os.mkdir(test_dir)

    img_dir = os.path.join(test_dir, 'images')
    os.mkdir(img_dir)

    ann_file = os.path.join(test_dir, 'annotations.json')
    generate_simple_shapes_coco(
        img_dir,
        ann_file,
        n_images=10
    )

    generate_simple_shapes_coco(
        img_dir,
        ann_file,
        n_images=10,
        min_rot=0, max_rot=0
    )

def test_simple_shapes_pvoc(tmp_dir):
    test_dir = os.path.join(tmp_dir, 'test_simple_shapes_pvoc')
    os.mkdir(test_dir)

    img_dir = os.path.join(test_dir, 'images')
    os.mkdir(img_dir)

    ann_dir = os.path.join(test_dir, 'annotations')
    os.mkdir(ann_dir)

    generate_simple_shapes_pvoc(
        img_dir,
        ann_dir,
        n_images=10
    )

    generate_simple_shapes_pvoc(
        img_dir,
        ann_dir,
        n_images=10,
        min_rot=0, max_rot=0
    )