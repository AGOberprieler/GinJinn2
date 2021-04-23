''' Tests for data_splitter.py
'''

import pytest
import mock
import tempfile
import os
from ginjinn.data_reader import data_splitter
import ginjinn.simulation as simulation

@pytest.fixture(scope='module', autouse=True)
def tmp_dir():
    tmpdir = tempfile.TemporaryDirectory()
    
    yield tmpdir.name

    tmpdir.cleanup()

@pytest.fixture(scope='module')
def simulate_simple(tmp_dir):
    sim_dir = os.path.join(tmp_dir, 'simulate_simple')
    os.mkdir(sim_dir)

    img_path = os.path.join(sim_dir, 'images')
    os.mkdir(img_path)
    ann_path = os.path.join(sim_dir, 'annotations.json')

    simulation.generate_simple_shapes_coco(
        img_dir=img_path,
        ann_file=ann_path,
        n_images=20
    )

    return img_path, ann_path

def test_split_simple(tmp_dir, simulate_simple):
    img_path, ann_path = simulate_simple

    with mock.patch('builtins.input', return_value="yes"):
        split_dir = os.path.join(tmp_dir, 'test_split_simple_splitdir_0')
        data_splitter.create_split(
            ann_path, img_path, split_dir, 'instance-segmentation', 'COCO', 0.2, 0.2
        )

    split_dir = os.path.join(tmp_dir, 'test_split_simple_splitdir_1')
    data_splitter.create_split_2(
        ann_path, img_path, split_dir, 'instance-segmentation', 'COCO', 0.2, 0.2
    )

