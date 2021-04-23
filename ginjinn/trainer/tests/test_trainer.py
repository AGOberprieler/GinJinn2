import pytest
import sys
import copy
import tempfile
import os
import mock
import pkg_resources
import yaml

from detectron2.data import DatasetCatalog

from ginjinn.ginjinn_config import GinjinnConfiguration
from ginjinn.simulation import generate_simple_shapes_coco
from ginjinn.data_reader.load_datasets import load_train_val_sets

from ginjinn.trainer.trainer import ValTrainer, Trainer

@pytest.fixture(scope='module', autouse=True)
def tmp_dir():
    tmpdir = tempfile.TemporaryDirectory()

    yield tmpdir.name

    tmpdir.cleanup()

@pytest.fixture(scope='module')
def simulate_coco_train(tmp_dir):
    sim_dir = os.path.join(tmp_dir, 'sim_coco_train')
    os.mkdir(sim_dir)

    img_dir = os.path.join(sim_dir, 'images')
    os.mkdir(img_dir)
    ann_path = os.path.join(sim_dir, 'annotations.json')
    generate_simple_shapes_coco(
        img_dir=img_dir, ann_file=ann_path, n_images=40,
    )
    return img_dir, ann_path

@pytest.fixture(scope='module')
def simulate_coco_validation(tmp_dir):
    sim_dir = os.path.join(tmp_dir, 'sim_coco_validation')
    os.mkdir(sim_dir)

    img_dir = os.path.join(sim_dir, 'images')
    os.mkdir(img_dir)
    ann_path = os.path.join(sim_dir, 'annotations.json')
    generate_simple_shapes_coco(
        img_dir=img_dir, ann_file=ann_path, n_images=20,
    )
    return img_dir, ann_path

@pytest.fixture(scope='module', autouse=True)
def example_config(tmp_dir, simulate_coco_train, simulate_coco_validation):
    img_dir_train, ann_path_train = simulate_coco_train
    img_dir_validation, ann_path_validation = simulate_coco_validation

    example_config_1_path = pkg_resources.resource_filename(
        'ginjinn', 'data/ginjinn_config/example_config_1.yaml',
    )

    with open(example_config_1_path) as config_f:
        config = yaml.load(config_f)

    config['input']['training']['annotation_path'] = ann_path_train
    config['input']['training']['image_path'] = img_dir_train

    config['input']['validation'] = {}
    config['input']['validation']['annotation_path'] = ann_path_validation
    config['input']['validation']['image_path'] = img_dir_validation

    config['augmentation'] = [aug for aug in config['augmentation'] if not 'crop' in list(aug.keys())[0]]
    # config['augmentation'] = [config['augmentation'][0]]

    config['training']['max_iter'] = 100

    config_dir = os.path.join(tmp_dir, 'example_config')
    os.mkdir(config_dir)
    os.mkdir(os.path.join(config_dir, 'outputs'))
    config['project_dir'] = os.path.abspath(config_dir)

    config_file = os.path.join(config_dir, 'ginjinn_config.yaml')
    with open(config_file, 'w') as config_f:
        yaml.dump(config, config_f)

    return (config, config_file)

def test_trainer(example_config):
    _, config_file = example_config
    config = GinjinnConfiguration.from_config_file(config_file)

    print(config)

    try:
        DatasetCatalog.remove('train')
    except:
        pass
    try:
        DatasetCatalog.remove('val')
    except:
        pass
    
    load_train_val_sets(config)
    
    try:
        trainer = ValTrainer.from_ginjinn_config(config)
        trainer.resume_or_load(resume=False)
        trainer.train()
    except AssertionError as err:
        if 'NVIDIA driver' in str(err):
            Warning(str(err))
        else:
            raise err
    except RuntimeError as err:
        if 'NVIDIA driver' in str(err):
            Warning(str(err))
        else:
            raise err
    except Exception as err:
        raise err