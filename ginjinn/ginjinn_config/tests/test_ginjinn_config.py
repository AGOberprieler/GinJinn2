'''Tests for GinjinnConfiguration
'''

import pkg_resources
import tempfile
import os
import pytest
import yaml
import copy

from ginjinn.ginjinn_config import GinjinnConfiguration, InvalidGinjinnConfigurationError
from ginjinn.ginjinn_config.config_error import InvalidModelConfigurationError

@pytest.fixture(scope='module', autouse=True)
def tmp_input_paths():
    tmpdir = tempfile.TemporaryDirectory()

    img_path_train = os.path.join(tmpdir.name, 'images_train')
    os.mkdir(img_path_train)
    img_path_test = os.path.join(tmpdir.name, 'images_test')
    os.mkdir(img_path_test)
    img_path_validation = os.path.join(tmpdir.name, 'images_validation')
    os.mkdir(img_path_validation)

    pvoc_ann_path_train = os.path.join(tmpdir.name, 'annotations_train')
    os.mkdir(pvoc_ann_path_train)
    pvoc_ann_path_test = os.path.join(tmpdir.name, 'annotations_test')
    os.mkdir(pvoc_ann_path_test)
    pvoc_ann_path_validation = os.path.join(tmpdir.name, 'annotations_validation')
    os.mkdir(pvoc_ann_path_validation)

    coco_ann_path_train = os.path.join(tmpdir.name, 'annotations_train.json')
    with open(coco_ann_path_train, 'w') as ann_f:
        ann_f.write('')
    coco_ann_path_test = os.path.join(tmpdir.name, 'annotations_test.json')
    with open(coco_ann_path_test, 'w') as ann_f:
        ann_f.write('')
    coco_ann_path_validation = os.path.join(tmpdir.name, 'annotations_valiation.json')
    with open(coco_ann_path_validation, 'w') as ann_f:
        ann_f.write('')

    yield {
        'coco_ann_path_train': coco_ann_path_train,
        'coco_ann_path_test': coco_ann_path_test,
        'coco_ann_path_validation': coco_ann_path_validation,
        'pvoc_ann_path_train': pvoc_ann_path_train,
        'pvoc_ann_path_test': pvoc_ann_path_test,
        'pvoc_ann_path_validation': pvoc_ann_path_validation,
        'img_path_train': img_path_train,
        'img_path_test': img_path_test,
        'img_path_validation': img_path_validation,
        'project_dir': tmpdir.name,
    }

    tmpdir.cleanup()


@pytest.fixture
def config_dicts(tmp_input_paths):
    simple_config = {
        'project_dir': tmp_input_paths['project_dir'],
        'task': 'bbox-detection',
        'input': {
            'type': 'PVOC',
            'training': {
                'annotation_path': tmp_input_paths['pvoc_ann_path_train'],
                'image_path': tmp_input_paths['img_path_train'],
            },
        },
        'model': {
            'name': 'faster_rcnn_R_50_FPN_3x',
            'weights': '',
            'model_parameters': {
                'roi_heads': {
                    'batch_size_per_image': 4096,
                    'iou_thresholds': [0.5],
                },
                'anchor_generator': {
                    'sizes': [[32, 64, 128, 256]],
                    'angles': [[-90, 0, 90]],
                    'aspect_ratios': [[0.25, 0.5, 0.75, 1.0, 1.25]]
                },
                'rpn': {
                    'iou_thresholds': [0.3, 0.7],
                    'batch_size_per_image': 256,
                },
                'box_head': {
                    'class_agnostic': False,
                    'train_on_pred_boxes': False,
                }
            },
        },
        'training': {
            'learning_rate': 0.002,
            'batch_size': 1,
            'max_iter': 10000,
        },
        'augmentation': [
            {
                'horizontal_flip': {
                    'probability': 0.25
                }
            },
            {
                'vertical_flip': {
                    'probability': 0.25
                }
            },
        ],
    }

    return [
        simple_config
    ]

@pytest.fixture
def config_file_examples():
    example_config_0_path = pkg_resources.resource_filename(
        'ginjinn', 'data/ginjinn_config/example_config_0.yaml',
    )
    example_config_1_path = pkg_resources.resource_filename(
        'ginjinn', 'data/ginjinn_config/example_config_1.yaml',
    )

    return [
        example_config_0_path,
        example_config_1_path,
    ]

def read_config_file(file_path):
    with open(file_path) as config_file:
        config = yaml.safe_load(config_file)
    return config

def test_from_dictionary_simple(config_dicts):
    simple_config_dict = config_dicts[0]

    ginjinn_config_0 = GinjinnConfiguration.from_dictionary(simple_config_dict)
    assert ginjinn_config_0.task == simple_config_dict['task'] and\
        ginjinn_config_0.project_dir == simple_config_dict['project_dir'],\
        'simple base configuration not set.'
    # TODO implement model and augmentation assertions!
    assert ginjinn_config_0.model.name == simple_config_dict['model']['name']
    assert ginjinn_config_0.training.learning_rate == simple_config_dict['training']['learning_rate']
    assert ginjinn_config_0.input.type == simple_config_dict['input']['type']

def test_from_config_file_simple(config_file_examples):
    simple_config_file_0 = config_file_examples[0]
    simple_config_dict_0 = read_config_file(simple_config_file_0)
    print(simple_config_dict_0)
    if not os.path.exists(simple_config_dict_0['input']['training']['annotation_path']):
        os.mkdir(simple_config_dict_0['input']['training']['annotation_path'])
    if not os.path.exists(simple_config_dict_0['input']['training']['image_path']):
        os.mkdir(simple_config_dict_0['input']['training']['image_path'])

    simple_config_0 = GinjinnConfiguration.from_config_file(simple_config_file_0)
    assert simple_config_0.task == simple_config_dict_0['task'] and\
        simple_config_0.project_dir == simple_config_dict_0['project_dir'] and\
        simple_config_0.input.train.annotation_path == os.path.abspath(simple_config_dict_0['input']['training']['annotation_path']) and\
        simple_config_0.input.train.image_path ==  os.path.abspath(simple_config_dict_0['input']['training']['image_path']),\
        'GinjinnConfig was not successfully constructed from simple configuration file.'
    
    assert simple_config_0.model.name == simple_config_dict_0['model']['name']
    assert simple_config_0.training.learning_rate == simple_config_dict_0['training']['learning_rate']

    os.rmdir(simple_config_dict_0['input']['training']['annotation_path'])
    os.rmdir(simple_config_dict_0['input']['training']['image_path'])


    simple_config_file_1 = config_file_examples[1]
    simple_config_dict_1 = read_config_file(simple_config_file_1)
    print(simple_config_dict_1)
    if not os.path.exists(simple_config_dict_1['input']['training']['annotation_path']):
        with open(simple_config_dict_1['input']['training']['annotation_path'], 'w') as f:
            f.write('')
    if not os.path.exists(simple_config_dict_1['input']['training']['image_path']):
        os.mkdir(simple_config_dict_1['input']['training']['image_path'])

    simple_config_1 = GinjinnConfiguration.from_config_file(simple_config_file_1)
    assert simple_config_1.task == simple_config_dict_1['task'] and\
        simple_config_1.project_dir == simple_config_dict_1['project_dir'] and\
        simple_config_1.input.train.annotation_path ==  os.path.abspath(simple_config_dict_1['input']['training']['annotation_path']) and\
        simple_config_1.input.train.image_path ==  os.path.abspath(simple_config_dict_1['input']['training']['image_path']),\
        'GinjinnConfig was not successfully constructed from simple configuration file.'
    
    assert simple_config_1.model.name == simple_config_dict_1['model']['name']
    assert simple_config_1.training.learning_rate == simple_config_dict_1['training']['learning_rate']

    os.remove(simple_config_dict_1['input']['training']['annotation_path'])
    os.rmdir(simple_config_dict_1['input']['training']['image_path'])


def test_invalid_task(config_dicts):
    simple_config_dict = copy.deepcopy(config_dicts[0])
    simple_config_dict['task'] = 'foobar'

    with pytest.raises(InvalidGinjinnConfigurationError):
        GinjinnConfiguration.from_dictionary(simple_config_dict)

def test_incompatible_task(config_dicts):
    simple_config_dict = copy.deepcopy(config_dicts[0])
    simple_config_dict['task'] = 'instance-segmentation'

    with pytest.raises(InvalidGinjinnConfigurationError):
        GinjinnConfiguration.from_dictionary(simple_config_dict)

def test_to_detectron2_config(config_dicts):
    simple_config_dict = config_dicts[0]

    ginjinn_config_0 = GinjinnConfiguration.from_dictionary(simple_config_dict)
    detectron2_config_0 = ginjinn_config_0.to_detectron2_config()

    # TODO additional tests

def test_model_parameters(config_dicts):
    
    config_dict = copy.deepcopy(config_dicts[0])
    config_dict['model']['model_parameters']['invalid_parameter'] = {}
    with pytest.raises(InvalidModelConfigurationError):
        GinjinnConfiguration.from_dictionary(config_dict)

    config_dict = copy.deepcopy(config_dicts[0])
    config_dict['model']['model_parameters']['anchor_generator']['invalid_parameter'] = 1
    with pytest.raises(InvalidModelConfigurationError):
        GinjinnConfiguration.from_dictionary(config_dict)

    config_dict = copy.deepcopy(config_dicts[0])
    config_dict['model']['model_parameters']['rpn']['invalid_parameter'] = 1
    with pytest.raises(InvalidModelConfigurationError):
        GinjinnConfiguration.from_dictionary(config_dict)

    config_dict = copy.deepcopy(config_dicts[0])
    config_dict['model']['model_parameters']['roi_heads']['invalid_parameter'] = 1
    with pytest.raises(InvalidModelConfigurationError):
        GinjinnConfiguration.from_dictionary(config_dict)

    config_dict = copy.deepcopy(config_dicts[0])
    config_dict['model']['model_parameters']['box_head']['invalid_parameter'] = 1
    with pytest.raises(InvalidModelConfigurationError):
        GinjinnConfiguration.from_dictionary(config_dict)

    config_dict = copy.deepcopy(config_dicts[0])
    config_dict['model']['name'] = 'mask_rcnn_R_50_C4_1x'
    config_dict['task'] = 'instance-segmentation'
    del config_dict['model']['model_parameters']['box_head']
    config_dict['model']['model_parameters']['mask_head'] = {
        'class_agnostic': False,
        'pooler_resolution': 10,
    }
    GinjinnConfiguration.from_dictionary(config_dict)

    config_dict['model']['model_parameters']['mask_head']['invalid_parameter'] = 1
    with pytest.raises(InvalidModelConfigurationError):
        GinjinnConfiguration.from_dictionary(config_dict)
