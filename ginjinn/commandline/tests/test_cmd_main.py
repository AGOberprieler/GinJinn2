# import pytest
# import sys
# import copy
# import tempfile
# import os
# import mock
# import pkg_resources
# import yaml

# from ginjinn.commandline import main, commandline_app, argument_parser
# from ginjinn.commandline import splitter
# from ginjinn.commandline import simulate
# from ginjinn.commandline import train

# from ginjinn.simulation import generate_simple_shapes_coco, generate_simple_shapes_pvoc

# from detectron2.data import DatasetCatalog

# @pytest.fixture(scope='module', autouse=True)
# def tmp_dir():
#     tmpdir = tempfile.TemporaryDirectory()

#     yield tmpdir.name

#     tmpdir.cleanup()

# @pytest.fixture(scope='module')
# def simulate_coco(tmp_dir):
#     sim_dir = os.path.join(tmp_dir, 'sim_coco')
#     os.mkdir(sim_dir)

#     img_dir = os.path.join(sim_dir, 'images')
#     os.mkdir(img_dir)
#     ann_path = os.path.join(sim_dir, 'annotations.json')
#     generate_simple_shapes_coco(
#         img_dir=img_dir, ann_file=ann_path, n_images=40,
#     )
#     return img_dir, ann_path

# @pytest.fixture(scope='module')
# def simulate_pvoc(tmp_dir):
#     sim_dir = os.path.join(tmp_dir, 'sim_pvoc')
#     os.mkdir(sim_dir)

#     img_dir = os.path.join(sim_dir, 'images')
#     os.mkdir(img_dir)
#     ann_dir = os.path.join(sim_dir, 'annotations')
#     os.mkdir(ann_dir)

#     generate_simple_shapes_pvoc(
#         img_dir=img_dir, ann_dir=ann_dir, n_images=40,
#     )
#     return img_dir, ann_dir

# @pytest.fixture(scope='module', autouse=True)
# def example_config(tmp_dir, simulate_coco):
#     img_dir, ann_path = simulate_coco

#     example_config_1_path = pkg_resources.resource_filename(
#         'ginjinn', 'data/ginjinn_config/example_config_1.yaml',
#     )

#     with open(example_config_1_path) as config_f:
#         config = yaml.load(config_f)

#     config['input']['training']['annotation_path'] = ann_path
#     config['input']['training']['image_path'] = img_dir

#     config_dir = os.path.join(tmp_dir, 'example_config')
#     os.mkdir(config_dir)
#     config['project_dir'] = os.path.abspath(config_dir)

#     config_file = os.path.join(config_dir, 'ginjinn_config.yaml')
#     with open(config_file, 'w') as config_f:
#         yaml.dump(config, config_f)

#     return (config, config_file)

# @pytest.fixture(scope='module', autouse=True)
# def example_project(tmp_dir, example_config):
#     config, _ = example_config

#     project_dir = os.path.join(tmp_dir, 'example_project')
#     config['project_dir'] = project_dir
#     os.mkdir(project_dir)
#     os.mkdir(os.path.join(project_dir, 'outputs'))

#     config_file = os.path.join(project_dir, 'ginjinn_config.yaml')
#     with open(config_file, 'w') as config_f:
#         yaml.dump(config, config_f)
    
#     return project_dir

# @pytest.fixture(scope='module', autouse=True)
# def example_config_pvoc(tmp_dir, simulate_pvoc):
#     img_dir, ann_dir = simulate_pvoc

#     example_config_1_path = pkg_resources.resource_filename(
#         'ginjinn', 'data/ginjinn_config/example_config_1.yaml',
#     )

#     with open(example_config_1_path) as config_f:
#         config = yaml.load(config_f)

#     config['task'] = 'bbox-detection'

#     config['model']['name'] = 'faster_rcnn_R_50_FPN_3x'

#     config['input']['type'] = 'PVOC'
#     config['input']['training']['annotation_path'] = ann_dir
#     config['input']['training']['image_path'] = img_dir

#     config_dir = os.path.join(tmp_dir, 'example_config_pvoc')
#     os.mkdir(config_dir)
#     config['project_dir'] = os.path.abspath(config_dir)

#     config_file = os.path.join(config_dir, 'ginjinn_config.yaml')
#     with open(config_file, 'w') as config_f:
#         yaml.dump(config, config_f)

#     return (config, config_file)

# @pytest.fixture(scope='module', autouse=True)
# def example_project_pvoc(tmp_dir, example_config_pvoc):
#     config, _ = example_config_pvoc

#     project_dir = os.path.join(tmp_dir, 'example_project_pvoc')
#     config['project_dir'] = project_dir
#     os.mkdir(project_dir)
#     os.mkdir(os.path.join(project_dir, 'outputs'))

#     config_file = os.path.join(project_dir, 'ginjinn_config.yaml')
#     with open(config_file, 'w') as config_f:
#         yaml.dump(config, config_f)

#     return project_dir

# def test_main_simple(tmp_dir):
#     project_dir = os.path.join(tmp_dir, 'test_new_0')

#     tmp = copy.deepcopy(sys.argv)
#     sys.argv = ['ginjinn', 'new', project_dir]
#     main()
#     sys.argv = tmp

# def test_splitting(tmp_dir, simulate_coco):
#     img_dir, ann_path = simulate_coco


#     split_dir = os.path.join(tmp_dir, 'test_splitting_0')
#     os.mkdir(split_dir)

#     args = argument_parser.GinjinnArgumentParser().parse_args(
#         [
#             'split',
#             '-i', img_dir,
#             '-a', ann_path,
#             '-o', split_dir,
#             '-d', 'instance-segmentation',
#             '-k', 'COCO'
#         ]
#     )

#     def y_gen():
#         while True:
#             yield 'y'
#     y_it = y_gen()

#     def y(*args, **kwargs):
#         return next(y_it)
    
#     with mock.patch('builtins.input', y):
#         splitter.ginjinn_split(args)

#     with mock.patch('builtins.input', y):
#         splitter.ginjinn_split(args)
    
#     with mock.patch('builtins.input', lambda *args: 'n'):
#         splitter.ginjinn_split(args)

# def test_splitting_pvoc(tmp_dir, simulate_pvoc):
#     img_dir, ann_dir = simulate_pvoc


#     split_dir = os.path.join(tmp_dir, 'test_splitting_pvoc_0')
#     os.mkdir(split_dir)

#     args = argument_parser.GinjinnArgumentParser().parse_args(
#         [
#             'split',
#             '-i', img_dir,
#             '-a', ann_dir,
#             '-o', split_dir,
#             '-d', 'bbox-detection',
#             '-k', 'PVOC'
#         ]
#     )

#     def y_gen():
#         while True:
#             yield 'y'
#     y_it = y_gen()

#     def y(*args, **kwargs):
#         return next(y_it)
    
#     with mock.patch('builtins.input', y):
#         splitter.ginjinn_split(args)

#     with mock.patch('builtins.input', y):
#         splitter.ginjinn_split(args)
    
#     with mock.patch('builtins.input', lambda *args: 'n'):
#         splitter.ginjinn_split(args)

# def test_simulate(tmp_dir):
#     simulate_dir = os.path.join(tmp_dir, 'test_simulate_0')

#     args = argument_parser.GinjinnArgumentParser().parse_args(
#         [
#             'simulate',
#             'shapes',
#             '-o', simulate_dir,
#             '-n', '5',
#         ]
#     )

#     simulate.ginjinn_simulate(args)

#     with mock.patch('builtins.input', lambda *args: 'y'):
#         simulate.ginjinn_simulate(args)

# def test_simulate_pvoc(tmp_dir):
#     simulate_dir = os.path.join(tmp_dir, 'test_simulate_pvoc_0')

#     args = argument_parser.GinjinnArgumentParser().parse_args(
#         [
#             'simulate',
#             'shapes',
#             '-a', 'PVOC',
#             '-o', simulate_dir,
#             '-n', '5',
#         ]
#     )

#     simulate.ginjinn_simulate(args)

#     with mock.patch('builtins.input', lambda *args: 'y'):
#         simulate.ginjinn_simulate(args)

# def test_train(example_project):
#     project_dir = example_project
#     args = argument_parser.GinjinnArgumentParser().parse_args(
#         [
#             'train',
#             project_dir
#         ]
#     )

#     try:
#         DatasetCatalog.remove('train')
#     except:
#         pass
#     try:
#         DatasetCatalog.remove('val')
#     except:
#         pass

#     try:
#         train.ginjinn_train(args)
#     except AssertionError as err:
#         if 'NVIDIA driver' in str(err):
#             Warning(str(err))
#         else:
#             raise err
#     except Exception as err:
#         raise err
    
# def test_train_pvoc(example_project_pvoc):
#     project_dir = example_project_pvoc
#     args = argument_parser.GinjinnArgumentParser().parse_args(
#         [
#             'train',
#             project_dir
#         ]
#     )

#     try:
#         DatasetCatalog.remove('train')
#     except:
#         pass
#     try:
#         DatasetCatalog.remove('val')
#     except:
#         pass

#     try:
#         train.ginjinn_train(args)
#     except AssertionError as err:
#         if 'NVIDIA driver' in str(err):
#             Warning(str(err))
#         else:
#             raise err
#     except Exception as err:
#         raise err
