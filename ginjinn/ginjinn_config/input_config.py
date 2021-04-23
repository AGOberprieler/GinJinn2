'''
GinJinn input configuration module
'''

import copy
import os
from typing import Optional
from .config_error import InvalidInputConfigurationError

ANNOTATION_TYPES = [
    'PVOC',
    'COCO'
]

class InputPaths: #pylint: disable=too-few-public-methods
    '''Class representing annotation and corresponding image paths.

    Parameters
    ----------
    ann_path : str
        Path to annotations. I.e. either a file or a folder path.
    img_path : str
        Path to the folder containing images.
    '''
    def __init__(
        self,
        ann_path: str,
        img_path: str,
    ):
        self.annotation_path = ann_path
        self.image_path = img_path

# TODO recycle this for the commandline script
class SplitConfig: #pylint: disable=too-few-public-methods
    '''Class representing test and validation split options.

    Parameters
    ----------
    test_split : float
        Fraction of data set to use for testing.
    validation_split : float
        Fraction of data set to use for validation.
    '''
    def __init__(
        self,
        test_split: Optional[float] = None,
        validation_split: Optional[float] = None,
    ):
        self.test = test_split
        self.val = validation_split

        self._check()

    def _check(self):
        ''' Checks validity of splitting.

        Raises
        ------
        InvalidInputConfigurationError
            Raised in case of invalid splitting options.
        '''
        if not self.test is None:
            if self.test <= 0.0 or self.test >= 1.0:
                raise InvalidInputConfigurationError(
                    'The proportion of the test split must be greater than 0.0 and less than 1.0.' #pylint: disable=line-too-long
                )
        if not self.val is None:
            if self.val <= 0.0 or self.val >= 1.0:
                raise InvalidInputConfigurationError(
                    'The proportion of the validation split must be greater than 0.0 and less than 1.0.' #pylint: disable=line-too-long
                )
        if (not self.test is None) and (not self.val is None):
            proportion = self.test + self.val
            if proportion >= 1.0 or proportion <= 0.0:
                raise InvalidInputConfigurationError(
                    'The sum of test and validation split proportions must be greater than 0.0 and less than 1.0.' #pylint: disable=line-too-long
                )

class GinjinnInputConfiguration: #pylint: disable=too-few-public-methods
    '''GinJinn input configuration class.

    A class representing the configuration of the input(s)
    for a GinJinn project. This includes the configuration
    or description of optionals train-validation-test
    splits of the data set.

    Train-validation-test can be
    - skipped, when leaving test_* and val_* arguments at default
    - custom, when specifying test_* and val_* arguments

    Parameters
    ----------
    ann_type : str
        Type of the object detection annotations.
        "PascalVOC" or "COCO".
    train_ann_path : str
        Path to the directory containing annotations files for "PascalVOC".
        Path to the annotation file for "COCO".
    train_img_path : str
        Path to the directory containing the images.
    test_ann_path : Optional[str], optional
        Path to the directory containing annotations files for "PascalVOC".
        Path to the annotation file for "COCO".
    test_img_path : Optional[str], optional
        Path to the directory containing the images.
    val_ann_path : Optional[str], optional
        Path to the directory containing annotations files for "PascalVOC".
        Path to the annotation file for "COCO".
    val_img_path : Optional[str], optional
        Path to the directory containing the images.
    project_dir : str
        GinJinn project directory.

    Raises
    ------
    InvalidInputConfigurationError
        If the input configuration is contradictionary or malformed.
    '''

    def __init__( #pylint: disable=too-many-arguments
        self,
        ann_type: str,
        train_ann_path: str,
        train_img_path: str,
        test_ann_path: Optional[str] = None,
        test_img_path: Optional[str] = None,
        val_ann_path: Optional[str] = None,
        val_img_path: Optional[str] = None,
        project_dir: str = '',
    ):
        self.project_dir = project_dir
        self.type = ann_type
        self.train = InputPaths(
            self._rel_to_project(train_ann_path),
            self._rel_to_project(train_img_path)
        )

        self.test = None
        self.val = None



        # type
        if not self.type in ANNOTATION_TYPES:
            raise InvalidInputConfigurationError(
                '"ann_type" must be one of {}.'.format(ANNOTATION_TYPES)
            )

        # test
        if (not test_ann_path is None) or (not test_img_path is None):
            if (test_ann_path is None) or (test_img_path is None):
                raise InvalidInputConfigurationError(
                    'If any of "test_ann_path" and "test_img_path" is passed, ' \
                    'the other must be passed too.'
                )
            self.test = InputPaths(
                self._rel_to_project(test_ann_path),
                self._rel_to_project(test_img_path),
            )

        # validation
        if (not val_ann_path is None) or (not val_img_path is None):
            if (val_ann_path is None) or (val_img_path is None):
                raise InvalidInputConfigurationError(
                    'If any of "val_ann_path" and "val_img_path" is passed, ' \
                    'the other must be passed too.'
                )
            self.val = InputPaths(
                self._rel_to_project(val_ann_path),
                self._rel_to_project(val_img_path),
            )

        # check for file path validity
        # TODO: think about whether this should be checked here or later
        #       in the data reader.
        self._check_filepaths()

    def update_detectron2_config(self, cfg, is_test: bool=False):
        '''update_detectron2_config

        Updates detectron2 config with the input configuration.

        Parameters
        ----------
        cfg
            Detectron2 configuration
        is_test: bool
            Whether current function call is in context of a test setting.
        '''
        if not is_test:
            if self.train:
                cfg.DATASETS.TRAIN = ('train', )
            if self.val:
                cfg.DATASETS.TEST = ('val', )
            else:
                cfg.DATASETS.TEST = ()
        else:
            if self.test:
                cfg.DATASETS.TEST = ('test', )

    @staticmethod
    def _check_pvoc_annotation_path(ann_path: str):
        ''' Check for PVOC annotation path validity, else raise an exception

        Parameters
        ----------
        ann_path : str
            Path to a directory containing annotations.

        Raises
        ------
        InvalidInputConfigurationError
            This exception is raised if the annotation path is not valid.
        '''
        if not os.path.isdir(ann_path):
            raise InvalidInputConfigurationError(
                '"{}" is not a valid PVOC annotation path. The path might not exist ' \
                'or refer to a file instead of a directory.'.format(ann_path)
            )

    @staticmethod
    def _check_coco_annotation_path(ann_path: str):
        ''' Check for COCO annotation path validity, else raise an exception

        Parameters
        ----------
        ann_path : str
            Path to an annotation JSON file.

        Raises
        ------
        InvalidInputConfigurationError
            This exception is raised if the annotation path is not valid.
        '''

        if not os.path.isfile(ann_path):
            raise InvalidInputConfigurationError(
                '"{}" is not a valid COCO annotation file path. The path might not exist ' \
                'or refer to a directory instead of a file.'.format(ann_path)
            )

    @staticmethod
    def _check_image_path(image_path: str):
        ''' Check for image path validity, else raise an exception

        Parameters
        ----------
        image_path : str
            Path to a directory containing images.

        Raises
        ------
        InvalidInputConfigurationError
            This exception is raised if the image path is not valid.
        '''
        if not os.path.isdir(image_path):
            raise InvalidInputConfigurationError(
                '"{}" is not a valid image directory path. The path might not exist ' \
                'or refer to a file.'.format(image_path)
            )

    def _check_filepaths(self):
        '''Check, whether file path configuration is valid
        '''

        # check for correct annotation type, i.e. files or folders
        if self.type == 'PVOC':
            self._check_pvoc_annotation_path(self.train.annotation_path)
            if not self.test is None:
                self._check_pvoc_annotation_path(self.test.annotation_path)
            if not self.val is None:
                self._check_pvoc_annotation_path(self.val.annotation_path)
        elif self.type == 'COCO':
            self._check_coco_annotation_path(self.train.annotation_path)
            if not self.test is None:
                self._check_coco_annotation_path(self.test.annotation_path)
            if not self.val is None:
                self._check_coco_annotation_path(self.val.annotation_path)

        # check if image directory exists
        self._check_image_path(self.train.image_path)
        if not self.test is None:
            self._check_image_path(self.test.image_path)
        if not self.val is None:
            self._check_image_path(self.val.image_path)

    @classmethod
    def from_dictionary(cls, config: dict, project_dir: str =''):
        '''Build GinjinnInputConfiguration from a dictionary object.

        Parameters
        ----------
        config : dict
            Dictionary object containing the input configuration.
        project_dir : str
            GinJinn project directory.
        Returns
        -------
        GinjinnInputConfiguration
            GinjinnInputConfiguration constructed with the configuration
            given in config.
        '''

        default_config = {
            'test': {
                'annotation_path': None,
                'image_path': None,
            },
            'validation': {
                'annotation_path': None,
                'image_path': None,
            },
            'split': {
                'test': None,
                'validation': None
            }
        }

        # Maybe implement this more elegantly...
        default_config.update(config)
        config = copy.deepcopy(default_config)

        return cls(
            ann_type = config['type'],
            train_ann_path = config['training']['annotation_path'],
            train_img_path = config['training']['image_path'],
            test_ann_path = config['test']['annotation_path'],
            test_img_path = config['test']['image_path'],
            val_ann_path = config['validation']['annotation_path'],
            val_img_path = config['validation']['image_path'],
            project_dir = project_dir,
        )

    def _rel_to_project(self, file_path: str) -> str:
        '''_rel_to_project

        Set root of relative file path to self.project_dir instead
        of the current shell root.

        Parameters
        ----------
        file_path : str
            File path to correct.

        Returns
        -------
        str
            Corrected file path
        '''

        if os.path.isabs(file_path):
            return file_path

        return os.path.abspath(
            os.path.join(self.project_dir, file_path)
        )
