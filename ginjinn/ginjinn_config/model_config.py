'''
GinJinn model configuration module
'''

import copy
import os
from typing import List
from .config_error import InvalidModelConfigurationError

class AnchorGeneratorConfig: #pylint: disable=too-few-public-methods
    '''AnchorGeneratorConfig

    Object representing AnchorGenerator model configurations.

    Parameters
    ----------
    sizes : List[List[float]]
        List of lists of anchor sizes in absolute pixels.
    aspect_ratios : List[List[float]]
        List of lists of anchor aspect ratios.
    angles : List[List[float]]
        List of lists of anchor angles.
    '''
    def __init__(
        self,
        sizes : List[List[float]] = None,
        aspect_ratios : List[List[float]] = None,
        angles : List[List[float]] = None,
    ):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.angles = angles

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build AnchorGeneratorConfig from a dictionary object.

        Parameters
        ----------
        config : dict
            Dictionary object containing the AnchorGenerator configuration.

        Returns
        -------
        AnchorGeneratorConfig
            AnchorGeneratorConfig constructed with the parameters in config.

        Raises
        ------
        InvalidModelConfigurationError
            Raised if unknown parameter in config dict.
        '''

        available_configs = ['sizes', 'aspect_ratios', 'angles']
        for cfg_name in config.keys():
            if not cfg_name in available_configs:
                err_msg = f'Unknown anchor generator parameter name "{cfg_name}". ' +\
                    f'Available parameters are {available_configs}.'
                raise InvalidModelConfigurationError(err_msg)

        default_config = {
        }

        default_config.update(config)
        config = copy.deepcopy(default_config)

        return cls(
            sizes=config.get('sizes', None),
            aspect_ratios=config.get('aspect_ratios', None),
            angles=config.get('angles', None),
        )

    def update_detectron2_config(self, cfg):
        '''update_detectron2_config

        Updates detectron2 config with the AnchorGenerator configuration.

        Parameters
        ----------
        cfg
            Detectron2 configuration

        '''

        if self.sizes:
            cfg.MODEL.ANCHOR_GENERATOR.SIZES = self.sizes
        if self.aspect_ratios:
            cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = self.aspect_ratios
        if self.angles:
            cfg.MODEL.ANCHOR_GENERATOR.ANGLES = self.angles

class RPNConfig: #pylint: disable=too-few-public-methods
    '''RPNConfig

    Object representing RPN model configurations.

    Parameters
    ----------
    iou_thresholds : list
        Background and foreground thresholds for the IoU.
    batch_size_per_image : int
        Number of regions per image used to train the RPN.
    '''
    def __init__(
        self,
        iou_thresholds : list = None,
        batch_size_per_image : int = None,
    ):
        self.iou_thresholds = iou_thresholds
        self.batch_size_per_image = batch_size_per_image

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build RPNConfig from a dictionary object.

        Parameters
        ----------
        config : dict
            Dictionary object containing the RPNC configuration.

        Returns
        -------
        RPNConfig
            RPNConfig constructed with the parameters in config.

        Raises
        ------
        InvalidModelConfigurationError
            Raised if unknown parameter in config dict.
        '''

        available_configs = ['iou_thresholds', 'batch_size_per_image']
        for cfg_name in config.keys():
            if not cfg_name in available_configs:
                err_msg = f'Unknown anchor generator parameter name "{cfg_name}". ' +\
                    f'Available parameters are {available_configs}.'
                raise InvalidModelConfigurationError(err_msg)

        default_config = {
        }

        default_config.update(config)
        config = copy.deepcopy(default_config)

        return cls(
            iou_thresholds=config.get('iou_thresholds', None),
            batch_size_per_image=config.get('batch_size_per_image', None),
        )

    def update_detectron2_config(self, cfg):
        '''update_detectron2_config

        Updates detectron2 config with the RPN configuration.

        Parameters
        ----------
        cfg
            Detectron2 configuration

        '''

        if self.iou_thresholds:
            cfg.MODEL.RPN.IOU_THRESHOLDS = self.iou_thresholds
        if self.batch_size_per_image:
            cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = self.batch_size_per_image

class ROIHeadsConfig: #pylint: disable=too-few-public-methods
    '''ROIHeadsConfig

    Object representing ROIHeads model configurations.

    Parameters
    ----------
    iou_thresholds : List[float]
        Overlap thresholds for an RoI to be considered foreground, by default None
    batch_size_per_image : int, optional
        Number of RoIs per image, by default None
    '''

    def __init__(
        self,
        iou_thresholds: List[float] = None,
        batch_size_per_image: int = None,
    ):
        self.iou_thresholds = iou_thresholds
        self.batch_size_per_image = batch_size_per_image

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build ROIHeadsConfig from a dictionary object.

        Parameters
        ----------
        config : dict
            Dictionary object containing the ROIHeads configuration.

        Returns
        -------
        ROIHeadsConfig
            ROIHeadsConfig constructed with the parameters in config.

        Raises
        ------
        InvalidModelConfigurationError
            Raised if unknown parameter in config dict.
        '''

        available_configs = ['iou_thresholds', 'batch_size_per_image']
        for cfg_name in config.keys():
            if not cfg_name in available_configs:
                err_msg = f'Unknown roi heads parameter name "{cfg_name}". ' +\
                    f'Available parameters are {available_configs}.'
                raise InvalidModelConfigurationError(err_msg)

        default_config = {
        }

        if config.get('iou_thresholds', False):
            if not isinstance(config['iou_thresholds'], list):
                config['iou_thresholds'] = [config['iou_thresholds']]

        default_config.update(config)
        config = copy.deepcopy(default_config)

        return cls(
            iou_thresholds=config.get('iou_thresholds', None),
            batch_size_per_image=config.get('batch_size_per_image', None),
        )

    def update_detectron2_config(self, cfg):
        '''update_detectron2_config

        Updates detectron2 config with the ROIHeads configuration.

        Parameters
        ----------
        cfg
            Detectron2 configuration

        '''

        if self.iou_thresholds:
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = self.iou_thresholds
        if self.batch_size_per_image:
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.batch_size_per_image


class BoxHeadConfig: #pylint: disable=too-few-public-methods
    '''BoxHeadConfig

    Object representing BoxHead model configurations.

    Parameters
    ----------
    class_agnostic : bool
        Whether to apply class agnostic bbox regression.
    train_on_pred_boxes : bool
        Whether to use predicted BoxHead boxes instead of proposal boxes.
    '''
    def __init__(
        self,
        class_agnostic : bool = None,
        train_on_pred_boxes : bool = None,
    ):
        self.class_agnostic = class_agnostic
        self.train_on_pred_boxes = train_on_pred_boxes

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build BoxHeadConfig from a dictionary object.

        Parameters
        ----------
        config : dict
            Dictionary object containing the BoxHead configuration.

        Returns
        -------
        BoxHeadConfig
            BoxHeadConfig constructed with the parameters in config.

        Raises
        ------
        InvalidModelConfigurationError
            Raised if unknown parameter in config dict.
        '''

        available_configs = ['class_agnostic', 'train_on_pred_boxes']
        for cfg_name in config.keys():
            if not cfg_name in available_configs:
                err_msg = f'Unknown anchor generator parameter name "{cfg_name}". ' +\
                    f'Available parameters are {available_configs}.'
                raise InvalidModelConfigurationError(err_msg)

        default_config = {
        }

        default_config.update(config)
        config = copy.deepcopy(default_config)

        return cls(
            class_agnostic=config.get('class_agnostic', None),
            train_on_pred_boxes=config.get('train_on_pred_boxes', None),
        )

    def update_detectron2_config(self, cfg):
        '''update_detectron2_config

        Updates detectron2 config with the BoxHead configuration.

        Parameters
        ----------
        cfg
            Detectron2 configuration

        '''

        if self.class_agnostic:
            cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = self.class_agnostic
        if self.train_on_pred_boxes:
            cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = self.train_on_pred_boxes

class MaskHeadConfig: #pylint: disable=too-few-public-methods
    '''MaskHeadConfig

    Object representing MaskHead model configurations.

    Parameters
    ----------
    class_agnostic : bool
        Whether to apply class agnostic mask regression.
    pooler_resolution : int
        Resolution of the pooling.
    '''
    def __init__(
        self,
        class_agnostic : bool = None,
        pooler_resolution : int = None,
    ):
        self.class_agnostic = class_agnostic
        self.pooler_resolution = pooler_resolution

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build MaskHeadConfig from a dictionary object.

        Parameters
        ----------
        config : dict
            Dictionary object containing the MaskHead configuration.

        Returns
        -------
        MaskHeadConfig
            MaskHeadConfig constructed with the parameters in config.

        Raises
        ------
        InvalidModelConfigurationError
            Raised if unknown parameter in config dict.
        '''

        available_configs = ['class_agnostic', 'pooler_resolution']
        for cfg_name in config.keys():
            if not cfg_name in available_configs:
                err_msg = f'Unknown anchor generator parameter name "{cfg_name}". ' +\
                    f'Available parameters are {available_configs}.'
                raise InvalidModelConfigurationError(err_msg)

        default_config = {
        }

        default_config.update(config)
        config = copy.deepcopy(default_config)

        return cls(
            class_agnostic=config.get('class_agnostic', None),
            pooler_resolution=config.get('pooler_resolution', None),
        )

    def update_detectron2_config(self, cfg):
        '''update_detectron2_config

        Updates detectron2 config with the BoxHead configuration.

        Parameters
        ----------
        cfg
            Detectron2 configuration

        '''

        if self.class_agnostic:
            cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = self.class_agnostic
        if self.pooler_resolution:
            cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = self.pooler_resolution

# see all models: detectron2.model_zoo.model_zoo._ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX
MODELS = {
    'faster_rcnn_R_50_C4_1x': {
        'config_file': 'COCO-Detection/faster_rcnn_R_50_C4_1x.yaml',
        'tasks': ['bbox-detection'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'box_head': BoxHeadConfig,
        },
    },
    'faster_rcnn_R_50_DC5_1x': {
        'config_file': 'COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml',
        'tasks': ['bbox-detection'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'box_head': BoxHeadConfig,
        },
    },
    'faster_rcnn_R_50_FPN_1x': {
        'config_file': 'COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml',
        'tasks': ['bbox-detection'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'box_head': BoxHeadConfig,
        },
    },
    'faster_rcnn_R_50_C4_3x': {
        'config_file': 'COCO-Detection/faster_rcnn_R_50_C4_3x.yaml',
        'tasks': ['bbox-detection'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'box_head': BoxHeadConfig,
        },
    },
    'faster_rcnn_R_50_DC5_3x': {
        'config_file': 'COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml',
        'tasks': ['bbox-detection'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'box_head': BoxHeadConfig,
        },
    },
    'faster_rcnn_R_50_FPN_3x': {
        'config_file': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
        'tasks': ['bbox-detection'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'box_head': BoxHeadConfig,
        },
    },
    'faster_rcnn_R_101_C4_3x': {
        'config_file': 'COCO-Detection/faster_rcnn_R_101_C4_3x.yaml',
        'tasks': ['bbox-detection'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'box_head': BoxHeadConfig,
        },
    },
    'faster_rcnn_R_101_DC5_3x': {
        'config_file': 'COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml',
        'tasks': ['bbox-detection'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'box_head': BoxHeadConfig,
        },
    },
    'faster_rcnn_R_101_FPN_3x': {
        'config_file': 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
        'tasks': ['bbox-detection'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'box_head': BoxHeadConfig,
        },
    },
    'faster_rcnn_X_101_32x8d_FPN_3x': {
        'config_file': 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
        'tasks': ['bbox-detection'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'box_head': BoxHeadConfig,
        },
    },
    'mask_rcnn_R_50_C4_1x': {
        'config_file': 'COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml',
        'tasks': ['instance-segmentation'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'mask_head': MaskHeadConfig,
        },
    },
    'mask_rcnn_R_50_DC5_1x': {
        'config_file': 'COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml',
        'tasks': ['instance-segmentation'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'mask_head': MaskHeadConfig,
        },
    },
    'mask_rcnn_R_50_FPN_1x': {
        'config_file': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml',
        'tasks': ['instance-segmentation'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'mask_head': MaskHeadConfig,
        },
    },
    'mask_rcnn_R_50_C4_3x': {
        'config_file': 'COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml',
        'tasks': ['instance-segmentation'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'mask_head': MaskHeadConfig,
        },
    },
    'mask_rcnn_R_50_DC5_3x': {
        'config_file': 'COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml',
        'tasks': ['instance-segmentation'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'mask_head': MaskHeadConfig,
        },
    },
    'mask_rcnn_R_50_FPN_3x': {
        'config_file': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
        'tasks': ['instance-segmentation'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'mask_head': MaskHeadConfig,
        },
    },
    'mask_rcnn_R_101_C4_3x': {
        'config_file': 'COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml',
        'tasks': ['instance-segmentation'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'mask_head': MaskHeadConfig,
        },
    },
    'mask_rcnn_R_101_DC5_3x': {
        'config_file': 'COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml',
        'tasks': ['instance-segmentation'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'mask_head': MaskHeadConfig,
        },
    },
    'mask_rcnn_R_101_FPN_3x': {
        'config_file': 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
        'tasks': ['instance-segmentation'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'mask_head': MaskHeadConfig,
        },
    },
    'mask_rcnn_X_101_32x8d_FPN_3x': {
        'config_file': 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml',
        'tasks': ['instance-segmentation'],
        'model_parameters': {
            'anchor_generator': AnchorGeneratorConfig,
            'rpn': RPNConfig,
            'roi_heads': ROIHeadsConfig,
            'mask_head': MaskHeadConfig,
        },
    },
}

MODEL_PARAMETERS_MAP = {
    'anchor_generator': AnchorGeneratorConfig,
    'rpn': RPNConfig,
    'roi_heads': ROIHeadsConfig,
    'box_head': BoxHeadConfig,
    'mask_head': MaskHeadConfig,
}

class GinjinnModelConfiguration: #pylint: disable=too-few-public-methods
    '''A class representing GinJinn model configurations.

    Parameters
    ----------
    name : str
        model name/identifier.
    weights : str
        Determines the initialization of the model weights.
        One of
            - '', meaning random weights initialization
            - 'pretrained', meaning pretrained weights from the Detectron2 model zoo, if available
            - the file path of a weights file.
    classification_threshold: float
        Classification threshold for training.
    model_parameters: dict
        dict of model-specific parameters.

    Raises
    ------
    InvalidModelConfigurationError
        Raised if invalid model name is passed.
    '''
    def __init__( #pylint: disable=too-many-arguments,dangerous-default-value
        self,
        name: str,
        weights: str,
        classification_threshold: float,
        model_parameters: dict = {},
    ):
        self.name = name
        if not name in MODELS.keys():
            raise InvalidModelConfigurationError('Invalid model name.')

        self.detectron2_config_name = MODELS[self.name]['config_file']

        self.weights = weights
        self.classification_threshold = classification_threshold

        self.model_parameters = model_parameters

        self._check_config()

    def to_detectron2_config(self):
        '''to_detectron2_config

        Convert model configuration to Detectron2 configuration.

        Returns
        -------
        detectron2_config
            Detectron2 configuration.
        '''

        # import here to reduce loading times, when detectron2 conversion is not
        # required
        from detectron2.config import get_cfg #pylint: disable=import-outside-toplevel
        from detectron2.model_zoo import get_config_file, get_checkpoint_url #pylint: disable=import-outside-toplevel

        cfg = get_cfg()
        model_config_file = get_config_file(self.detectron2_config_name)
        cfg.merge_from_file(model_config_file)

        if self.weights == 'pretrained':
            model_url = get_checkpoint_url(self.detectron2_config_name)
            cfg.MODEL.WEIGHTS = model_url
        else:
            cfg.MODEL.WEIGHTS = self.weights

        for model_param in self.model_parameters.values():
            model_param.update_detectron2_config(cfg)

        return cfg

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build GinjinnModelConfiguration from a dictionary object.

        Parameters
        ----------
        config : dict
            Dictionary object containing the model configuration.

        Returns
        -------
        GinjinnModelConfiguration
            GinjinnModelConfiguration constructed with the configuration
            given in config.

        Raises
        ------
        InvalidModelConfigurationError
            Raised if unknown model_parameters entry in config dict.
        '''

        default_config = {
            'weights': '',
            'classification_threshold': 0.5,
            'model_parameters': {},
        }

        # Maybe implement this more elegantly...
        default_config.update(config)
        config = copy.deepcopy(default_config)

        # model parameters
        for mp_name, mp_cfg in config['model_parameters'].items():
            mp_class = MODEL_PARAMETERS_MAP.get(mp_name, None)
            if not mp_class:
                err_msg = f'Unknown model_parameters entry "{mp_name}". Available ' +\
                    f'model_parameters are {", ".join(MODEL_PARAMETERS_MAP.keys())}.'
                raise InvalidModelConfigurationError(err_msg)

            config['model_parameters'][mp_name] = mp_class.from_dictionary(mp_cfg)

        return cls(
            name=config['name'],
            weights=config['weights'],
            classification_threshold=config['classification_threshold'],
            model_parameters=config['model_parameters'],
        )

    def _check_weights(self):
        '''Check weights option

        Raises
        ------
        InvalidModelConfigurationError
            Raised if an invalid weights option is passed.
        '''

        if self.weights != '' and self.weights != 'pretrained':
            if not os.path.isfile(self.weights):
                raise InvalidModelConfigurationError(
                    'weights must be either "", "pretrained", or a valid weights file path.' #pylint: disable=line-too-long
                )

    def _check_classification_threshold(self):
        '''_check_classification_threshold

        Raises
        ------
        InvalidModelConfigurationError
            Raised if an invalid classification_threshold option is passed.
        '''
        if self.classification_threshold <= 0.0 or self.classification_threshold > 1.0:
            raise InvalidModelConfigurationError(
                'classification_threshold must be between 0.0 and 1.0.'
            )

    def _check_model_parameters(self):
        '''_check_model_parameters

        Raises
        ------
        InvalidModelConfigurationError
            Raised if an invalid model_parameters option is passed.
        '''

        for name, cfg in self.model_parameters.items():
            model_par_class = MODELS[self.name]['model_parameters'].get(name)

            if not model_par_class:
                err_msg = f'Unknown model_parameters entry "{name}". Availabe model_parameters ' +\
                    f'are {", ".join(MODELS[self.name]["model_parameters"].keys())}'
                raise InvalidModelConfigurationError(err_msg)

            expected_class = MODELS[self.name]['model_parameters'][name]
            if not isinstance(cfg, expected_class):
                err_msg = f'Model parameter "{name}" must be instance of class {expected_class}.'

    def _check_config(self):
        '''_check_config

        Check model configuration.
        '''
        self._check_weights()
        self._check_classification_threshold()
        self._check_model_parameters()
