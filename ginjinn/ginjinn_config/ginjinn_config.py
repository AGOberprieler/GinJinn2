'''
A module for managing the representation of GinJinn configurations.
'''


# import copy
# from typing import Optional
import yaml
import os
from .config_error import InvalidGinjinnConfigurationError
from .input_config import GinjinnInputConfiguration
from .model_config import GinjinnModelConfiguration, MODELS
from .augmentation_config import GinjinnAugmentationConfiguration
from .detectron_config import GinjinnDetectronConfiguration
from .options_config import GinjinnOptionsConfiguration
from .training_config import GinjinnTrainingConfiguration

TASKS = [
    'bbox-detection',
    # 'semantic-segmentation',
    'instance-segmentation',
]

class GinjinnConfiguration: #pylint: disable=too-many-arguments,too-many-instance-attributes
    '''GinJinn configuration class.

    A class representing the configuration of a GinJinn project.

    Parameters
    ----------
    project_dir : str
        Project directory. All outputs will be written to this directory.
    task : str
        Object detection task type.
    input_configuration : GinjinnInputConfiguration
        Object describing the input.
    model_configuration : GinjinnModelConfiguration
        Object describing the model.
    training_configuration : GinjinnTrainingConfiguration
        Object desribing the training.
    augmentation_configuration : GinjinnAugmentationConfiguration
        Object describing the augmentation.
    detectron_configuration : GinjinnDetectronConfiguration
        Object describing additional detectron2 configurations.
        Only use this option if you know what you are doing
    options_configuration: GinjinnOptionsConfiguration
        Object describing additional GinJinn options.

    Raises
    ------
    InvalidGinjinnConfigurationError
        If any of the general configuration is contradictionary or malformed.
    '''
    def __init__(
        self,
        project_dir: str,
        task: str,
        input_configuration: GinjinnInputConfiguration,
        model_configuration: GinjinnModelConfiguration,
        training_configuration: GinjinnTrainingConfiguration,
        augmentation_configuration: GinjinnAugmentationConfiguration,
        detectron_configuration: GinjinnDetectronConfiguration = GinjinnDetectronConfiguration(),
        options_configuration: GinjinnOptionsConfiguration =
            GinjinnOptionsConfiguration.from_dictionary({}),
    ):
        self.project_dir = project_dir
        self.task = task
        self.input = input_configuration
        self.model = model_configuration
        self.training = training_configuration
        self.augmentation = augmentation_configuration
        self.detectron_config = detectron_configuration
        self.options = options_configuration

        self._check()

    def to_detectron2_config(self, is_test: bool = False):
        '''to_detectron2_config

        Convert GinJinn configuration to Detectron2 configuration.

        Parameters
        ----------
        is_test: bool
            Whether current function call is in context of a test setting.

        Returns
        -------
        detectron2_config
            Detectron2 configuration.
        '''

        # model
        config = self.model.to_detectron2_config()

        # input
        self.input.update_detectron2_config(config, is_test=is_test)

        # TODO:
        # training
        self.training.update_detectron2_config(config)
        # options, TODO: implement additional options
        self.options.update_detectron2_config(config)
        # extra detectron config TODO
        self.detectron_config.update_detectron2_config(config)

        # detectron2 output dir
        config.OUTPUT_DIR = os.path.join(self.project_dir, 'outputs')

        # maybe remove this
        print(config)

        return config

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build GinjinnConfiguration from dictionary.

        Parameters
        ----------
        config : dict
            Dictionary object describing the GinJinn configuration.

        Returns
        -------
        GinjinnConfiguration
            GinjinnConfiguration constructed with the configuration
            given in config.
        '''

        project_dir = config['project_dir']

        input_configuration = GinjinnInputConfiguration.from_dictionary(
            config['input'],
            project_dir=project_dir,
        )
        model_configuration = GinjinnModelConfiguration.from_dictionary(
            config['model']
        )
        training_configuration = GinjinnTrainingConfiguration.from_dictionary(
            config.get('training', {})
        )
        augmentation_configuration = GinjinnAugmentationConfiguration.from_dictionaries(
            config.get('augmentation', [])
        )
        detectron_configuration = GinjinnDetectronConfiguration.from_dictionary(
            config.get('detectron', {})
        )
        options_configuration = GinjinnOptionsConfiguration.from_dictionary(
            config.get('options', {})
        )

        return cls(
            project_dir=project_dir,
            task=config['task'],
            input_configuration=input_configuration,
            model_configuration=model_configuration,
            training_configuration=training_configuration,
            augmentation_configuration=augmentation_configuration,
            detectron_configuration=detectron_configuration,
            options_configuration=options_configuration,
        )

    @classmethod
    def from_config_file(cls, file_path: str):
        '''Build GinjinnConfiguration from YAML configuration file.

        Parameters
        ----------
        file_path : str
            Path to GinJinn YAML configuration file.

        Returns
        -------
        GinjinnConfiguration
            GinjinnConfiguration constructed with the configuration
            given in the config file.
        '''

        with open(file_path) as config_file:
            config = yaml.safe_load(config_file)

        return cls.from_dictionary(config)

    def _check(self):
        '''_check

        Check validity of configuration options.

        Raises
        ------
        InvalidGinjinnConfigurationError
            Raised if invalid task passed.
        InvalidGinjinnConfigurationError
            Raised if task incompatible with model.
        '''
        if not self.task in TASKS:
            raise InvalidGinjinnConfigurationError(
                '"task" must be one of {}'.format(TASKS)
            )

        model_tasks = MODELS[self.model.name]['tasks']
        if not self.task in model_tasks:
            err_msg = f'Task "{self.task}" is incompatible with model ' +\
                f'"{self.model.name}" (available tasks: {", ".join(model_tasks)}).'
            raise InvalidGinjinnConfigurationError(err_msg)
