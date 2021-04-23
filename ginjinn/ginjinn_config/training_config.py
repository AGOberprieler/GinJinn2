'''
GinJinn training configuration module
'''

import copy
# from typing import Optional
from .config_error import InvalidTrainingConfigurationError

class GinjinnTrainingConfiguration: #pylint: disable=too-few-public-methods
    '''A class representing GinJinn training configurations.

    Parameters
    ----------
    learning_rate : float
        learning rate for model training.
    batch_size : int
        batch size for model training and evaluation.
    max_iter: int
        maximum number of training iterations.
    warmup_iter: int
        number of warmup iterations.
    momentum: float
        momentum for solver.
    eval_period: int
        evaluation period.
    checkpoint_period: int
        checkpoint period.
    '''
    def __init__( #pylint: disable=too-many-arguments
        self,
        learning_rate: float,
        batch_size: int,
        max_iter: int,
        warmup_iter: int = 1000,
        momentum: float = 0.9,
        eval_period: int = 0,
        checkpoint_period: int = 0,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.momentum = momentum
        self.eval_period = eval_period
        self.checkpoint_period = checkpoint_period

        self._check_config()

    def update_detectron2_config(self, cfg):
        '''update_detectron2_config

        Updates detectron2 config with the training configuration.

        Parameters
        ----------
        cfg
            Detectron2 configuration
        '''

        cfg.SOLVER.IMS_PER_BATCH = self.batch_size
        cfg.SOLVER.BASE_LR = self.learning_rate
        cfg.SOLVER.MAX_ITER = self.max_iter
        cfg.SOLVER.WARMUP_ITERS = self.warmup_iter
        cfg.SOLVER.MOMENTUM = self.momentum
        cfg.TEST.EVAL_PERIOD = self.eval_period
        cfg.SOLVER.CHECKPOINT_PERIOD = self.checkpoint_period

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build GinjinnTrainingConfiguration from a dictionary object.

        Parameters
        ----------
        config : dict
            Dictionary object containing the training configuration.

        Returns
        -------
        GinjinnTrainingConfiguration
            GinjinnTrainingConfiguration constructed with the configuration
            given in config.
        '''

        default_config = {
            'learning_rate': 0.001,
            'batch_size': 1,
            'max_iter': 40000,
            'warmup_iter': 1000,
            'momentum': 0.9,
            'eval_period': 0,
            'checkpoint_period': 5000,
        }

        # Maybe implement this more elegantly...
        default_config.update(config)
        config = copy.deepcopy(default_config)

        return cls(
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            max_iter=config['max_iter'],
            warmup_iter=config['warmup_iter'],
            momentum=config['momentum'],
            eval_period=config['eval_period'],
            checkpoint_period=config['checkpoint_period'],
        )

    def _check_learning_rate(self):
        ''' Check learning rate config

        Raises
        ------
        InvalidTrainingConfigurationError
            Raised for invalid learning rate values.
        '''
        if self.learning_rate < 0.0:
            raise InvalidTrainingConfigurationError(
                'learning_rate must be greater than 0'
            )

    def _check_batch_size(self):
        ''' Check batch size config

        Raises
        ------
        InvalidTrainingConfigurationError
            Raised for invalid batch size values.
        '''
        if self.batch_size < 1:
            raise InvalidTrainingConfigurationError(
                'batch_size must be greater than or equal to 1'
            )

    def _check_max_iter(self):
        ''' Check max iter config

        Raises
        ------
        InvalidTrainingConfigurationError
            Raised for invalid max iter values.
        '''
        if self.max_iter < 1:
            raise InvalidTrainingConfigurationError(
                'max_iter must be greater than or equal to 1'
            )

    def _check_warmup_iter(self):
        ''' Check warmup iter config

        Raises
        ------
        InvalidTrainingConfigurationError
            Raised for invalid warmup iter values.
        '''
        if self.warmup_iter < 1:
            raise InvalidTrainingConfigurationError(
                'warmup_iter must be greater than or equal to 1'
            )

    def _check_momentum(self):
        '''_check_momentum

        Raises
        ------
        InvalidTrainingConfigurationError
            Raised for invalid momentum value.
        '''

        if self.momentum < 0.0:
            raise InvalidTrainingConfigurationError(
                'momentum must be positive.'
            )

    def _check_eval_period(self):
        '''_check_eval_period

        Raises
        ------
        InvalidTrainingConfigurationError
            Raised for invalid eval_period value.
        '''

        if self.eval_period < 0:
            raise InvalidTrainingConfigurationError(
                'eval_period must be positive.'
            )

    def _check_checkpoint_period(self):
        '''_check_checkpoint_period

        Raises
        ------
        InvalidTrainingConfigurationError
            Raised for invalid checkpoint_period value.
        '''

        if self.eval_period < 0:
            raise InvalidTrainingConfigurationError(
                'checkpoint_period must be positive.'
            )

    def _check_config(self):
        ''' Check configs
        '''
        self._check_learning_rate()
        self._check_batch_size()
        self._check_max_iter()
        self._check_momentum()
        self._check_eval_period()
        self._check_checkpoint_period()
