'''
GinJinn options configuration module
'''

import copy
import os
# from typing import Optional
from .config_error import InvalidOptionsConfigurationError

N_CORES = os.cpu_count()

class GinjinnOptionsConfiguration: #pylint: disable=too-few-public-methods
    '''A class representing GinJinn model configurations.

    Parameters
    ----------
    resume : bool
        Determines, whether a previous run should be resumed
    n_threads : int
        Number of CPU threads to use.
    device: str
        Device to run the model on. E.g. "cuda", "cpu". 
    '''
    def __init__( #pylint: disable=too-many-arguments
        self,
        resume: bool,
        n_threads: int,
        device: str,
    ):
        self.resume = resume
        self.n_threads = n_threads
        self.device = device

        self._check_config()

    def update_detectron2_config(self, cfg):
        '''update_detectron2_config

        Updates detectron2 config with the options configuration.

        Parameters
        ----------
        cfg
            Detectron2 configuration
        '''

        cfg.DATALOADER.NUM_WORKERS = self.n_threads
        cfg.MODEL.DEVICE = self.device

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build GinjinnOptionsConfiguration from a dictionary object.

        Parameters
        ----------
        config : dict
            Dictionary object containing the options configuration.

        Returns
        -------
        GinjinnOptionsConfiguration
            GinjinnOptionsConfiguration constructed with the configuration
            given in config.
        '''

        default_config = {
            'resume': False,
            'n_threads': N_CORES - 1 if N_CORES > 1 else N_CORES,
            'device': 'cuda',
        }

        # Maybe implement this more elegantly...
        default_config.update(config)
        config = copy.deepcopy(default_config)

        return cls(
            resume=config['resume'],
            n_threads=config['n_threads'],
            device=config['device'],
        )

    def _check_n_threads(self):
        ''' Check n_threads config

        Raises
        ------
        InvalidOptionsConfigurationError
            Raised if n_threads value is invalid.
        '''
        if self.n_threads < 1:
            raise InvalidOptionsConfigurationError(
                'n_threads must be a positive number.'
            )

    def _check_device(self):
        '''_check_device

        Raises
        ------
        InvalidOptionsConfigurationError
            Raised if device value is invalid.
        '''
        err_msg = f'Invalid device option "{self.device}"; device should be either "cpu", ' +\
            '"cuda", or "cpu:" or "cuda:" followed by the respective device number (e.g "cuda:0").'

        if self.device in ['cpu', 'cuda']:
            return

        if self.device.startswith('cpu:') or self.device.startswith('cuda:'):
            try:
                int(self.device.split(':')[1])
            except Exception as err:
                raise InvalidOptionsConfigurationError(err_msg) from err
        else:
            raise InvalidOptionsConfigurationError(err_msg)


    def _check_config(self):
        ''' Check configuration values for validity.
        '''
        self._check_n_threads()
        self._check_device()
