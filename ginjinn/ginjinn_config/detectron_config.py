'''
Detectron2 configuration module
'''

# import copy
# from typing import Optional

import collections.abc

#source: https://stackoverflow.com/a/3233356/5665958
def update(a, b):
    '''update

    Update mappable object a with mappable object b.

    Parameters
    ----------
    a
        Mappable object (e.g. dict)
    b
        Mappable object (e.g. dict)

    Returns
    -------
    mappable
        Updated object a.
    '''
    for key, val in b.items():
        if isinstance(val, collections.abc.Mapping):
            a[key] = update(a.get(key, {}), val)
        else:
            a[key] = val
    return a

class GinjinnDetectronConfiguration: #pylint: disable=too-few-public-methods
    '''A class representing additional Detectron2 configurations

    Parameters
    ----------
    config : dict, optional
        A dictionary describing additional Detectron2 configurations, by default {}
    '''

    def __init__(self, config: dict = {}): #pylint: disable=dangerous-default-value
        self.config = config

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build GinjinnAugmentationConfiguration from a dictionary object.

        Parameters
        ----------
        config : dict
            Dictionary object containing the augmentation configuration.

        Returns
        -------
        GinjinnDetectronConfiguration
            GinjinnDetectronConfiguration constructed with the configuration
            given in config.
        '''

        return cls(config)

    def update_detectron2_config(self, cfg):
        '''update_detectron2_config

        Updates detectron2 config with the detectron2 extra configuration.

        Parameters
        ----------
        cfg
            Detectron2 configuration
        '''

        update(cfg, self.config)
