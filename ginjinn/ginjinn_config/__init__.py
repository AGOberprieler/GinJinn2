'''
A module for managing the representation of GinJinn configurations.
'''

from .ginjinn_config import GinjinnConfiguration
from .input_config import GinjinnInputConfiguration
from .model_config import GinjinnModelConfiguration
from .augmentation_config import GinjinnAugmentationConfiguration
from .config_error import \
    InvalidGinjinnConfigurationError,\
    InvalidInputConfigurationError, \
    InvalidAugmentationConfigurationError
