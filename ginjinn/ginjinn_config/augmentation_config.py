'''
GinJinn augmentation configuration module
'''

from typing import List
import detectron2.data.transforms as T
from .config_error import InvalidAugmentationConfigurationError

def _check_probability(probability: float):
    '''Helper function to check augmentation probabilities

    Parameters
    ----------
    probability : float
        Augmentation probability

    Raises
    ------
    InvalidAugmentationConfigurationError
        Raised when an invalid probability value is passed.
    '''
    if probability < 0.0 or probability > 1.0:
        raise InvalidAugmentationConfigurationError(
            'The probability of an augmentation must be between 0.0 and 1.0.'
        )

class HorizontalFlipAugmentationConfiguration: #pylint: disable=too-few-public-methods
    '''Horizontal Flip Augmentation Configuration

    Parameters
    ----------
    probability : float, optional
        Probability of applying the augmentation, by default 1.0 (always applied).
    '''

    def __init__(self, probability: float = 1.0):
        _check_probability(probability)

        self.probability = probability

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build HorizontalFlipAugmentationConfiguration from dictionary

        Parameters
        ----------
        config : dict
            Dictionary containing horizontal flip configurations.

        Returns
        -------
        HorizontalFlipAugmentation
            HorizontalFlipAugmentation object.
        '''
        probability = config.get('probability', 1.0)
        return cls(probability = probability)

    def to_detectron2_augmentation(self):
        '''Convert to Detectron2 augmentation

        Returns
        -------
        Augmentation
            Detectron2 augmentation
        '''
        return T.RandomFlip(
            prob=self.probability,
            horizontal=True,
            vertical=False
        )

class VerticalFlipAugmentationConfiguration: #pylint: disable=too-few-public-methods
    '''Vertical Flip Augmentation Configuration

    Parameters
    ----------
    probability : float, optional
        Probability of applying the augmentation, by default 1.0 (always applied).
    '''

    def __init__(self, probability: float = 1.0):
        _check_probability(probability)

        self.probability = probability

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build VerticalFlipAugmentation from dictionary

        Parameters
        ----------
        config : dict
            Dictionary containing vertical flip configurations.

        Returns
        -------
        VerticalFlipAugmentation
            VerticalFlipAugmentation object.
        '''
        probability = config.get('probability', 1.0)
        return cls(probability = probability)

    def to_detectron2_augmentation(self):
        '''Convert to Detectron2 augmentation

        Returns
        -------
        Augmentation
            Detectron2 augmentation
        '''
        return T.RandomFlip(
            prob=self.probability,
            horizontal=False,
            vertical=True
        )

class BrightnessAugmentationConfiguration: #pylint: disable=too-few-public-methods
    '''Random Brightness Augmentation Configuration

    Parameters
    ----------
    brightness_min : float
        Relative minimal brightness
    brightness_max : float
        Relative maximal brightness
    probability : float, optional
        Probability of applying the augmentation, by default 1.0 (always applied).
    '''

    def __init__(
        self,
        brightness_min: float,
        brightness_max: float,
        probability: float = 1.0
    ):
        _check_probability(probability)

        self.probability = probability
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self._check_brightness()

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build BrightnessAugmentationConfiguration from dictionary

        Parameters
        ----------
        config : dict
            Dictionary containing brightness configurations.

        Returns
        -------
        BrightnessAugmentationConfiguration
            BrightnessAugmentationConfiguration object.

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised when an invalid config is passed.
        '''
        probability = config.get('probability', 1.0)
        brightness_min = config.get('brightness_min', None)
        brightness_max = config.get('brightness_max', None)
        if brightness_min is None:
            raise InvalidAugmentationConfigurationError(
                '"brightness_min" required but not in config dictionary'
            )
        if brightness_max is None:
            raise InvalidAugmentationConfigurationError(
                '"brightness_max" required but not in config dictionary'
            )

        return cls(
            brightness_min=brightness_min,
            brightness_max=brightness_max,
            probability = probability
        )

    def to_detectron2_augmentation(self):
        '''Convert to Detectron2 augmentation

        Returns
        -------
        Augmentation
            Detectron2 augmentation
        '''
        return T.RandomApply(
            T.RandomBrightness(
                intensity_min=self.brightness_min,
                intensity_max=self.brightness_max,
            ),
            prob=self.probability
        )

    def _check_brightness(self):
        '''Check brightness values for validity

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised if brightness values not valid
        '''
        if self.brightness_min <= 0:
            raise InvalidAugmentationConfigurationError(
                'brightness_min must greather than 0.'
            )
        if self.brightness_max <= 0:
            raise InvalidAugmentationConfigurationError(
                'brightness_max must greather than 0.'
            )

        if self.brightness_min > self.brightness_max:
            raise InvalidAugmentationConfigurationError(
                'brightness_min must the less than brightness_max'
            )

class ContrastAugmentationConfiguration: #pylint: disable=too-few-public-methods
    '''Random Contrast Augmentation Configuration

    Parameters
    ----------
    contrast_min : float
        Relative minimal contrast
    contrast_max : float
        Relative maximal contrast
    probability : float, optional
        Probability of applying the augmentation, by default 1.0 (always applied).
    '''

    def __init__(
        self,
        contrast_min: float,
        contrast_max: float,
        probability: float = 1.0
    ):
        _check_probability(probability)

        self.probability = probability
        self.contrast_min = contrast_min
        self.contrast_max = contrast_max
        self._check_contrast()

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build ContrastAugmentationConfiguration from dictionary

        Parameters
        ----------
        config : dict
            Dictionary containing contrast configurations.

        Returns
        -------
        ContrastAugmentationConfiguration
            ContrastAugmentationConfiguration object.

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised when an invalid config is passed.
        '''
        probability = config.get('probability', 1.0)
        contrast_min = config.get('contrast_min', None)
        contrast_max = config.get('contrast_max', None)
        if contrast_min is None:
            raise InvalidAugmentationConfigurationError(
                '"contrast_min" required but not in config dictionary'
            )
        if contrast_max is None:
            raise InvalidAugmentationConfigurationError(
                '"contrast_max" required but not in config dictionary'
            )

        return cls(
            contrast_min=contrast_min,
            contrast_max=contrast_max,
            probability = probability
        )

    def to_detectron2_augmentation(self):
        '''Convert to Detectron2 augmentation

        Returns
        -------
        Augmentation
            Detectron2 augmentation
        '''
        return T.RandomApply(
            T.RandomContrast(
                intensity_min=self.contrast_min,
                intensity_max=self.contrast_max,
            ),
            prob=self.probability
        )

    def _check_contrast(self):
        '''Check contrast values for validity

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised if contrast values not valid
        '''
        if self.contrast_min <= 0:
            raise InvalidAugmentationConfigurationError(
                'contrast_min must greather than 0.'
            )
        if self.contrast_max <= 0:
            raise InvalidAugmentationConfigurationError(
                'contrast_max must greather than 0.'
            )

        if self.contrast_min > self.contrast_max:
            raise InvalidAugmentationConfigurationError(
                'contrast_min must the less than contrast_max'
            )

class SaturationAugmentationConfiguration: #pylint: disable=too-few-public-methods
    '''Random Saturation Augmentation Configuration

    Parameters
    ----------
    saturation_min : float
        Relative minimal saturation
    saturation_max : float
        Relative maximal saturation
    probability : float, optional
        Probability of applying the augmentation, by default 1.0 (always applied).
    '''

    def __init__(
        self,
        saturation_min: float,
        saturation_max: float,
        probability: float = 1.0
    ):
        _check_probability(probability)

        self.probability = probability
        self.saturation_min = saturation_min
        self.saturation_max = saturation_max
        self._check_saturation()

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build SaturationAugmentationConfiguration from dictionary

        Parameters
        ----------
        config : dict
            Dictionary containing saturation configurations.

        Returns
        -------
        SaturationAugmentationConfiguration
            SaturationAugmentationConfiguration object.

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised when an invalid config is passed.
        '''
        probability = config.get('probability', 1.0)
        saturation_min = config.get('saturation_min', None)
        saturation_max = config.get('saturation_max', None)
        if saturation_min is None:
            raise InvalidAugmentationConfigurationError(
                '"saturation_min" required but not in config dictionary'
            )
        if saturation_max is None:
            raise InvalidAugmentationConfigurationError(
                '"saturation_max" required but not in config dictionary'
            )

        return cls(
            saturation_min=saturation_min,
            saturation_max=saturation_max,
            probability = probability
        )

    def to_detectron2_augmentation(self):
        '''Convert to Detectron2 augmentation

        Returns
        -------
        Augmentation
            Detectron2 augmentation
        '''
        return T.RandomApply(
            T.RandomSaturation(
                intensity_min=self.saturation_min,
                intensity_max=self.saturation_max,
            ),
            prob=self.probability
        )

    def _check_saturation(self):
        '''Check saturation values for validity

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised if saturation values not valid
        '''
        if self.saturation_min <= 0:
            raise InvalidAugmentationConfigurationError(
                'saturation_min must greather than 0.'
            )
        if self.saturation_max <= 0:
            raise InvalidAugmentationConfigurationError(
                'saturation_max must greather than 0.'
            )

        if self.saturation_min > self.saturation_max:
            raise InvalidAugmentationConfigurationError(
                'saturation_min must the less than saturation_max'
            )

class CropRelativeAugmentationConfiguration: #pylint: disable=too-few-public-methods
    '''Random Crop Augmentation Configuration

    Parameters
    ----------
    width : float
        Relative width of crop.
    height : float
        Relative height of crop.
    probability : float, optional
        Probability of applying the augmentation, by default 1.0 (always applied).
    '''

    def __init__(
        self,
        width: float,
        height: float,
        probability: float = 1.0
    ):
        _check_probability(probability)

        self.probability = probability
        self.width = width
        self.height = height
        self._check_wh()

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build CropRelativeAugmentationConfiguration from dictionary

        Parameters
        ----------
        config : dict
            Dictionary containing crop configurations.

        Returns
        -------
        CropRelativeAugmentationConfiguration
            CropRelativeAugmentationConfiguration object.

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised when an invalid config is passed.
        '''
        probability = config.get('probability', 1.0)
        width = config.get('width', None)
        height = config.get('height', None)
        if width is None:
            raise InvalidAugmentationConfigurationError(
                '"width" required but not in config dictionary'
            )
        if height is None:
            raise InvalidAugmentationConfigurationError(
                '"height" required but not in config dictionary'
            )

        return cls(
            width=width,
            height=height,
            probability = probability
        )

    def to_detectron2_augmentation(self):
        '''Convert to Detectron2 augmentation

        Returns
        -------
        Augmentation
            Detectron2 augmentation
        '''
        return T.RandomApply(
            T.RandomCrop(
                crop_type='relative',
                crop_size=(self.height, self.width),
            ),
            prob=self.probability
        )

    def _check_wh(self):
        '''Check width and height values for validity

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised if width or height values not valid
        '''
        if self.width <= 0.0 or self.width >= 1.0:
            raise InvalidAugmentationConfigurationError(
                'width must between 0.0 and 1.0 (exclusive).'
            )
        if self.height <= 0.0 or self.height >= 1.0:
            raise InvalidAugmentationConfigurationError(
                'height must between 0.0 and 1.0 (exclusive).'
            )

class CropAbsoluteAugmentationConfiguration: #pylint: disable=too-few-public-methods
    '''Random Crop Augmentation Configuration

    Parameters
    ----------
    width : int
        Absolute width of crop in pixel.
    height : int
        Absolute height of crop in pixel.
    probability : float, optional
        Probability of applying the augmentation, by default 1.0 (always applied).
    '''

    def __init__(
        self,
        width: int,
        height: int,
        probability: float = 1.0
    ):
        _check_probability(probability)

        self.probability = probability
        self.width = width
        self.height = height
        self._check_wh()

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build CropAbsoluteAugmentationConfiguration from dictionary

        Parameters
        ----------
        config : dict
            Dictionary containing crop configurations.

        Returns
        -------
        CropAbsoluteAugmentationConfiguration
            CropAbsoluteAugmentationConfiguration object.

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised when an invalid config is passed.
        '''
        probability = config.get('probability', 1.0)
        width = config.get('width', None)
        height = config.get('height', None)
        if width is None:
            raise InvalidAugmentationConfigurationError(
                '"width" required but not in config dictionary'
            )
        if height is None:
            raise InvalidAugmentationConfigurationError(
                '"height" required but not in config dictionary'
            )

        return cls(
            width=width,
            height=height,
            probability = probability
        )

    def to_detectron2_augmentation(self):
        '''Convert to Detectron2 augmentation

        Returns
        -------
        Augmentation
            Detectron2 augmentation
        '''
        return T.RandomApply(
            T.RandomCrop(
                crop_type='absolute',
                crop_size=(self.height, self.width),
            ),
            prob=self.probability
        )

    def _check_wh(self):
        '''Check width and height values for validity

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised if width or height values not valid
        '''
        if self.width <= 0:
            raise InvalidAugmentationConfigurationError(
                'width must be greater than 0.'
            )
        if self.height <= 0:
            raise InvalidAugmentationConfigurationError(
                'height must be greater than 0.'
            )

class RotationRangeAugmentationConfiguration(): #pylint: disable=too-few-public-methods
    '''Rotation range augmentation

    Rotate randomly in the interval between angle_min and angle_max.

    Parameters
    ----------
    angle_min: float
        Minimum angle of rotation.
    angle_max: float
        Maximum angle of rotation.
    expand: bool
        image should be resized to fit the rotated image, alternatively cropped.
        By default True (resized).
    probability : float, optional
        Probability of applying the augmentation, by default 1.0 (always applied).
    '''

    def __init__(
        self,
        angle_min: float,
        angle_max: float,
        expand: bool = True,
        probability: float = 1.0,
    ):
        _check_probability(probability)

        self.angle_min = angle_min
        self.angle_max = angle_max
        self.expand = expand
        self.probability = probability

        self._check_angles()

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build RotationRangeAugmentationConfiguration from dictionary

        Parameters
        ----------
        config : dict
            Dictionary containing rotation configurations.

        Returns
        -------
        RotationRangeAugmentationConfiguration
            RotationRangeAugmentationConfiguration object.

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised if required dictionary field is missing
        '''
        probability = config.get('probability', 1.0)
        expand = config.get('expand', True)
        angle_min = config.get('angle_min', None)
        angle_max = config.get('angle_max', None)

        if angle_min is None:
            raise InvalidAugmentationConfigurationError(
                '"angle_min" required but not in config dictionary'
            )
        if angle_max is None:
            raise InvalidAugmentationConfigurationError(
                '"angle_min" required but not in config dictionary'
            )

        return cls(
            angle_min=angle_min,
            angle_max=angle_max,
            expand=expand,
            probability = probability
        )

    def to_detectron2_augmentation(self):
        '''Convert to Detectron2 augmentation

        Returns
        -------
        Augmentation
            Detectron2 augmentation
        '''
        return T.RandomApply(
            T.RandomRotation(
                angle=(self.angle_min, self.angle_max),
                expand=self.expand,
                sample_style='range'
            ),
            prob=self.probability
        )

    def _check_angles(self):
        '''Check angles for validity

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised if angles are not valid
        '''

        if self.angle_min > self.angle_max:
            raise InvalidAugmentationConfigurationError(
                'angle_min must the less than angle_max'
            )

class RotationChoiceAugmentationConfiguration(): #pylint: disable=too-few-public-methods
    '''Rotation selection augmentation

    Rotate randomly in the interval between angle_min and angle_max.

    Parameters
    ----------
    angles: list
        list of angles from which a random one will be chosen for each rotation augmentation.
    expand: bool
        image should be resized to fit the rotated image, alternatively cropped.
        By default True (resized).
    probability : float, optional
        Probability of applying the augmentation, by default 1.0 (always applied).
    '''

    def __init__(
        self,
        angles: list,
        expand: bool = True,
        probability: float = 1.0,
    ):
        _check_probability(probability)

        self.angles = angles
        self.expand = expand
        self.probability = probability

        self._check_angles()

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build RotationChoiceAugmentationConfiguration from dictionary

        Parameters
        ----------
        config : dict
            Dictionary containing rotation configurations.

        Returns
        -------
        RotationChoiceAugmentationConfiguration
            RotationChoiceAugmentationConfiguration object.

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised if required dictionary field is missing
        '''
        probability = config.get('probability', 1.0)
        expand = config.get('expand', True)
        angles = config.get('angles', None)

        if angles is None:
            raise InvalidAugmentationConfigurationError(
                '"angles" required but not in config dictionary'
            )

        return cls(
            angles=angles,
            expand=expand,
            probability = probability
        )

    def to_detectron2_augmentation(self):
        '''Convert to Detectron2 augmentation

        Returns
        -------
        Augmentation
            Detectron2 augmentation
        '''
        return T.RandomApply(
            T.RandomRotation(
                angle=self.angles,
                expand=self.expand,
                sample_style='choice'
            ),
            prob=self.probability
        )

    def _check_angles(self):
        '''Check angles for validity

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised if angles are not valid
        '''
        if len(self.angles) < 1:
            raise InvalidAugmentationConfigurationError(
                'There must be at least one angle to chose from.'
            )

class GinjinnAugmentationConfiguration: #pylint: disable=too-few-public-methods
    '''A class representing GinJinn augmentation configurations.
    '''

    AVAILABLE_AUGMENTATIONS = {
        'horizontal_flip': HorizontalFlipAugmentationConfiguration,
        'vertical_flip': VerticalFlipAugmentationConfiguration,
        'rotation_range': RotationRangeAugmentationConfiguration,
        'rotation_choice': RotationChoiceAugmentationConfiguration,
        'brightness': BrightnessAugmentationConfiguration,
        'contrast': ContrastAugmentationConfiguration,
        'saturation': SaturationAugmentationConfiguration,
        'crop_relative': CropRelativeAugmentationConfiguration,
        'crop_absolute': CropAbsoluteAugmentationConfiguration,
    }

    def __init__(
        self,
        augmentations: list
    ):
        '''Class representing augmentation configurations

        Parameters
        ----------
        augmentations : list
            List of Augmentation objects.
        '''

        self.augmentations = augmentations
        self._check_augmentations()

    @classmethod
    def from_dictionaries(cls, augmentation_dicts: List[dict]):
        '''Build augmentations configuration from list of dictionaries.

        Each augmentation dictionary should consist of single key naming
        the augmentation that should be performed with the a corresponding
        value, which is again a dictionary, listing the augmentation options.

        The following is an example for a horizontal flip augmentation dict:
        {
            'horizontal_flip': {
                probability: 0.25
            }
        }

        Parameters
        ----------
        augmentation_dicts : list[dict]
            List of dictionaries describing augmentations.

        Returns
        -------
        GinjinnAugmentationConfiguration
            GinjinnAugmentationConfiguration object.

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised when an invalid augmentation name is passed.
        '''

        augmentations = []
        for aug_dict in augmentation_dicts:
            # we expect only 1 key, see from_dictionaries documentation
            aug_name = list(aug_dict.keys())[0]
            aug_constructor = cls.AVAILABLE_AUGMENTATIONS.get(aug_name, None)
            if aug_constructor is None:
                raise InvalidAugmentationConfigurationError(
                    'Unknown augmentation "{}".'.format(aug_name)
                )

            aug = aug_constructor.from_dictionary(aug_dict[aug_name])

            augmentations.append(aug)

        return cls(augmentations)

    def to_detectron2_augmentations(self):
        '''Convert to Detectron2 augmentation list

        Returns
        -------
        Augmentations
            A list of Detectron2 augmentations
        '''
        augmentations = []
        for aug in self.augmentations:
            augmentations.append(aug.to_detectron2_augmentation())

        return augmentations

    def _check_augmentations(self):
        '''Check augmentations for validity

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised when an invalid augmentation was found.
        '''

        # nothing to check if there are no augmentations
        if len(self.augmentations) == 0:
            return

        for aug in self.augmentations:
            if not any(
                [isinstance(aug, av_aug) for av_aug in self.AVAILABLE_AUGMENTATIONS.values()]
            ):
                raise InvalidAugmentationConfigurationError(
                    'Unknown augmentation class "{}".'.format(type(aug))
                )
