''' Module to test augmentation_config.py
'''

import pytest

from ginjinn.ginjinn_config import GinjinnAugmentationConfiguration, InvalidAugmentationConfigurationError
from ginjinn.ginjinn_config.augmentation_config import HorizontalFlipAugmentationConfiguration, \
    VerticalFlipAugmentationConfiguration, \
    RotationRangeAugmentationConfiguration, \
    RotationChoiceAugmentationConfiguration, \
    BrightnessAugmentationConfiguration, \
    ContrastAugmentationConfiguration, \
    SaturationAugmentationConfiguration, \
    CropRelativeAugmentationConfiguration, \
    CropAbsoluteAugmentationConfiguration

@pytest.fixture
def simple_augmentation_list():
    return (
        [
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
            {
                'rotation_range': {
                    'angle_min': -10,
                    'angle_max': 10,
                    'expand': True,
                    'probability': 0.25
                }
            },
            {
                'rotation_choice': {
                    'angles': [
                        -10,
                        -20,
                        10,
                        20,
                    ],
                    'expand': True,
                    'probability': 0.25
                }
            },
            {
                'brightness': {
                    'brightness_min': 0.5,
                    'brightness_max': 1.5,
                    'probability': 0.75
                }
            },
            {
                'contrast': {
                    'contrast_min': 0.5,
                    'contrast_max': 1.5,
                    'probability': 0.75
                }
            },
            {
                'saturation': {
                    'saturation_min': 0.5,
                    'saturation_max': 1.5,
                    'probability': 0.75
                }
            },
            {
                'crop_relative': {
                    'width': 0.75,
                    'height': 0.75,
                    'probability': 0.3
                }
            },
            {
                'crop_absolute': {
                    'width': 128,
                    'height': 128,
                    'probability': 0.3
                }
            },
        ],
        [
            HorizontalFlipAugmentationConfiguration,
            VerticalFlipAugmentationConfiguration,
            RotationRangeAugmentationConfiguration,
            RotationChoiceAugmentationConfiguration,
            BrightnessAugmentationConfiguration,
            ContrastAugmentationConfiguration,
            SaturationAugmentationConfiguration,
            CropRelativeAugmentationConfiguration,
            CropAbsoluteAugmentationConfiguration,
        ]
    )

@pytest.fixture
def invalid_augmentation_list():
    return [
        {
            'invalid_augmentation': {
                'probability': 0.25
            }
        },
    ]

def test_simple(simple_augmentation_list):
    aug = GinjinnAugmentationConfiguration.from_dictionaries(
        simple_augmentation_list[0]
    )

    assert len(aug.augmentations) == len(simple_augmentation_list[0])
    # assert isinstance(aug.augmentations[0], simple_augmentation_list[1][0])
    assert aug.augmentations[0].probability == simple_augmentation_list[0][0]['horizontal_flip']['probability']
    # assert isinstance(aug.augmentations[1], simple_augmentation_list[1][1])
    assert aug.augmentations[1].probability == simple_augmentation_list[0][1]['vertical_flip']['probability']

    for i, aug_conf in enumerate(aug.augmentations):
        assert isinstance(aug_conf, simple_augmentation_list[1][i])

def test_invalid_aug_name(invalid_augmentation_list):
    with pytest.raises(InvalidAugmentationConfigurationError):
        aug = GinjinnAugmentationConfiguration.from_dictionaries(invalid_augmentation_list)

def test_invalid_aug_class():
    with pytest.raises(InvalidAugmentationConfigurationError):
        aug = GinjinnAugmentationConfiguration([{}, {}])

def test_empty_aug():
    aug_1 = GinjinnAugmentationConfiguration.from_dictionaries([])
    assert len(aug_1.augmentations) == 0

    aug_2 = GinjinnAugmentationConfiguration([])
    assert len(aug_2.augmentations) == 0

def test_invalid_probability():
    with pytest.raises(InvalidAugmentationConfigurationError):
        aug = GinjinnAugmentationConfiguration.from_dictionaries([
            {'horizontal_flip': {
                'probability': -0.1
            }}
        ])

    with pytest.raises(InvalidAugmentationConfigurationError):
        aug = GinjinnAugmentationConfiguration.from_dictionaries([
            {'horizontal_flip': {
                'probability': 1.1
            }}
        ])

def test_invalid_rotation_range():
    with pytest.raises(InvalidAugmentationConfigurationError):
        RotationRangeAugmentationConfiguration.from_dictionary(
            {
                'angle_min': 11,
                'angle_max': 10,
                'expand': True,
                'probability': 0.25
            }
        )
    with pytest.raises(InvalidAugmentationConfigurationError):
        RotationRangeAugmentationConfiguration.from_dictionary(
            {
                'angle_min': -10,
                'expand': True,
                'probability': 0.25
            }
        )
    with pytest.raises(InvalidAugmentationConfigurationError):
        RotationRangeAugmentationConfiguration.from_dictionary(
            {
                'angle_max': 20,
                'expand': True,
                'probability': 0.25
            }
        )
    with pytest.raises(InvalidAugmentationConfigurationError):
        RotationChoiceAugmentationConfiguration.from_dictionary(
            {
                'angles': [],
                'expand': True,
                'probability': 0.25
            }
        )
    
    with pytest.raises(InvalidAugmentationConfigurationError):
        RotationChoiceAugmentationConfiguration.from_dictionary(
            {
                'expand': True,
                'probability': 0.25
            }
        )
    
    with pytest.raises(InvalidAugmentationConfigurationError):
        BrightnessAugmentationConfiguration.from_dictionary(
            {
                'brightness_min': 0.1,
                'probability': 0.25
            }
        )
    
    with pytest.raises(InvalidAugmentationConfigurationError):
        BrightnessAugmentationConfiguration.from_dictionary(
            {
                'brightness_max': 0.1,
                'probability': 0.25
            }
        )
    
    with pytest.raises(InvalidAugmentationConfigurationError):
        BrightnessAugmentationConfiguration.from_dictionary(
            {
                'brightness_min': 0.2,
                'brightness_max': 0.1,
                'probability': 0.25
            }
        )
    
    with pytest.raises(InvalidAugmentationConfigurationError):
        BrightnessAugmentationConfiguration.from_dictionary(
            {
                'brightness_min': -0.1,
                'brightness_max': 0.1,
                'probability': 0.25
            }
        )
    
    with pytest.raises(InvalidAugmentationConfigurationError):
        BrightnessAugmentationConfiguration.from_dictionary(
            {
                'brightness_min': 0.1,
                'brightness_max': -0.1,
                'probability': 0.25
            }
        )

    with pytest.raises(InvalidAugmentationConfigurationError):
        ContrastAugmentationConfiguration.from_dictionary(
            {
                'contrast_min': 0.1,
                'probability': 0.25
            }
        )
    
    with pytest.raises(InvalidAugmentationConfigurationError):
        ContrastAugmentationConfiguration.from_dictionary(
            {
                'contrast_max': 0.1,
                'probability': 0.25
            }
        )
    
    with pytest.raises(InvalidAugmentationConfigurationError):
        ContrastAugmentationConfiguration.from_dictionary(
            {
                'contrast_min': 0.2,
                'contrast_max': 0.1,
                'probability': 0.25
            }
        )
    
    with pytest.raises(InvalidAugmentationConfigurationError):
        ContrastAugmentationConfiguration.from_dictionary(
            {
                'contrast_min': -0.1,
                'contrast_max': 0.1,
                'probability': 0.25
            }
        )
    
    with pytest.raises(InvalidAugmentationConfigurationError):
        ContrastAugmentationConfiguration.from_dictionary(
            {
                'contrast_min': 0.1,
                'contrast_max': -0.1,
                'probability': 0.25
            }
        )

    with pytest.raises(InvalidAugmentationConfigurationError):
        SaturationAugmentationConfiguration.from_dictionary(
            {
                'saturation_min': 0.1,
                'probability': 0.25
            }
        )
    
    with pytest.raises(InvalidAugmentationConfigurationError):
        SaturationAugmentationConfiguration.from_dictionary(
            {
                'saturation_max': 0.1,
                'probability': 0.25
            }
        )
    
    with pytest.raises(InvalidAugmentationConfigurationError):
        SaturationAugmentationConfiguration.from_dictionary(
            {
                'saturation_min': 0.2,
                'saturation_max': 0.1,
                'probability': 0.25
            }
        )
    
    with pytest.raises(InvalidAugmentationConfigurationError):
        SaturationAugmentationConfiguration.from_dictionary(
            {
                'saturation_min': -0.1,
                'saturation_max': 0.1,
                'probability': 0.25
            }
        )
    
    with pytest.raises(InvalidAugmentationConfigurationError):
        SaturationAugmentationConfiguration.from_dictionary(
            {
                'saturation_min': 0.1,
                'saturation_max': -0.1,
                'probability': 0.25
            }
        )

    with pytest.raises(InvalidAugmentationConfigurationError):
        CropRelativeAugmentationConfiguration.from_dictionary(
            {
                'width': 0.7,
                'probability': 0.25
            }
        )
    
    with pytest.raises(InvalidAugmentationConfigurationError):
        CropRelativeAugmentationConfiguration.from_dictionary(
            {
                'height': 0.7,
                'probability': 0.25
            }
        )

    with pytest.raises(InvalidAugmentationConfigurationError):
        CropRelativeAugmentationConfiguration.from_dictionary(
            {'width': 0.7, 'height': 0.0, 'probability': 0.25}
        )
    
    with pytest.raises(InvalidAugmentationConfigurationError):
        CropRelativeAugmentationConfiguration.from_dictionary(
            {'width': 0.7, 'height': 1.1, 'probability': 0.25}
        )
    
    with pytest.raises(InvalidAugmentationConfigurationError):
        CropRelativeAugmentationConfiguration.from_dictionary(
            {'width': 0.0, 'height': 0.7, 'probability': 0.25}
        )

    with pytest.raises(InvalidAugmentationConfigurationError):
        CropRelativeAugmentationConfiguration.from_dictionary(
            {'width': 1.1, 'height': 0.7, 'probability': 0.25}
        )

    with pytest.raises(InvalidAugmentationConfigurationError):
        CropRelativeAugmentationConfiguration.from_dictionary(
            {'width': 1.1, 'height': 1.1, 'probability': 0.25}
        )

    with pytest.raises(InvalidAugmentationConfigurationError):
        CropAbsoluteAugmentationConfiguration.from_dictionary(
            {'width': 128, 'probability': 0.25}
        )

    with pytest.raises(InvalidAugmentationConfigurationError):
        CropAbsoluteAugmentationConfiguration.from_dictionary(
            {'height': 128, 'probability': 0.25}
        )

    with pytest.raises(InvalidAugmentationConfigurationError):
        CropAbsoluteAugmentationConfiguration.from_dictionary(
            {'width': 128, 'height': 0, 'probability': 0.25}
        )

    with pytest.raises(InvalidAugmentationConfigurationError):
        CropAbsoluteAugmentationConfiguration.from_dictionary(
            {'width': 0, 'height': 128, 'probability': 0.25}
        )


def test_detectron2_conversion(simple_augmentation_list):
    aug = GinjinnAugmentationConfiguration.from_dictionaries(
        simple_augmentation_list[0]
    )

    d_augs = aug.to_detectron2_augmentations()

    assert d_augs[0].prob == simple_augmentation_list[0][0]['horizontal_flip']['probability']
    assert d_augs[0].horizontal == True
    assert d_augs[0].vertical == False

    assert d_augs[1].prob == simple_augmentation_list[0][1]['vertical_flip']['probability']
    assert d_augs[1].horizontal == False
    assert d_augs[1].vertical == True

    assert d_augs[2].prob == simple_augmentation_list[0][2]['rotation_range']['probability']
    assert d_augs[2].aug.angle[0] == simple_augmentation_list[0][2]['rotation_range']['angle_min']
    assert d_augs[2].aug.angle[1] == simple_augmentation_list[0][2]['rotation_range']['angle_max']
    assert d_augs[2].aug.expand == simple_augmentation_list[0][2]['rotation_range']['expand']

    assert d_augs[3].prob == simple_augmentation_list[0][3]['rotation_choice']['probability']
    l1 = len(d_augs[3].aug.angle)
    l2 = len(simple_augmentation_list[0][3]['rotation_choice']['angles'])
    assert l1 == l2
    for a1, a2 in zip(
        d_augs[3].aug.angle,
        simple_augmentation_list[0][3]['rotation_choice']['angles']
    ):
        assert a1 == a2

    assert d_augs[3].aug.expand == simple_augmentation_list[0][3]['rotation_choice']['expand']

    assert d_augs[4].prob == simple_augmentation_list[0][4]['brightness']['probability']
    assert d_augs[4].aug.intensity_min == simple_augmentation_list[0][4]['brightness']['brightness_min']
    assert d_augs[4].aug.intensity_max == simple_augmentation_list[0][4]['brightness']['brightness_max']

    assert d_augs[5].prob == simple_augmentation_list[0][5]['contrast']['probability']
    assert d_augs[5].aug.intensity_min == simple_augmentation_list[0][5]['contrast']['contrast_min']
    assert d_augs[5].aug.intensity_max == simple_augmentation_list[0][5]['contrast']['contrast_max']

    assert d_augs[6].prob == simple_augmentation_list[0][6]['saturation']['probability']
    assert d_augs[6].aug.intensity_min == simple_augmentation_list[0][6]['saturation']['saturation_min']
    assert d_augs[6].aug.intensity_max == simple_augmentation_list[0][6]['saturation']['saturation_max']

    assert d_augs[7].prob == simple_augmentation_list[0][7]['crop_relative']['probability']
    assert d_augs[7].aug.crop_size[0] == simple_augmentation_list[0][7]['crop_relative']['height']
    assert d_augs[7].aug.crop_size[1] == simple_augmentation_list[0][7]['crop_relative']['width']
    assert d_augs[7].aug.crop_type == 'relative'

    assert d_augs[8].prob == simple_augmentation_list[0][8]['crop_absolute']['probability']
    assert d_augs[8].aug.crop_size[0] == simple_augmentation_list[0][8]['crop_absolute']['height']
    assert d_augs[8].aug.crop_size[1] == simple_augmentation_list[0][8]['crop_absolute']['width']
    assert d_augs[8].aug.crop_type == 'absolute'
