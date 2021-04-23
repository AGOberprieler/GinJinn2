'''GinjinnModelConfiguration test module
'''

import pytest
import tempfile
import os
from ginjinn.ginjinn_config.model_config import GinjinnModelConfiguration, MODELS
from ginjinn.ginjinn_config.config_error import InvalidModelConfigurationError

@pytest.fixture(scope='module', autouse=True)
def tmp_input_path():
    tmpdir = tempfile.TemporaryDirectory()

    tmp_file_path = os.path.join(tmpdir.name, 'weights.pkl')

    with open(tmp_file_path, 'w') as tmp_f:
        tmp_f.write('')

    yield tmp_file_path

    tmpdir.cleanup()

def test_simple_model(tmp_input_path):
    name = list(MODELS.keys())[0]
    weights = ''

    model = GinjinnModelConfiguration(
        name=name,
        weights=weights,
        classification_threshold=0.5,
    )

    assert model.name == name
    assert model.weights == weights

    weights = 'pretrained'
    model = GinjinnModelConfiguration(
        name=name,
        weights=weights,
        classification_threshold=0.5,
    )

    assert model.name == name
    assert model.weights == weights

def test_invalid_model():
    name = 'some_invalid_name'
    weights = ''

    with pytest.raises(InvalidModelConfigurationError):
        model = GinjinnModelConfiguration(
            name=name,
            weights = '',
            classification_threshold=0.5,
        )
    
    valid_name = list(MODELS.keys())[0]
    with pytest.raises(InvalidModelConfigurationError):
        GinjinnModelConfiguration(name=valid_name, weights='xyz', classification_threshold=0.5,)

    with pytest.raises(InvalidModelConfigurationError):
        GinjinnModelConfiguration(name=valid_name, weights='', classification_threshold=-0.1,)
    
    with pytest.raises(InvalidModelConfigurationError):
        GinjinnModelConfiguration(name=valid_name, weights='', classification_threshold=1.1,)
