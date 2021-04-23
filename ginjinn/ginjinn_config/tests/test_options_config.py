''' Module for testing options_config.py
'''

import pytest
from ginjinn.ginjinn_config.options_config import GinjinnOptionsConfiguration
from ginjinn.ginjinn_config.config_error import InvalidOptionsConfigurationError

def test_simple():
    n_threads=1
    resume=True
    device='cuda'

    options_0 = GinjinnOptionsConfiguration(
        resume=resume,
        n_threads=n_threads,
        device=device,
    )
    assert options_0.resume == resume
    assert options_0.n_threads == n_threads
    assert options_0.device == device

    options_1 = GinjinnOptionsConfiguration.from_dictionary({
        'n_threads': n_threads,
        'resume': resume,
    })
    assert options_1.resume == resume
    assert options_1.n_threads == n_threads
    assert options_1.device == device

    options_2 = GinjinnOptionsConfiguration(
        resume=resume,
        n_threads=n_threads,
        device='cpu',
    )
    assert options_2.resume == resume
    assert options_2.n_threads == n_threads
    assert options_2.device == 'cpu'

    GinjinnOptionsConfiguration(
        resume=resume,
        n_threads=n_threads,
        device='cpu:0',
    )

    GinjinnOptionsConfiguration(
        resume=resume,
        n_threads=n_threads,
        device='cuda:0',
    )

def test_invalid():
    n_threads=0
    resume=True
    device='cuda'

    with pytest.raises(InvalidOptionsConfigurationError):
        GinjinnOptionsConfiguration(
            resume=resume,
            n_threads=n_threads,
            device=device,
        )

    with pytest.raises(InvalidOptionsConfigurationError):
        GinjinnOptionsConfiguration(
            resume=resume,
            n_threads=2,
            device='',
        )
    
    with pytest.raises(InvalidOptionsConfigurationError):
        GinjinnOptionsConfiguration(
            resume=resume,
            n_threads=2,
            device='cuda:x',
        )
