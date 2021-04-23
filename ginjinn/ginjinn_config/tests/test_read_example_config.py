import pkg_resources
import yaml

def test_yaml_config_loading():
    '''test_yaml_config_loading

    Summary
    -------
    Test to check, whether the example config can be loaded using
    the yaml python package.
    '''
    example_config_0_path = pkg_resources.resource_filename('ginjinn', 'data/ginjinn_config/example_config_0.yaml')
    with open(example_config_0_path) as f:
        example_config_0 = yaml.safe_load(f)
