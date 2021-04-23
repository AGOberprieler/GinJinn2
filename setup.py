''' Setup script for GinJinn
'''

import re
import glob
from setuptools import setup, find_packages


# get version from __init__.py file
with open('ginjinn/__init__.py', 'r') as f:
    VERSION = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        f.read(),
        re.M
    ).group(1)

DESCRIPTION = 'An object detection pipeline for the extraction of structures from herbarium specimens.'
AUTHOR = 'Tankred Ott, Ulrich Lautenschlager'
AUTHOR_EMAIL = 'tankred.ott@ur.de, ulrich.lautenschlager@ur.de'

def install_requires():
    '''Get requirements from requirements.txt'''
    # with open('requirements.txt') as f:
    #     return f.read().splitlines()
    return []

setup(
    name='ginjinn',
    version=VERSION,
    url='https://github.com/AGOberprieler/ginjinn',
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=install_requires(),
    entry_points={
        'console_scripts': [
            'ginjinn = ginjinn.commandline.main:main',
        ]
    },
    package_data={
        'ginjinn': [
            'data/ginjinn_config/template_config.yaml',
            'data/example_data.txt',
            'data/ginjinn_config/example_config_0.yaml',
            'data/ginjinn_config/example_config_1.yaml',
            'data/ginjinn_config/templates',
            'data/ginjinn_config/templates/*'
        ],
    }
)
