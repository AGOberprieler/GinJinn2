''' GinJinn

Summary
-------
GinJinn is an object detection framework for the detection of structures in
digitized herbarium specimens.
GinJinn includes bounding-box detection and segmentation functionality using
state of the art computer vision models.
Further, it includes functionality for loading und exporting data in most
of the common data formats.

References
----------
Ott, T., C. Palm, R. Vogt, and C. Oberprieler. 2020. GinJinn: An object-detection
pipeline for automated feature extraction from herbarium specimens.
Applications in Plant Sciences 2020 8(6).
'''
__version__ = '0.0.1'

import pkg_resources

# an example for loading package data
example_data_path = pkg_resources.resource_filename('ginjinn', 'data/example_data.txt')
with open(example_data_path) as f:
    example_data = f.read()
