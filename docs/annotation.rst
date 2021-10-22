.. _annotation:

Dataset Annotation
==================

GinJinn2 supports `PVOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ and `COCO <https://cocodataset.org/#format-data>`_ image annotation formats.
The COCO format allows annotations of bounding box and segmentations, while PVOC can only encode bounding boxes.
Hence, we recommend to prepare annotations in COCO format, if possible.

Many of the publically available annotation tools support at least one of those formats.
The following is a non-exhaustive list of software for this purpose:

- `CVAT <https://github.com/openvinotoolkit/cvat>`_
- `COCO Annotator <https://github.com/jsbroks/coco-annotator>`_
- `Labelbox <https://labelbox.com>`_
- `LabelImg <https://github.com/tzutalin/labelImg>`_
- `VIA <https://www.robots.ox.ac.uk/~vgg/software/via/>`_

We recommend the use of `CVAT <https://github.com/openvinotoolkit/cvat>`_, since this software outputs COCO annotations that are relatively strictly following the COCO dataset specification and are compatible with other tools working with COCO formatted datasets.

