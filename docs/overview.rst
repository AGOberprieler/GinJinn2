.. _overview:

Overview
========

After installation (:doc:`install instructions <installation>`), all GinJinn2 functionality can be accessed using the ``ginjinn`` commandline application.

Help
----

``ginjinn`` and all of its subcommands provide help pages, which can be displayed using the argument ``-h`` or ``--help``, e.g.
    
+  ``ginjinn -h`` (get list of all essential GinJinn commands)
+  ``ginjinn utils -h`` (get list of GinJinn's additional utilities)
+  ``ginjinn utils crop -h`` (get usage information for the cropping utility)

The help pages briefly describe basic functionality and command-specific arguments.

Dataset formats
---------------

A (labeled) input dataset should consist of a single image directory containing JPG images at its top level and accompanying annotations.
GinJinn2 supports two common annotation formats, `COCO's data format <https://cocodataset.org/#format-data>`_ (one JSON file per dataset), which is also used as output format, and XML files as used by `PASCAL VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ (one file per image).
The latter, however, is only supported for bounding-box object detection.

+ COCO
    .. code-block::

        data
        ├── annotations.json
        └── images
            ├── 1.jpg
            ├── 2.jpg
            └── ...

+ Pascal Voc
    .. code-block::

        data
        ├── annotations
        │   ├── 1.xml
        │   ├── 2.xml
        │   └── ...
        └── images
            ├── 1.jpg
            ├── 2.jpg
            └── ...
            
In case of nested image directories, ``ginjinn utils flatten`` helps to convert datasets to an accepted format.

Basic workflow
--------------

1.  ``ginjinn new``

    This command generates a new project directory, which is required for training, evaluation, and prediction.
    Initially, it only contains an empty output folder and the GinJinn2 configuration file ("ginjinn_config.yaml"), a simple, formatted text file for storing various settings, which can be edited by the user for customization. When executing certain GinJinn2 commands, further data may be written to the project directory.
    To avoid inconsistencies, it is strongly recommended to keep the configuration file fixed throughout subsequent steps of the analysis.
    The ``-t``/``--template`` and ``-d``/``--data_dir`` options allow to automatically specify various settings such that a valid configuration can be created without manually editing the configuration file.

2.  ``ginjinn train``

    This command trains the model and simultaneously evaluates it using the validation dataset, if available.
    Checkpoints are saved periodically at predefined intervalls.
    While the training is running, its progress can be most easily monitored via the "outputs/metrics.pdf" file in your project directory.
    If the training has finished or in case of interruption, it can be resumed with ``-r``/``--resume``.
    The number of iterations as specified in the configuration file can be overwritten with ``-n``/``--n_iter``.

3.  ``ginjinn evaluate``

    This command calculates [COCO evaluation metrics](https://cocodataset.org/#detection-eval) for a trained model using the test dataset.

4.  ``ginjinn predict``

    This command uses a trained model to predict objects for new, unlabeled images.
    It provides several optional types of output: a COCO annotation file, object visualization on the original images, and cropped images (segmentation masks) for each predicted object.


Concrete workflows including more complex examples are described in XYZ.
