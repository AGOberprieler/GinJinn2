.. _4-toolbox:

Toolbox
=======

The essential tools for working with `Detectron2 <https://github.com/facebookresearch/detectron2>`_'s object detection models, i.e. ``ginjinn new``, ``ginjinn train``, ``ginjinn evaluate``, and ``ginjinn predict``, have already been introduced under :doc:`Getting Started <3-getting_started>`.
Here, we give an overview over additional functionality provided by GinJinn2.
For command-specific details, please also have a look at the corresponding help pages.

Object Cropping
---------------

Sometimes it is necessary to crop objects from the images on which they are annotated.
GinJinn2 provides such functionality in two ways:

#.  Arbitrary COCO datasets consisting of an annotation file and an image directory can be cropped using the ``ginjinn utils crop`` command. In this case, a new annotation file referring to the cropped sub-images will be created as well.
#.  When predicting from a trained object detection model with ``ginjinn predict``, cropped objects (sub-images only) can be saved directly using the ``-c/--crop`` option.

In either case, rectangular bounding boxes, each enclosing an object, are cropped from the original images.
If you want to crop instance segmentations (``ginjinn utils crop --type segmentation ...``), GinJinn2 ignores the "bbox" entries from the annotation file and recalculates bounding boxes for cropping based on the "segmentation" entries instead.

Both ``ginjinn utils crop`` and ``ginjinn predict`` allow to extend the cropping range beyond the bounding box of an object via the ``--padding`` option. If, for instance, the cropped images are intended to be used to train a model, it may be advisable to prevent objects from touching the image borders. To completely disable padding, simply specify ``--padding 0``.

Dataset Filtering
-----------------

GinJinn2 offers two commands for dataset filtering:

#.  ``ginjinn utils filter_cat``: Filter objects by their category.
#.  ``ginjinn utils filter_size``: Filter objects (or parts of them) by size.

ginjinn utils filter_cat
^^^^^^^^^^^^^^^^^^^^^^^^

For the simulated dataset created in :ref:`Getting Started <3-getting_started_sim>`, the following command could be used to create a new annotation file with only one object category "circle" (``-f circle``):

.. code-block:: bash

    ginjinn utils filter_cat \
        -a shapes_ds/annotations.json \
        -o shapes_ds_filtered \
        -f circle

Here, ``-a shapes_ds/annotations.json`` denotes the annotation file to filtered while ``-o shapes_ds_filtered`` sets the output directory.
To specify multiple categories, the ``-f`` option has to be used repeatedly (e.g. ``-f circle -f triangle``).
``-d/--drop`` would invert the default behavior and discard the specified categories instead of keeping them. 

If you explicitely specify a directory comprising the annotated images (``-i shapes_ds/images``), a new, filtered image directory will be created as well.
In example above, the latter would only contain images with at least one annotated circle.
Note that, in order to save disk space, these filtered images are not copied, but hard-linked by default.
This means that the same single image file may then appear under different paths.

ginjinn utils filter_size
^^^^^^^^^^^^^^^^^^^^^^^^^

Annotated objects as a whole can by filtered using the ``-x/--min_width``, ``-y/--min_height``, and ``-r/--min_area`` options.
In case of our :ref:`simulated dataset <3-getting_started_sim>`, the following command would discard objects with an area of less than 25 pixels (``-r 25``) and store the resulting, filtered annotations under "shapes_ds/annotations_filtered.json".

.. code-block:: bash

    ginjinn utils filter_size \
        -a shapes_ds/annotations.json \
        -o shapes_ds/annotations_filtered.json \
        -r 25 \
        --task instance-segmentation

Here, the total area inside the segmentation polygon(s) of an object is used as filter criterion.
In contrast, the area inside its bounding box would be used if ``--task bbox-detection`` is specified.

In case of instance segmentation, objects are allowed to consist of multiple disjunct fragments.
The latter can be filtered using ``-f/--min_fragment_area``, without necessarily discarding the whole object.
This option can, for example, be useful to clean the results of segmentation refinement.

NOTE: Since ``ginjinn utils filter_size`` is mainly intended for post-processing of prediction outputs, only annotations in COCO format are currently supported.


Dataset Merging
---------------

``ginjinn utils merge`` allows two combine multiple datasets into a single new one.
There are two requirements for the datasets to be merged:

#.  They have to be of the same annotation type, i.e. COCO or Pascal VOC.
#.  There must not be duplicated images or image file names. However, it is allowed two merge datasets sharing the same image directory.

The following command could be used to revert the train/validation/test split from :ref:`Getting Started <3-getting_started_sim>`:

.. code-block:: bash

    ginjinn utils filter_size \
        -t COCO \
        -a shapes_ds_split/train/annotations.json \
        -a shapes_ds_split/val/annotations.json \
        -a shapes_ds_split/test/annotations.json \
        -i shapes_ds_split/train/images \
        -i shapes_ds_split/val/images \
        -i shapes_ds_split/test/images \
        -o shapes_ds_merged

Here, the three sub-datasets "train", "val", and "test" are merged and written to "shapes_ds_merged".

In order to save disk space, it may be desirable to use the ``-l/--link_images`` option.
In this case, the input images will not be copied into the output directory, but hard-linked instead.

NOTE: If the image directories supplied as input contain files with the same name or content, an error message is printed.

Dataset Simulation
------------------

To explore GinJinn2's functionality without having suitably formatted data available, artificial datasets can be generated with ``ginjinn simulate shapes``.
This command allows to create noisy images containing annotated circles and triangles of varying size, color, orientation, and number.
In case of simulated COCO datasets, the generated annotations comprise both bounding boxes and segmentation polygons while annotations in Pascal VOC format only contain the former.
For an exemplary application see :ref:`Getting Started <3-getting_started_sim>`.


Dataset Summary
---------------

There are two commands to summarize the contents of a dataset:

#.  ``ginjinn info``: Compact summary.
#.  ``ginjinn utils count``: Image-wise summary.

ginjinn info
^^^^^^^^^^^^

This command prints a short overview about the contents of a dataset.
It lists the number of images in the dataset and the number of annotated objects per category.
Objects are further classified into segmented (#seg) and non-segmented (#bbox) ones.

In case of our :ref:`simulated dataset <3-getting_started_sim>`, the output of ``ginjinn info -a shapes_ds/annotations.json`` would look like this:

.. code-block:: none

    Dataset info for dataset
            ann_path: shapes_ds/annotations.json
            img_dir: /home/user/shapes_ds/images

    # images: 200

    category distribution:
              #seg  #bbox  total
    circle     189      0    189
    triangle   209      0    209
    total      398      0    398

NOTE: Annotations in COCO format usually comprise bounding boxes for segmented objects as well.
These are not included in #bbox.

ginjinn utils count
^^^^^^^^^^^^^^^^^^^

Unlike ``ginjinn info``, ``ginjinn utils count`` counts object occurrences per category and image.
The tabular results are then written to a comma-separated text file (CSV).
Applied to the above dataset (``ginjinn utils count -a shapes_ds/annotations.json -o counts.csv``), the output file "counts.csv" may look may look like this:

.. code-block:: none

    image,circle,triangle
    img_1.jpg,1,2
    img_2.jpg,0,1
    img_3.jpg,0,2
    img_4.jpg,2,1
    img_5.jpg,0,2
    ...
    img_196.jpg,0,1
    img_197.jpg,1,0
    img_198.jpg,0,1
    img_199.jpg,1,1
    img_200.jpg,2,1 


Dataset Visualization
---------------------

Currently, object occurrences are visualized by two commands:

#.  Arbitrary datasets with annotations in COCO or Pascal VOC format can be visualized using the ``ginjinn visualize`` (or abbreviated ``ginjinn vis``) command.
#.  When predicting from a trained object detection model with ``ginjinn predict``, detected objects can be visualized directly using the ``-v/--visualize`` option.

In both cases, Detectron2's visualization capabilities are used to store images overlaid by bounding boxes and segmentation polygons (if applicable).
Each object is further labeled by its category and, in case of ``ginjinn predict``, by its confidence score.
For an exemplary application see :ref:`Getting Started <3-getting_started_sim>`.


Dealing with Nested Image Directories
-------------------------------------

As illustrated in :ref:`Overview <2-overview_ds_formats>`, all images of a GinJinn2-compatible dataset have to be located within the same directory.
While image directories containing sub-directories cannot be used directly, they can be converted to a compatible format using ``ginjinn utils flatten``.
This command can be used in two ways:

#.  If a directory comprising images is passed as sole input, all images within this directory and its sub-directories are recursively collected and copied to a user-defined output directory.
#.  When providing both a (nested) image directory and a COCO annotation file, a new, adjusted annotation file is created as well.

In both cases, slashes in the original image paths are replaced by some reserved character (e.g. "~"), i.e., the former folder hierarchy is encoded by the new filenames:

+ Before flattening
    .. code-block:: none

        images_nested
        ├── Filago
        │   ├── aegaea
        │   │   ├── 1.jpg
        │   │   └── 2.jpg
        │   └── cretensis
        │       ├── 1.jpg
        │       └── 2.jpg
        ├── Lifago
        │   └── dielsii
        │       ├── 1.jpg
        │       └── 2.jpg
        └── Logfia
            └── gallica
                ├── 1.jpg
                └── 2.jpg

+ After flattening
    .. code-block:: none
                
        images
        ├── images_nested~Filago~aegaea~1.jpg
        ├── images_nested~Filago~aegaea~2.jpg
        ├── images_nested~Filago~cretensis~1.jpg
        ├── images_nested~Filago~cretensis~2.jpg
        ├── images_nested~Lifago~dielsii~1.jpg
        ├── images_nested~Lifago~dielsii~2.jpg
        ├── images_nested~Logfia~gallica~1.jpg
        └── images_nested~Logfia~gallica~2.jpg


Sliding-Window Cropping
-----------------------

Due to the limited spatial resolution of common object detection models, smaller objects tend to be less reliably detected than larger ones.
A way to circumvent this problem is to cut the original images into smaller sub-images such that objects become larger in relation to the image size.
To avoid losing objects at the cutting sites, neighboring sub-images should have some overlap.
``ginjinn utils sw_split`` allows to split images and corresponding annotations (optional) into such sliding windows.

For example, ``ginjinn utils sw_split -I shapes_ds -o shapes_sw`` could be used to crop our :ref:`simulated dataset <3-getting_started_sim>` into sliding windows.
If you also want to split your original dataset into train/validation/test datasets with ``ginjinn split``, this should be done before sliding-window cropping.
Otherwise, due to the overlap between adjacent sliding windows, identical image regions may end up in different sub-datasets and thus distort the assessment of the models' generalization capability.
For this reason, the ``-I`` option can also be used to provide input data with an existing train/validation/test split.

Window size and overlap can be specified using the options ``-s/--window_size`` and ``-p/--overlap``, respectively.
Ideally, the overlap between sliding windows should be chosen to be larger than the objects (see :ref:`Sliding-Window Merging <4-toolbox_sw_merge>`).
To ensure that all sub-images are of the same size, these may be filled up with black pixels ("padding") at the borders of an input image.

Usually it is preferred to annotate not only complete objects inside an image, but also incomplete ones at the borders of an image.
Therefore, by default, the output dataset(s) may contain objects trimmed by the sliding-window cropping.
If the user is only interested in complete objects (e.g. for the purpose of measurements), trimmed objects can be discarded using the ``-c/--remove_incomplete`` option.
It is also possible to discard whole sub-images without annotated objects using ``-r/--remove_empty`` if desired. This may save computation time at the expense of prediction accuracy.


.. _4-toolbox_sw_merge:

Sliding-Window Merging
----------------------

Once predictions have been generated for sliding-window cropped data, it may be desirable to project them back onto the original images.
This can be done with ``ginjinn utils sw_merge``, which reconstructs the original images along with object annotations based on annotated sub-images.

The main criterion for merging objects from neighboring sub-images is their Intersection over Union (IoU) inside the window overlap.
Simply spoken, we assess whether two objects occupy more or less the same pixels there.
In case on instance segmentation, an IoU threshold alone may already be sufficient to obtain reasonable results.

As we do not now the exact location of a non-segmented object inside a bounding box, bounding boxes are more difficult to merge.
To mitigate this problem, the IoS ("Intersection over Smaller") can be used as an additional criterion.
It allows to merge two objects if the smaller one is more or less enclosed by the other one.
Here, we consider the objects as a whole rather than only regions inside the window overlap.

Two objects will be merged if at least one of IoU and IoS is above some user-defined threshold (``-u/--iou_threshold``, ``-s/--ios_threshold``, both typically between 0.5 and 1), provided that their total number of overlapping pixels exceeds a certain value.
The latter can be specified using ``-c/--intersection_threshold`` and can prevent objects from being merged because of tiny erratic fragments.

NOTE: Especially in case of bounding boxes, it is easy to think about objects which cannot be handled satisfactorily by either IoU and IoS.
Such problems can be avoided if the overlap between sliding windows is chosen to be larger than the objects.


Train/Validation/Test-Splitting
-------------------------------

See :ref:`Getting Started <3-getting_started_split>`.

