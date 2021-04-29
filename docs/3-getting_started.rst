.. _3-getting_started:

Getting Started
===============

Considering an annotated dataset is available, the general GinJinn2 workflow consists of

#.  train(-validation)-test split of the dataset
#.  GinJinn2 project intialization
#.  Project configuration
#.  Model training
#.  Model evaluation
#.  Model application (prediction)

In the following sections this workflow will be illustrated using a simulated dataset.

Preparation
-----------

First, make sure that GinJinn2 is installed (:doc:`install instructions <1-installation>`) and can be called from the terminal:

.. code-block:: bash

    ginjinn -h

Simulation
----------

GinJinn2 ships with a simple dataset simulation utility.
The ``ginjinn simulate shapes`` command generates a dataset in `COCO <https://cocodataset.org/#format-data>`_ or `PASCAL VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ format, comprising images with several triangles and circles.
As usual, you can get the list of possible arguments by executing ``ginjinn simulate shapes -h``.

For testing purposes, we will simulate a dataset called "shapes_ds" (``-o shapes_ds``) with 200 (``-n 200``) COCO-annotated (``-a COCO``) images:

.. code-block:: bash

    ginjinn simulate shapes \
        -o shapes_ds \
        -n 200 \
        -a COCO

The previous command will create a new folder "shapes_ds" in your current working directory.
Inside this folder, there is an "images" directory and an "annotations.json" file.
The "images" folder contains the simulated images;
"annotations.json" stores the corresponding instance-segmentation annotations in COCO format.

**optional**: You can visualize annotations using the ``ginjinn utils vis`` command.
The following command generates instance-segmentation visualizations (``-v segmentation``) for the simulated dataset (``-a shapes_ds/annotations.json``) in a new folder "shapes_ds_vis" (``-o shapes_ds_vis``).

.. code-block:: bash

    ginjinn utils vis \
        -a shapes_ds/annotations.json \
        -o shapes_ds_vis \
        -v segmentation

Since we now have an annotated dataset, we can get started with the GinJinn2 workflow.

1. Train-Validation-Test Split
------------------------------

Splitting datasets into training, validation (sometimes also called "development"), and test datasets is a common practice when working with predictive machine learning models.
The training set, as the name suggests is used to train a model, while the validation ("development") set is used to evaluate the model quality while tuning hyperparameters like, for example, learning rate, batch size, and so on.
Finally, the test set is used to get an unbiased measure of the model performance, since it is neither used for training nor for hyper parameter tuning.


GinJinn2 provides the ``ginjinn split`` command for splitting datasets COCO and PASCAL VOC format.
We will use it to split the simulated data (``-a shapes_ds/annotation.json``) into sub-datasets comprising 60%, 20% (``-v 0.2``), and 20% (``tv 0.2``) of the whole dataset for training, validation and testing of an instance-segmentation (``-d instance-segmentation``) model.
The splits will be written to a new folder "shapes_ds_split" (``-o shapes_ds_split``).
GinJinn2 implements an heuristic for generating those splits, which tries to equally distribute instance from different categories.
Thus, when executing the following command, you will be asked, whether you want to accept the proposed split, or try again.

.. code-block:: bash

    ginjinn split \
        -a shapes_ds/annotations.json \
        -o shapes_ds_split \
        -d instance-segmentation \
        -v 0.2 \
        -t 0.2

After executing the above command, a new folder "shapes_ds_split" will be created, containing the three subfolders "train", "val", and "test".
The subfolders will each contain a subset of the images and corresponding annotations of the whole dataset.

