# GinJinn2

![GitHub](https://img.shields.io/github/license/agoberprieler/ginjinn_development)
[![Documentation Status](https://readthedocs.org/projects/ginjinn2/badge/?version=latest)](https://ginjinn2.readthedocs.io/en/latest/?badge=latest)

GinJinn2 provides a collection of command-line tools for bounding-box object detection and instance segmentation based on [Detectron2](https://github.com/facebookresearch/detectron2).
Besides providing a convenient interface to the latter, GinJinn2 offers several utility functions to facilitate building custom pipelines.

## Documentation

See our Read the Docs [Documentation](https://ginjinn2.readthedocs.io/en/latest/).

## Installation
### Requirements
- Linux Operating System (e.g. Debian, Ubuntu)
- NVidia GPU compatible with CUDA toolkit version >= 9.2 (compute capability >= 3; see https://developer.nvidia.com/cuda-gpus)
- NVidia GPU driver installed (https://www.nvidia.com/en-us/drivers/unix/)

### Installation via Conda
It is recommended to install GinJinn via [Conda](https://docs.conda.io/en/latest/), an open-source package management system for Python and R, which also includes facilities for environment management system. See the [official installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) for further information.

To install Conda, run the following commands in your Linux terminal:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Once Conda ist installed run the following command to install GinJinn2 (insert your CUDA version):
```bash
conda install -c agoberprieler -c conda-forge -c pytorch cudatoolkit=10.1 ginjinn2
```

Test your installation using:
```bash
ginjinn -h
```

## Usage
Make sure to activate your Conda environment via `conda activate MY_ENV_NAME` prior to running any ginjinn command.


`ginjinn` and all of its subcommands provide help pages, which can be displayed using the argument `-h` or `--help`, e.g.

- `ginjinn -h` (get list of all essential GinJinn commands)
- `ginjinn utils -h` (get list of GinJinn's additional utilities)
- `ginjinn utils crop -h` (get usage information for the cropping utility)

The help pages briefly describe basic functionality and command-specific arguments.


### Dataset formats

A (labeled) input dataset should consist of a single image directory containing JPG images at its top level and accompanying annotations. GinJinn2 supports two common annotation formats, [COCO's data format](https://cocodataset.org/#format-data) (one JSON file per dataset), which is also used as output format, and XML files as used by [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) (one file per image). The latter, however, is only supported for bounding-box object detection.

Although not mandatory, it is recommended to place image directory and annotations in a common directory to enable more compact command invocations. If datasets are structured as shown below, the user does not have specify the image directory explicitely. Note that the file names are arbitrarily chosen.

- COCO

  ```
  data
  ├── annotations.json
  └── images
      ├── 1.jpg
      ├── 2.jpg
      └── ...
  ```

- Pascal VOC

  ```
  data
  ├── annotations
  │   ├── 1.xml
  │   ├── 2.xml
  │   └── ...
  └── images
      ├── 1.jpg
      ├── 2.jpg
      └── ...
  ```

In case of nested image directories, `ginjinn utils flatten` helps to convert datasets to an accepted format.

### Train-val-test split
In addition to the dataset for training the model, it is advisable to provide a validation dataset, which can be used to optimize (hyper)parameters and to detect overfitting. A further test dataset, if available, allows to obtain an unbiased evaluation of the final, trained model.

`ginjinn split` can be used to partition a single dataset such that each image along with its annotated objects is assigned to only one of two or three sub-datasets ("train", "val", "test"). Aiming at a balanced split across different object categories, a simple heuristic is used to propose dataset partitions. The generated output has the following structure:

- COCO

  ```
  data
  ├── train
  │   ├── annotations.json
  │   └── images
  ├── val
  │   ├── annotations.json
  │   └── images
  └── test
      ├── annotations.json
      └── images
  ```

- Pascal VOC

  ```
  data
  ├── train
  │   ├── annotations
  │   └── images
  ├── val
  │   ├── annotations
  │   └── images
  └── test
      ├── annotations
      └── images
  ```

### Basic workflow

1. `ginjinn new`

	This command generates a new project directory, which is required for training, evaluation, and prediction. Initially, it only contains an empty output folder and the GinJinn2 configuration file (“ginjinn_config.yaml”), a simple, formatted text file for storing various settings, which can be edited by the user for customization. When executing certain GinJinn2 commands, further data may be written to the project directory. To avoid inconsistencies, it is strongly recommended to keep the configuration file fixed throughout subsequent steps of the analysis.

    The `-t`/`--template` and `-d`/`--data_dir` options allow to automatically specify various settings such that a valid configuration can be created without manually editing the configuration file.

2. `ginjinn train`

    This command trains the model and simultaneously evaluates it using the validation dataset, if available. Checkpoints are saved periodically at predefined intervalls. While the training is running, its progress can be most easily monitored via the "outputs/metrics.pdf" file in your project directory.
  If the training has finished or in case of interruption, it can be resumed with `-r`/`--resume`. The number of iterations as specified in the configuration file can be overwritten with `-n`/`--n_iter`.

3. `ginjinn evaluate`

    This command calculates [COCO evaluation metrics](https://cocodataset.org/#detection-eval) for a trained model using the test dataset.

4. `ginjinn predict`

    This command uses a trained model to predict objects for new, unlabeled images. It provides several optional types of output: a COCO annotation file, object visualization on the original images, and cropped images (segmentation masks) for each predicted object.


Concrete workflows including more complex examples are described in XYZ.

## License

GinJinn2 is released under the [Apache 2.0 license](LICENSE).

## Citation

XYZ
