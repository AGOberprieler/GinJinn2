.. _installation:

Installation
============

Requirements
------------
#.  Linux operating system (e.g., Debian, Ubuntu)
#.  NVidia GPU (compute capability >= 3; see https://developer.nvidia.com/cuda-gpus)
#.  GPU driver installed (https://www.nvidia.com/en-us/drivers/unix/)

Installation via Conda
----------------------

The recommended way to install GinJinn2 is via Conda, an open-source package management system for Python and R, which also includes facilities for environment management. See the `official installation guide <https://conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_ for further information.

To install Conda, run the following commands in your Linux terminal:

.. code-block:: bash

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh

Once Conda ist installed run the following command to install GinJinn2 (insert your CUDA version, 10.1 should work for most modern GPUs):

.. code-block:: bash

    conda install -c agoberprieler -c pytorch -c conda-forge cudatoolkit=10.1 ginjinn2

In case the installation takes too long or gets stuck, we recommend using the mamba solver, which can be installed via ``conda install -n base conda-libmamba-solver``:

.. code-block:: bash

    conda install -c agoberprieler -c pytorch -c conda-forge cudatoolkit=10.1 ginjinn2 --experimental-solver=libmamba

Finally, test your installation:

.. code-block:: bash

    ginjinn -h
