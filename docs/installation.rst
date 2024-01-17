.. _installation:

Installation
============

Requirements
------------
#.  Linux operating system (e.g., Debian, Ubuntu)
#.  NVidia GPU (compute capability >= 3; see https://developer.nvidia.com/cuda-gpus)
#.  GPU driver installed (https://www.nvidia.com/en-us/drivers/unix/)

Installation via Conda/Mamba
----------------------------

The originally recommended way to install GinJinn2 was via Conda, an open-source package management system for Python and R, which also includes facilities for environment management (for further information, see `<https://conda.io/projects/conda/en/latest/index.html>`_). Since installing GinJinn2 via Conda/`Miniconda <https://docs.conda.io/projects/miniconda/en/latest/>`_ has become slow and error-prone, we now recommend using `Mamba <https://mamba.readthedocs.io/en/latest/>`_ (from the `Miniforge <https://github.com/conda-forge/miniforge>`_ distribution), a much faster and also more robust reimplementation of the Conda package manager, whose usage is nearly identical.

To install Mamba, run the following commands in your Linux terminal:

.. code-block:: bash

    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh
    mamba init

Before installing GinJinn2, we recommend creating a new Conda/Mamba environment to avoid possible version conflicts with existing software.
Here, we use Python 3.8; other versions may also work.
The environment to be created is named "gj".

.. code-block:: bash

    mamba create -n gj python=3.8

To activate this environment, run:

.. code-block:: bash

    mamba activate gj

Inside the activated environment, run the following command to install GinJinn2 (insert your CUDA version, 10.1 should work for most modern GPUs):

.. code-block:: bash

    mamba install -c agoberprieler -c pytorch cudatoolkit=10.1 ginjinn2
    
(Note: When using ``conda`` from Miniconda, you also have to specify the conda-forge channel via ``-c conda-forge``.

Finally, test your installation:

.. code-block:: bash

    ginjinn -h

NOTE: The activation step is usually required each time you start a new (pseudo)terminal session, otherwise GinJinn2 will not be found.
Within the activated environment, commands such as ``python3`` or ``ginjinn`` point to programs from your environment, which are separated from system-wide installations.
(Try ``which python3`` before and after the activation step to see the difference.)
