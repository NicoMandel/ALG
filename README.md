# ALG
African Lovegrass (ALG) classification repository

# Repository Structure
Following the [Good Research Code Handbook](https://goodresearch.dev/)
```ALG
|   .gitignore - top folder gitignore
|   environment.yml- the file for creating the conda environment currently necessary to run the code
|   README.md
|   setup.py - a python package setup file
|
└─── config - where configuration files are sitting that define settings
|
└─── data - where pytorchs dataloader expects data - mainly gitignored for size reasons
|       |       .gitignore - to ensure that the folder exists
|       |
|       └─── images
|       |       | <files with a name>.<and an extension>
|       |       
|       └─── masks
|               | <masks with the **exact same name as the image**>.<and some extension>
|
└─── results - intermediate files and results. mostly gitignored
|       |       .gitignore - to ensure that the folder exists
|
└─── scripts - executable scripts that are intended to run programs. Mainly contain an argument parser and lots of includes from the [src directory](./src/).
|               can always be run with python <script_name>.py --help to see information
|
└─── src - importable code. Should be installed using pip. Can be used by placing <import alg.<filename>> at the top of a file
|       └─── alg
|       |   __init__.py - necessary for installation
|       |   utils.py - general utilities that are used across everywhere. should be first point to import, for consistency across scripts. 
|       |   model.py - model definitions and steps that are used in lightning training
|       |   dataloader.py - dataset and data loading tools for use with pytorch pipeline
|
```

## Install
1. using conda, install from file by running `conda env create -f environment.yml`
2. install `pip` into the conda environment
3. install local files by running `pip install -e .` in the base folder
   