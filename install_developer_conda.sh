#!/bin/bash
set -e
# The script is supposed to run in an activated conda environment.
# e.g. after 
# conda create -y --name BFMM
# conda activate BFMM
# and in this directory

conda install -c conda-forge python=3 #should install the tested version of python3
conda install -c conda-forge --file requirements.conda
python setup.py develop

