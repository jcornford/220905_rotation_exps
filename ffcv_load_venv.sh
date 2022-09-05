#!/bin/bash
module purge
module load anaconda/3
module load pytorch/1.8.1

VENV_NAME='danns-ffcv'
conda activate $VENV_NAME

