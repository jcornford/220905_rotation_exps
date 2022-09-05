#!/bin/bash
module --force purge
module load python/3.7
module load cuda/11.1/cudnn/8.0 

module list


VENV_NAME='220905_rotation_exps'
VENV_DIR=$HOME'/venvs/'$VENV_NAME

echo 'Building virtual env: '$VENV_NAME' in '$VENV_DIR

mkdir $VENV_DIR
# Add .gitignore to folder just in case it is in a repo
# Ignore everything in the directory apart from the gitignore file
echo "*" > $VENV_DIR/.gitignore
echo "!.gitignore" >> $VENV_DIR/.gitignore

virtualenv $VENV_DIR

source $VENV_DIR'/bin/activate'

# install python packages not provided by modules
pip install torch==1.8.1
pip install torchvision==0.9.1 --no-deps
pip install ipython --ignore-installed
pip install ipykernel

# grab the allen sdk!
#pip install allensdk
pip install numpy
pip install matplotlib pandas scipy 
pip install pillow
pip install wandb

pip install terminaltables fastargs
pip install tqdm

# Orion installations
pip install orion 

# # install bleeding edge orion - bug fix now ignore code changes works
# #pip install git+https://github.com/epistimio/orion.git@develop
# pip install git+https://github.com/epistimio/orion.git@9f3894f3f95c71530249f8149b11beb0f31699bc

# # install grid search plugin
# pip install git+https://github.com/bouthilx/orion.algo.grid_search.git

# set up MILA jupyterlab
echo which ipython
ipython kernel install --user --name=$VENV_NAME

