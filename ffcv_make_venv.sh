module purge
module load anaconda/3
module load pytorch/1.8.1
#   pytorch/1.8.1       -> python/3.7/cuda/11.1/cudnn/8.0/pytorch/1.8.1  
module list

VENV_NAME='changeme'

conda create -y -n $VENV_NAME python=3.9 cupy pkg-config compilers libjpeg-turbo opencv numba -c pytorch -c conda-forge
conda activate $VENV_NAME
conda install -c anaconda libstdcxx-ng # we used this because of an error with cv2

pip install torchvision==0.9.1 --no-deps
pip install ffcv
pip install ipython --ignore-installed
pip install ipykernel
pip install fastargs

# grab the allen sdk!
#pip install allensdk
pip install matplotlib pandas scipy #wandb
pip install pillow
pip install wandb

# Orion installations
pip install orion 

echo which ipython
ipython kernel install --user --name=$VENV_NAME



