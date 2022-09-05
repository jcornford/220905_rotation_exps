import sys, os
from pathlib import Path

from fastargs import Section, Param, get_current_config
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils
from lib.dann_rnn import EiRNNCell
from lib.base_rnn import RNNCell
from lib.squential import Sequential


Section('train', 'Training Parameters').params(
    batch_size=Param(int, 'batch-size', default=32),
    epochs=Param(int, 'epochs', default=10), 
    lr=Param(float, 'learning-rate', default=0.05),
    seed=Param(int, 'seed', default=777),
    weight_decay=Param(float,'value', default=0),
    )

Section('model', 'Model Parameters').params(
    algorithm=Param(str, 'learning algorithm', default='adamW_gd'),
    is_dann=Param(bool,'network is a dan network', default=False)
)

Section('exp', 'General experiment details').params(
    ckpt_dir=Param(str, 'ckpt-dir', default=""),
    num_workers=Param(int, 'num of CPU workers', default=4),
    use_autocast=Param(bool, 'autocast fp16', default=True),
    log_interval=Param(int, 'log-interval in terms of updates', default=1),
    use_wandb=Param(bool, 'flag to use wandb', default=True),
    wand_project=Param(str, 'project name', default="220905_rotations"),
)
def get_params_to_log_wandb(p):
    """
    Returns dictionary of parameter configurations we log to wanbd
    """
    params_to_log = dict()#use_autocast = p.exp.use_autocast)
    params_to_log.update(p.train.__dict__)
    params_to_log.update(p.model.__dict__)
    return params_to_log

def build_model(p):
    n_inputs = 1
    n_hidden = 10
    if p.model.is_dann:
        cells = [EiRNNCell(n_inputs, n_hidden, ni_i2h=n_hidden//10,
                           ni_h2h=n_hidden//10, nonlinearity=None)]
    else:
        cells = [RNNCell(n_inputs, n_hidden, ni_i2h=n_hidden//10,
                         ni_h2h=n_hidden//10, nonlinearity=None)]
    return Sequential(cells)

if __name__ == "__main__":
    # %%
    p = utils.get_params()
    utils.set_seed_all(p.train.seed)
    if p.exp.use_wandb:
        os.environ['WANDB_DIR'] = str(Path.home()/ "scratch/")
        params_to_log = get_params_to_log_wandb(p)
        run = wandb.init(reinit=True, project=p.exp.wand_project,
                         config=params_to_log)