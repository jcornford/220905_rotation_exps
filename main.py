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
from lib.sequential import Sequential
from lib.optimization import AdamW, SGD
from lib.init_policies import W_IdentityInit

Section('train', 'Training Parameters').params(
    bs=Param(int, 'batch-size', default=32),
    epochs=Param(int, 'epochs', default=10), 
    seed=Param(int, 'seed', default=777),
    timesteps=Param(int, "n_steps to run forward", default=1),
    n_updates=Param(int, "n updates to train on", default=100),
    )

Section('opt', 'optimiser parameters').params(
    algorithm=Param(str, 'learning algorithm', default='SGD'),
    exponentiated=Param(bool,'eg vs gd', default=False),
    lr=Param(float, 'lr and Wex if dann', default=0.0005),
    lr_wei=Param(float,'lr for Wei if dann', default=0.05),
    lr_wix=Param(float,'lr for Wix if dann', default=0.05),
    global_lr=Param(bool, 'sep lr for bias and gain',default=False),
    lr_b=Param(float,'lr for bias', default=0.05),
    lr_g=Param(float,'lr for gains', default=0.05),
    weight_decay=Param(float,'value', default=0),
)

Section('model', 'Model Parameters').params(
    is_dann=Param(bool,'network is a dan network', default=False),
    n_inputs = Param(int,'', default=1),
    n_hidden = Param(int,'n units in the hidden layer',default=10)
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
    n_inputs = p.model.n_inputs
    n_hidden = p.model.n_hidden
    if p.model.is_dann:
        cells = [EiRNNCell(n_inputs, n_hidden, ni_i2h=n_hidden//10,
                           ni_h2h=n_hidden//10, nonlinearity=None)]
    else:
        cells = [RNNCell(n_inputs, n_hidden, nonlinearity=None, bias=False)]
        cells[0].h2h_init_policy=W_IdentityInit()
    return Sequential(cells)

def get_optimizer(p):
    if p.opt.algorithm.lower() == "sgd":
        return SGD(model.parameters(),
                   lr = p.opt.lr,
                   weight_decay=p.opt.weight_decay,
                   exponentiated_grad=p.opt.exponentiated) 

    elif p.opt.algorithm.lower() == "adamw":
        return AdamW(model.parameters(),
                     lr=p.opt.lr,
                     weight_decay=p.opt.weight_decay,
                     exponentiated_grad=p.opt.exponentiated)
    


def train_batch(p, bi):
    h0 = torch.normal(0,1,(p.model.n_hidden, p.train.bs))
    model[0].h = h0
    model.train()
    x = torch.zeros(size=(p.train.bs, p.model.n_inputs))
    y = None
    # tqdm
    for step in range(p.train.timesteps):
        with torch.no_grad():
            if y is None: y = torch.mm(Q, model[0].h)
            else: y = torch.mm(Q, y)
        model.forward(x)
    loss = torch.linalg.norm(y-model[0].h)
    if p.exp.use_wandb:
        wandb.log({"loss":loss.data, "update":bi})
    loss.backward()
    opt.step()
    # print(bi, loss.data)
    # sum? 
        
def calc_target_rot_mat(shape):
    """
    http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
    """
    A = torch.normal(0,1,shape)
    Q, R = torch.linalg.qr(A)
    return Q



if __name__ == "__main__":
    # %%
    p = utils.get_params()
    utils.set_seed_all(p.train.seed)
    
    if p.exp.use_wandb:
        os.environ['WANDB_DIR'] = str(Path.home()/ "scratch/")
        params_to_log = get_params_to_log_wandb(p)
        run = wandb.init(reinit=True, project=p.exp.wand_project,
                         config=params_to_log)

    model = build_model(p)
    print(model)

    Q = calc_target_rot_mat((p.model.n_hidden, p.model.n_hidden))

    opt = get_optimizer(p)

    for bi in range(p.train.n_updates):
        train_batch(p, bi)

