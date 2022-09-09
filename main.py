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
from lib.init_policies import W_IdentityInit, W_NormalInit, EiDenseWeightInit_WexMean 

Section('train', 'Training Parameters').params(
    bs=Param(int, 'batch-size', default=32),
    epochs=Param(int, 'epochs', default=10), 
    seed=Param(int, 'seed', default=2345232),
    timesteps=Param(int, "n_steps to run forward", default=3),
    n_updates=Param(int, "n updates to train on", default=5000),
    )

Section('opt', 'optimiser parameters').params(
    algorithm=Param(str, 'learning algorithm', default='SGD'),
    exponentiated=Param(bool,'eg vs gd', default=False),
    lr=Param(float, 'lr and Wex if dann', default=0.0005),
    lr_wei=Param(float,'lr for Wei if dann', default=0.0005),
    lr_wix=Param(float,'lr for Wix if dann', default=0.0005),
    global_lr=Param(bool, 'sep lr for bias and gain',default=False),
    lr_b=Param(float,'lr for bias', default=0.05),
    lr_g=Param(float,'lr for gains', default=0.05),
    weight_decay=Param(float,'value', default=0),
) 

Section('model', 'Model Parameters').params(
    is_dann=Param(bool,'network is a dan network', default=True),
    n_inputs = Param(int,'', default=3),
    n_hidden = Param(int,'n units in the hidden layer',default=3)
)

Section('exp', 'General experiment details').params(
    ckpt_dir=Param(str, 'ckpt-dir', default=""),
    num_workers=Param(int, 'num of CPU workers', default=4),
    use_autocast=Param(bool, 'autocast fp16', default=True),
    log_interval=Param(int, 'log-interval in terms of updates', default=1),
    use_wandb=Param(bool, 'flag to use wandb', default=True),
    wand_project=Param(str, 'project name', default="220905_rotations_1"),
)
def get_params_to_log_wandb(p):
    """
    Returns dictionary of parameter configurations we log to wanbd
    """
    params_to_log = dict()#use_autocast = p.exp.use_autocast)
    params_to_log.update(p.train.__dict__)
    params_to_log.update(p.model.__dict__)
    params_to_log.update(p.opt.__dict__)
    return params_to_log

def build_model(p):
    n_inputs = p.model.n_inputs
    n_hidden = p.model.n_hidden
    if p.model.is_dann:
        cells = [EiRNNCell(n_inputs, n_hidden, ni_i2h=1,
                           ni_h2h=1, nonlinearity=None)]
        cells[0].h2h_init_policy=EiDenseWeightInit_WexMean(numerator=1)
        cells[0].ni = cells[0].ni_h2h # code up this properly
        # todo! ther is something wrong
    else:
        cells = [RNNCell(n_inputs, n_hidden, nonlinearity=None, bias=False)]
        #cells[0].h2h_init_policy=W_IdentityInit()
        cells[0].h2h_init_policy=W_NormalInit(numerator=1)
        
    return Sequential(cells)

def get_optimizer(p):
    if p.model.is_dann:
        if p.opt.algorithm.lower() == "sgd":
            params_list = []
            for name, param in model.named_parameters():
                #print(name)
                w_name = name.split(".")[-1]
                #print(w_name)
                if w_name.lower()[0] in(["w", "u"]): 
                    positive_only = True 
                else: positive_only= False
                
                if w_name.endswith("ex"): lr_ = p.opt.lr
                elif w_name.endswith("ix"): lr_ = p.opt.lr_wix
                elif w_name.endswith("ei"): lr_ = p.opt.lr_wei
                else: lr_ = p.opt.lr
    
                params_list.append(
                    {'params':[param], 
                     'lr':lr_,
                     'positive_only':positive_only,
                     'name' : ".".join(name.split(".")[1:])
                     }
                )
            #print(params_list)
            return SGD( params_list, 
                        lr = p.opt.lr,
                        weight_decay=p.opt.weight_decay,
                        exponentiated_grad=p.opt.exponentiated) 
    else: 
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
    h0 = torch.normal(0,1,(p.model.n_hidden, p.train.bs)).cuda()
    model[0].h = h0
    model.train()
    model.zero_grad()
    x = torch.zeros(size=(p.train.bs, p.model.n_inputs)).cuda()
    y = None
    # tqdm
    for step in range(p.train.timesteps):
        #breakpoint()
        with torch.no_grad():
            if y is None: y = torch.mm(Q, model[0].h)
            else: y = torch.mm(Q, y)
        model.forward(x)
        loss = torch.linalg.norm(y-model[0].h)/p.train.timesteps
        loss.backward(retain_graph=True)

    if p.exp.use_wandb:
        wandb.log({"loss":loss.data, "update":bi})
   #breakpoint()
    opt.step()

    
    #print(bi, loss.data)
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
    model.init_weights()
    model.cuda()
    print(model)
    if p.model.is_dann:
        for param in model.parameters():
            param.postive_only = True

    Q = calc_target_rot_mat((p.model.n_hidden, p.model.n_hidden)).cuda()    

    opt = get_optimizer(p)
   
    print(Q)
    print(model[0].W)

    for bi in range(p.train.n_updates):
        train_batch(p, bi)

    print(Q)
    print(model[0].W)
    print(" ***** printing dann weights: ")
    print(model[0].Wex.sign())
    print(model[0].Wix.sign())
    print(model[0].Wei.sign())