import sys
import argparse
import random

import numpy as np
import torch

from fastargs import get_current_config

def get_params():
    config = get_current_config()
    if not hasattr(sys, 'ps1'): # if not interactive mode 
        parser = argparse.ArgumentParser(description='fastargs demo')
        config.augment_argparse(parser) # adds cl flags and --config-file field to argparser
        config.collect_argparse_args(parser)
        config.validate(mode='stderr')
        config.summary()  # print summary
    params = config.get()
    return params

def set_seed_all(seed):
    """
    Sets all random states
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_cudnn_flags():
    """Set CuDNN flags for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Returns torch.device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')