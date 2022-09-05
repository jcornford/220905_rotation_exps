import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils
from lib.utils import acc_func

from lib.dense_layers import DenseLayer, SGD

from lib.dense_layers import DenseLayer, EiDenseWithShunt, SGD
from lib.init_policies import EiDenseWithShunt_WeightInitPolicy_ICLR, EiConv_WeightInitPolicy, HeConv2d_WeightInitPolicy
from lib.update_policies import DalesANN_cSGD_UpdatePolicy, DalesANN_conv_cSGD_UpdatePolicy

class Flatten(nn.Module):
    """Flattens all but the batch dimension"""
    def __init__(self, input_shape=None):
        """
        Args:
            input_shape: Shape of each batch element, ie. x.shape[1:]
                         optional, used for n_output property.
        """
        super().__init__()

        self.input_shape = input_shape
        self.network_index = None  # this will be set by the Network class
        self.network_key = None  # the layer's key for network's ModuleDict

    def forward(self,x):
        #print(x.shape)
        batch_size=x.shape[0]
        #print(x.reshape(batch_size,-1).shape)
        return x.reshape(batch_size,-1)

    @property
    def output_shape(self):
        if self.input_shape is None:
            return None
        else:
            return np.prod(self.input_shape)

# export
class Dropout(nn.Module):
    """An unneccesary wrap of torch's nn.Dropout"""
    def __init__(self, drop_prob=0.5, input_shape=None):
        super().__init__()
        self.dropout = nn.Dropout(p=drop_prob, inplace=False)
        self.input_shape = input_shape
    def forward(self, x):
        return self.dropout(x)

    @property
    def output_shape(self):
        if self.input_shape is None:return None
        else: return self.input_shape

# export
class MaxPool(nn.Module):
    """
    Like the conv layer, this is a simple wrapper around
    torch.nn.MaxPool2d
    """
    def __init__(self, kernel_size, stride, padding, input_shape=None):
        """
        Args:
            kernel_size, stride, padding - see nn.MaxPool2d docs
            https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

            input_shape: Shape of each batch element, ie. x.shape[1:]
                         optional, used for n_output property.
        """
        super().__init__()
        self.maxpool2d = nn.MaxPool2d(kernel_size, stride, padding)
        self.input_shape = input_shape

        self.network_index = None  # this will be set by the Network class
        self.network_key = None  # the layer's key for network's ModuleDict

    def forward(self, x):
        return self.maxpool2d(x)

    @property
    def output_shape(self):
        if self.input_shape is None:
            return None
        else:
            data = torch.rand(self.input_shape).unsqueeze(0)
            return self.forward(data).shape[1:]

    def __repr__(self):
        """
        Here we are hijacking torch from printing details
        in a clunky way (as it views this as being two children)
        """
        return self.maxpool2d.__repr__()

# export
class ConvLayer(nn.Module):
    """
    Standard Conv2d

    This is a clunky implementation just wrapping Conv2d for similarity with ei conv layers.
    By defining this way network classes and polcies etc can be shared and there should be no
    issues with the training loop.
    """
    def __init__(self, in_channels, out_channels, kernel_size, nonlinearity = F.relu,
                 weight_init_policy=HeConv2d_WeightInitPolicy(), update_policy=SGD(), input_shape=None,
                 conv2d_kwargs={'bias':True, 'stride':1, 'padding':0, 'dilation':1, 'groups':1,'padding_mode':'zeros'}):

        super().__init__()
        self.nonlinearity = nonlinearity
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, **conv2d_kwargs)
        self.input_shape = input_shape

        self.weight_init_policy = weight_init_policy
        self.update_policy = update_policy
        self.network_index = None  # this will be set by the Network class
        self.network_key = None  # the layer's key for network's ModuleDict

    def forward(self, x):
        self.z = self.conv2d(x)
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z
        return self.h

    def update(self, **kwargs):
        self.update_policy.update(self, **kwargs)

    def init_weights(self, **kwargs):
        "Not sure if it is best to code this as be passing self.conv tbh"
        self.weight_init_policy.init_weights(self.conv2d, **kwargs)

    @property
    def output_shape(self):
        if self.input_shape is None:
            return None
        else:
            data = torch.rand(self.input_shape).unsqueeze(0)
            return self.forward(data).shape[1:]

    def extra_repr(self):
        return ""

    def __repr__(self):
        """
        Here we are hijacking torch from printing details
        of the weight init policies
        """

        return self.conv2d.__repr__()

class EiConvLayer(nn.Module):
    """
    """
    def __init__(self, in_channels, e_channels, i_channels, e_kernel_size, i_kernel_size,
                 nonlinearity = F.relu, update_policy=DalesANN_conv_cSGD_UpdatePolicy(),
                 weight_init_policy = EiConv_WeightInitPolicy(),
                 e_param_dict = {'stride':1, 'padding':0, 'dilation':1, 'groups':1, 'bias':False, 'padding_mode':'zeros'},
                 i_param_dict = None, learn_gain_bias=True):
        """
        Args:
        
            i_param_dict : if None, inherits the e_param dict. For now this should be None
            learn_gain_bias : Whether to learn the gain and bias of the normalisation operation
                            Expected that learn_gain_bias Truem and the e/i convs have no bias
                            
        AT THE MOMENT learn_gain_bias does nothing!
                            
        """
        super().__init__()
        self.nonlinearity = nonlinearity
        
        # Fisher corrections are only correct for same e and i filter params 
        # Therefore set i_params to e_params
        if i_param_dict is not None: raise
        else: i_param_dict = e_param_dict
            
        if e_param_dict['bias'] == False and learn_gain_bias == False:
            print("Warning, are you sure you want both conv bias and gain bias False?")
        if e_param_dict['bias'] and learn_gain_bias:
            print("Warning, are you sure you want both conv bias and gain bias True?")
        
        if e_param_dict['bias']: 
            print('For not you were intending to not learn conv biases?')
            raise 
            
        self.e_conv = nn.Conv2d(in_channels, e_channels, e_kernel_size, **e_param_dict)
        self.i_conv = nn.Conv2d(in_channels, i_channels, i_kernel_size, **i_param_dict)
        
        # inhibitory to excitatory weights for each output activation map
        self.Wei = nn.Parameter(torch.randn(e_channels, i_channels))
        self.alpha = nn.Parameter(torch.ones(size=(i_channels, 1, 1)), requires_grad=True)
        
        self.epsilon = 1e-8 # for adding to gamma_map
    
        # one gain and bias for each filter
        self.g = nn.Parameter(torch.ones(e_channels, 1,1))
        self.b = nn.Parameter(torch.zeros(e_channels, 1,1))
        
        self.update_policy = update_policy
        self.weight_init_policy = weight_init_policy
        
        # assign d (fan_in) for weight init etc 
        self.d = int(np.prod(self.e_conv.weight.shape[1:])) #? shape is out_c, in_c, kernel W, kernel H
                  
    def forward(self, x):
        #print(x.shape)
        self.e_act_map = self.e_conv(x)
        self.i_act_map = self.i_conv(x)
        
        # produce subtractive map
        self.subtractive_map = (self.Wei @ self.i_act_map.permute(2,3,1,0)).permute(3,2,0,1)
        
        # produce a divisve map 
        self.gamma = self.Wei @ (torch.exp(self.alpha) * self.i_act_map).permute(2,3,1,0)
        self.gamma = self.gamma.permute(3,2,0,1) + self.epsilon
        
        self.zhat = self.e_act_map - self.subtractive_map
        self.z_dot = (1/ self.gamma) * self.zhat
        
        self.z = self.g*self.z_dot + self.b
            
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z
        return self.h
    
    def update(self, **kwargs):
        self.update_policy.update(self,**kwargs)
        
    def init_weights(self, **kwargs):
        self.weight_init_policy.init_weights(self, **kwargs)

    def extra_repr(self):
        return "Nonlinearity: "+str(self.nonlinearity.__name__)
    
    def __repr__(self):
        """
        Here we are hijacking torch from printing details 
        of the weight init policies
        
        You should make two reprs , one to orint these detaisl
        """
        return f'e{self.e_conv.__repr__()} \n     i{self.i_conv.__repr__()}'

