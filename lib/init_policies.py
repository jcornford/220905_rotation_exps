from cmath import e
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from . import utils
"""
It is expected that you reimplement 

def extra_repr(self):
    r = super().extra_repr()
    return r 

to print out details of the nn.module when you print the model.


Note: Some of the classes use "WeightInitPolicy" in their name, and some just have "Init",
      for simpicity we will move to just using "Init" in future classes
"""
# ------------ Dense Layer Specific ------------  #
class DenseNormalInit:
    """ 
    Initialises a Dense layer's weights (W) from a normal dist,
    and sets bias to 0.

    Note this is more a combination of Lecun init (just fan-in)
    and He init.

    References:
        https://arxiv.org/pdf/1502.01852.pdf
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

    For eg. use numerator=1 for sigmoid, numerator=2 for relu
    """
    def __init__(self, numerator=2):
        self.numerator = numerator

    def init_weights(self, layer):
        nn.init.normal_(layer.W, mean=0, 
                        std=np.sqrt((self.numerator / layer.n_input)))
        nn.init.zeros_(layer.b)
                 
def calc_ln_mu_sigma(mean, var, ex2=None):
    "Given desired mean and var returns ln mu and sigma"
    mu_ln = np.log(mean**2 / np.sqrt(mean**2 + var))
    sigma_ln = np.sqrt(np.log(1 + (var /mean**2)))
    return mu_ln, sigma_ln

class EiDenseWeightInit_WexMean:
    """
    Initialises inhibitory weights to exactly perform the centering operation of Layer Norm.
    
    Sets Wix as copies of the mean row of Wex, Wei is a random vector squashed to sum to 1.  

    Todo: Look at the difference between log normal
    """
    def __init__(self, numerator=2, wex_distribution="lognormal"):
        """
        
        """
        self.numerator = numerator
        self.wex_distribution = wex_distribution

    def init_weights(self, layer):
        # this stems from calculating var(\hat{z}) = d * ne-1/ne * var(wex)E[x^2] when
        # we have set Wix to mean row of Wex and Wei as summing to 1.
        target_std_wex = np.sqrt(self.numerator*layer.ne/(layer.n_input*(layer.ne-1)))
        exp_scale = target_std_wex # The scale parameter, \beta = 1/\lambda = std
        
        if self.wex_distribution =="exponential":
            Wex_np = np.random.exponential(scale=exp_scale, size=(layer.ne, layer.n_input))
            Wei_np = np.random.exponential(scale=exp_scale, size=(layer.ne, layer.ni))
        
        elif self.wex_distribution =="lognormal":
            mu, sigma = calc_ln_mu_sigma(target_std_wex,target_std_wex**2)
            Wex_np = np.random.lognormal(mu, sigma, size=(layer.ne, layer.n_input))
            Wei_np = np.random.lognormal(mu, sigma, size=(layer.ne, layer.ni))
        
        Wei_np /= Wei_np.sum(axis=1, keepdims=True)
        Wix_np = np.ones(shape=(layer.ni,1))*Wex_np.mean(axis=0,keepdims=True)
        layer.Wex.data = torch.from_numpy(Wex_np).float()
        layer.Wix.data = torch.from_numpy(Wix_np).float()
        layer.Wei.data = torch.from_numpy(Wei_np).float()
        nn.init.zeros_(layer.b)

class EiDenseWithShunt_Init(EiDenseWeightInit_WexMean): 
    """
    Initialisation for network with forward equations of:

    Z = (1/c + gamma) * g*\hat(z) +b

    Where:
        c is a constant, that protects from division by a small value
        gamma_k = \sum_j wei_kj * alpha_j \sum_i Wix_ji x_i
        alpha = ln(e^\rho +1)

    Init strategy is to initialise:
        alpha = 1-c/ne E[Wex] E[X], therefore
        rho = ln(e^{(1-c)/ne E[Wex] E[X]} -1)

    Note** alpha is not a parameter anymore, so need to change the forward
    methods!!  

    Assumptions:
        X ~ rectified half normal with variance =1, therefore
            E[x] = 1/sqrt(2*pi)
        E[Wex] is the same as std(Wex) and both are equal to:
            sigma = np.sqrt(self.numerator*layer.ne/(layer.n_input*(layer.ne-1)))
    """
    
    def init_weights(self, layer, c=None):
        super().init_weights(layer)
        if c is None: c_np = (5**0.5-1) /2 # golden ratio 0.618....
        else: c_np = c

        e_wex = np.sqrt(self.numerator*layer.ne/(layer.n_input*(layer.ne-1)))
        e_x   = 1/np.sqrt(2*np.pi)
        rho_np = np.log(np.exp(((1-layer.c)/layer.ne*e_wex*e_x)) -1) # torch softplus is alternative
        
        layer.c.data = torch.from_numpy(c_np).float()
        layer.rho.data = torch.from_numpy(rho_np).float()

class EiDenseWeightInit_WexMean_Groups(EiDenseWeightInit_WexMean):
    """
    To implement: if n_groups !=1, we split Wex and Wix into `n_groups` and uses the mean row
                of the Wex "group" for the corresponding Wix "group". But this is to be decided later
    """
    def __init__(self, n_groups=1):
        """
        n_groups : int
        # we might want a tuple with integers specifying the size of the groups 
        """
        self.n_groups = n_groups
                

# ------------ RNN Specific Initializations ------------
# 1. "Standard" RNNCell Parameter Initializations
# 2. 
# 3. 
# ------------- 1. "Standard" RNNCell Parameter Initializations ----------
class W_NormalInit:
    """ Initialization of cell.W for h2h for RNNCell from a normal dist"""

    def __init__(self, numerator=2):
        self.numerator = numerator

    def init_weights(self, cell):
        nn.init.normal_(cell.W, mean=0, 
                        std=np.sqrt((self.numerator / cell.n_hidden)))

class U_NormalInit:
    """ Initialization of cell.U for i2h for RNNCell from a normal dist"""

    def __init__(self, numerator=2):
        self.numerator = numerator

    def init_weights(self, cell):
        nn.init.normal_(cell.U, mean=0,
                        std=np.sqrt((self.numerator / cell.n_input)))

class W_IdentityInit:
    """
    Identity matrix init of cell.W for h2h for RNNCell:

    see A Simple Way to Initialize Recurrent Networks of Rectified Linear Units
    https://arxiv.org/abs/1504.00941
    """

    @staticmethod
    def init_weights(cell):
        nn.init.eye_(cell.W)

class W_HybridInit_Uniform:
    """
    W = (1-p)*Id + p*W(Uniform Init)
    """
    def __init__(self, p=0.85):
        self.p = p

    def init_weights(self, cell):
        n_hidden = cell.n_hidden
        bound = 1 / math.sqrt(n_hidden)
        W_p_np = np.random.uniform(-bound, bound, size=(n_hidden, n_hidden))
        W_id_np = np.eye(n_hidden)
        W = self.p * W_p_np + (1-self.p) * W_id_np
        device = utils.get_device()
        cell.W.data = torch.from_numpy(W).float().to(device)

class Bias_ZerosInit:
    @staticmethod
    def init_weights(cell):
        nn.init.zeros_(cell.b)

class W_TorchInit:
    """
    Initialization replicating pytorch's initialisation approach:

    All parameters are initialsed as
        init.uniform_(self.bias, -bound, bound)
        bound = 1 / math.sqrt(fan_in)

    Where fan_in is taken to be n_hidden for rnn cells.

    Assumes an RNN cell with W, U & b parameter tensors. Note that RNNCells
    in pytorch have two bias vectors - one for h2h and one for i2h. Wherease
    this init only assumes one.

    ## Init justification:
    I think this is a "good-bug" they have kept around due to empircal
    performance.

    Todo: add / write documentation on justiication / history of this

    e.g.
    https://soumith.ch/files/20141213_gplus_nninit_discussion.htm
    https://github.com/pytorch/pytorch/issues/57109
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L44-L48

    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    """

    def init_weights(self, cell):
        bound = 1 / math.sqrt(cell.n_hidden)
        nn.init.uniform_(cell.W, -bound, bound)
class U_TorchInit:
    """
    See documentation for W_TorchInit
    """

    def init_weights(self, cell):
        bound = 1 / math.sqrt(cell.n_hidden)
        nn.init.uniform_(cell.U, -bound, bound)
class Bias_TorchInit:
    """
    See documentation for W_TorchInit
    """

    def init_weights(self, cell):
        bound = 1 / math.sqrt(cell.n_hidden)
        nn.init.uniform_(cell.b, -bound, bound)

# ------------- Hidden State Initialization ----------

class Hidden_ZerosInit(nn.Module):
    def __init__(self, n_hidden, requires_grad=False):
        """
        Class to reset hidden state, for example between batches.
        To learn this initial hidden state, pass requires_grad = True.
        If requires_grad = False, hidden state will always be reset back to 0s.
        """
        super().__init__()
        self.h0 = nn.Parameter(torch.zeros(n_hidden, 1), requires_grad)

    def reset(self, cell, batch_size):
        # print("Hidden_ZerosInit",batch_size)
        cell.h = self.h0.repeat(1, batch_size)  # Repeat tensor along bath dim.
class EiRNNCell_WeightInitPolicy():
    """
    This weight init policy assumes model with attrs:
    Wex,Wix,Wei,b, where ni >= 1.
    """
    def __init__(self, numerator=1/3):
        """
        2 for he, 1/3 for pytorch, 1 for xavier
        """
        self.numerator = numerator
    
    def init_weights(self,layer):
        # first for U
        # target_std = np.sqrt(2*np.pi/(layer.n_input*(2*np.pi-1)))
        target_std = np.sqrt( (layer.ne/((layer.ne-1))  * (self.numerator/layer.n_input)))
        exp_scale = target_std # The scale parameter, \beta = 1/\lambda = std
        Uex_np = np.random.exponential(scale=exp_scale, size=(layer.ne, layer.n_input))

        if layer.ni == 1: # for example the output layer
            Uix_np = Uex_np.mean(axis=0, keepdims=True)  # not random as only one int
            Uei_np = np.ones(shape=(layer.ne, layer.ni))/layer.ni

        elif layer.ni != 1:
            # We consider wee ~ wie, and the inhib outputs are Wei <- 1/ni
            Uix_np = np.random.exponential(scale=exp_scale, size=(layer.ni, layer.n_input))
            Uei_np = np.ones(shape=(layer.ne, layer.ni))/layer.ni
        else:
            Uix_np, Uei_np = None, None
            raise ValueError('Invalid value for layer.ni, should be a positive integer.')

        layer.Uex.data = torch.from_numpy(Uex_np).float().to('cuda' if torch.cuda.is_available() else 'cpu')
        layer.Uix.data = torch.from_numpy(Uix_np).float().to('cuda' if torch.cuda.is_available() else 'cpu')
        layer.Uei.data = torch.from_numpy(Uei_np).float().to('cuda' if torch.cuda.is_available() else 'cpu')

        # now for W
        target_std = np.sqrt( (layer.ne/((layer.ne-1))  * (self.numerator/layer.n_input)))
        exp_scale = target_std # The scale parameter, \beta = 1/\lambda = std
        Wex_np = np.random.exponential(scale=exp_scale, size=(layer.ne, layer.ne))

        if layer.ni == 1: # for example the output layer
            Wix_np = Wex_np.mean(axis=0,keepdims=True) # not random as only one int
            Wei_np = np.ones(shape = (layer.ne, layer.ni))/layer.ni

        elif layer.ni != 1:
            # We consider wee ~ wie, and the inhib outputs are Wei <- 1/ni
            Wix_np = np.random.exponential(scale=exp_scale, size=(layer.ni, layer.ne))
            Wei_np = np.ones(shape=(layer.ne, layer.ni))/layer.ni
        else:
            Wix_np, Wei_np = None, None
            raise ValueError('Invalid value for layer.ni, should be a positive integer.')

        layer.Wex.data = torch.from_numpy(Wex_np).float().to('cuda' if torch.cuda.is_available() else 'cpu')
        layer.Wix.data = torch.from_numpy(Wix_np).float().to('cuda' if torch.cuda.is_available() else 'cpu')
        layer.Wei.data = torch.from_numpy(Wei_np).float().to('cuda' if torch.cuda.is_available() else 'cpu')

        # finally bias
        nn.init.zeros_(layer.b)

class EiRNNCell_W_InitPolicy():
    """
    This weight init policy assumes model with attrs:
    Wex,Wix,Wei,b, where ni >= 1.

    Todo - update this with the different ideas
    """
    def __init__(self,numerator=1/3):
        """
        2 for he, 1/3 for pytorch, 1 for xavier
        """
        self.numerator = numerator
    
    def init_weights(self,layer):
        # for W
        # target_std = np.sqrt(2*np.pi/(layer.n_input*(2*np.pi-1)))
        target_std_wex = np.sqrt(self.numerator*layer.ne/(layer.n_hidden*(layer.ne-1)))
        exp_scale = target_std_wex # The scale parameter, \beta = 1/\lambda = std
        mu, sigma = calc_ln_mu_sigma(target_std_wex,target_std_wex**2)
        # set proportional like exp
        
        Wex_np = np.random.lognormal(mu, sigma, size=(layer.ne, layer.n_hidden))
        Wei_np = np.random.lognormal(mu, sigma, size=(layer.ne, layer.ni_h2h))
        Wei_np /= Wei_np.sum(axis=1, keepdims=True)
        Wix_np = np.ones(shape=(layer.ni_h2h,1))*Wex_np.mean(axis=0,keepdims=True)
        layer.Wex.data = torch.from_numpy(Wex_np).float()
        layer.Wix.data = torch.from_numpy(Wix_np).float()
        layer.Wei.data = torch.from_numpy(Wei_np).float()
        layer.Wex.data.positive_only = True
        layer.Wix.data.positive_only = True
        layer.Wei.data.positive_only = True

class EiRNNCell_U_InitPolicy():
    """
    This weight init policy assumes model with attrs:
    Wex,Wix,Wei,b, where ni >= 1.

    Todo - update this with the different ideas
    """
    def __init__(self,numerator=1/3):
        """
        2 for he, 1/3 for pytorch, 1 for xavier
        """
        self.numerator = numerator
    
    def init_weights(self,layer):
        # for U
        # target_std = np.sqrt(2*np.pi/(layer.n_input*(2*np.pi-1)))
        target_std = np.sqrt( (layer.ne/((layer.ne-1))  * (self.numerator/layer.n_input)))
        exp_scale = target_std # The scale parameter, \beta = 1/\lambda = std
        Uex_np = np.random.exponential(scale=exp_scale, size=(layer.ne, layer.n_input))

        if layer.ni_i2h == 1: # for example the output layer
            Uix_np = Uex_np.mean(axis=0, keepdims=True)  # not random as only one int
            Uei_np = np.ones(shape=(layer.ne, layer.ni_i2h))/layer.ni_i2h

        elif layer.ni_i2h != 1:
            # We consider wee ~ wie, and the inhib outputs are Wei <- 1/ni
            Uix_np = np.random.exponential(scale=exp_scale, size=(layer.ni_i2h, layer.n_input))
            Uei_np = np.ones(shape=(layer.ne, layer.ni_i2h))/layer.ni_i2h
        else:
            Uix_np, Uei_np = None, None
            raise ValueError('Invalid value for layer.ni, should be a positive integer.')

        layer.Uex.data = torch.from_numpy(Uex_np).float().to('cuda' if torch.cuda.is_available() else 'cpu')
        layer.Uix.data = torch.from_numpy(Uix_np).float().to('cuda' if torch.cuda.is_available() else 'cpu')
        layer.Uei.data = torch.from_numpy(Uei_np).float().to('cuda' if torch.cuda.is_available() else 'cpu')


# -----------------------Song Init---------------------

def calculate_ColumnEi_layer_params(total: int, ratio: int):
    """
    For a ColumnEi model layer_params is a total number of units, and a ratio e.g 20, for 20:1.
    This is a util function to calculate n_e, n_i.
    Args:
        ratio : int
        total : int, total number of units (typically n_input to a layer)
    """
    fraction = total / (ratio+1)
    n_i = int(np.ceil(fraction))
    n_e = int(np.floor(fraction * ratio))
    return n_e, n_i


class ColumnEiCell_W_InitPolicy:
    """
    Class to initiliase weights for column ei RNN cell.

    Positive weights are drawn from an exponential distribution.
    Use D atricices to param the e and i cells.
    """
    def __init__(self, numerator=None, radius=None):
        self.numerator = numerator
        self.spectral_radius = radius

    # @staticmethod
    def init_weights(self, layer):
        """
        todo
        """
        ne = layer.ne_h
        ni = layer.ni_h

        self.numerator = ((2 * np.pi - 1) / (2 * np.pi)) * (ne + (ne ** 2 / ni))

        sigma_we = np.sqrt(1 / self.numerator)
        sigma_wi = (ne / ni) * sigma_we

        We_np = np.random.exponential(scale=sigma_we, size=(layer.n_hidden, ne))
        Wi_np = np.random.exponential(scale=sigma_wi, size=(layer.n_hidden, ni))

        W = np.concatenate([We_np, Wi_np], axis=1)
        if self.spectral_radius is not None: 
            _, v = np.linalg.eig(W)
            rho = np.max(np.real(np.diag(v)))
            W *= (self.spectral_radius / rho)

        layer.W_pos.data = torch.from_numpy(W).float()

        # D matrix (last ni columns are -)
        layer.D_W.data = torch.eye(ne + ni).float()
        layer.D_W.data[:, -ni:] *= -1

        # bias
        # nn.init.zeros_(layer.b)


class ColumnEi_FirstCell_U_InitPolicy:
    """
    This is the weight init that should be used for the first layer a ColumnEi RNN.

    We use the bias term to center the activations, and glorot init to set the weight variance.
        bias  <- layer.n_input * sigma * mnist_mean *-1

    Weights are drawn from an exponential distribution.
    """
    def __init__(self, dataset):
        if dataset == 'MNIST':
            self.pixel_mean = .1307
        elif dataset == 'KMNIST':
            self.pixel_mean = .1918
        elif dataset == 'FashionMNIST':
            self.pixel_mean = .2860
        # print(dataset, self.pixel_mean)

    def init_weights(self, layer):

        # Weights
        # print(layer.n_hidden, layer.n_input)
        sigma = np.sqrt(1/layer.n_input)
        U_np = np.random.exponential(scale=sigma, size=(layer.n_hidden, layer.n_input))

        layer.U_pos.data = torch.from_numpy(U_np).float()

        # D matrix (is all positive)
        layer.D_U.data = torch.eye(layer.n_input).float()

        # bias
        z_mean = layer.n_input * sigma * self.pixel_mean
        nn.init.constant_(layer.b, val=-z_mean)


class ColumnEiCell_U_InitPolicy:
    """
    Weights are drawn from an exponential distribution.
    """
    def init_weights(self, layer):

        ne = layer.ne
        ni = layer.ni

        # Weights
        # print(layer.n_hidden, layer.n_input)
        sigma = np.sqrt(1/layer.n_input)
        U_np = np.random.exponential(scale=sigma, size=(layer.n_hidden, layer.n_input))
        layer.U_pos.data = torch.from_numpy(U_np).float()

        # D matrix (is all positive)
        layer.D_U.data = torch.eye(ne + ni).float()
        layer.D_U.data[:, -ni:] *= -1

        # bias
        nn.init.zeros_(layer.b)


class Hidden_ZerosInit(nn.Module):
    def __init__(self, n_hidden, requires_grad=False):
        """
        Class to reset hidden state, for example between batches.
        To learn this initial hidden state, pass requires_grad = True.
        If requires_grad = False, hidden state will always be reset back to 0s.
        """
        super().__init__()
        self.h0 = nn.Parameter(torch.zeros(n_hidden, 1), requires_grad)

    #         print(self.hidden_init.shape)

    def reset(self, cell, batch_size):
        # print("Hidden_ZerosInit",batch_size)
        cell.h = self.h0.repeat(1, batch_size)  # Repeat tensor along bath dim.

# -------------------------------

class EiRNNCellWithShunt_WeightInitPolicy(EiRNNCell_WeightInitPolicy):
    def init_weights(self, layer):
        super().init_weights(layer)
        # todo
#         a_numpy = np.sqrt((2*np.pi-1)/layer.n_input) * np.ones(shape=layer.alpha.shape)
#         a = torch.from_numpy(a_numpy)
#         alpha_val = torch.log(a)
#         layer.alpha.data = alpha_val.float()

if __name__ == "__main__":
    # In order for imports to work correctly:
    # run in the shell 'export PYTHONPATH=$PYTHONPATH:~/dann-rnns'
    # 
    pass
