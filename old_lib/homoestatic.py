import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class ConvHomeostaticMixin():
    def __init__(self, *args, l_mu=.5, l_sig=.5,  **kwargs):
        """
        Set the main homeostatic lambda (l_h) on construction, if passed to update, will override
        
        - alpha dictates the moving average. 
        """
        super().__init__(*args,**kwargs)

    def group_norm(self, x, num_groups):
        
        N,C,H,W = x.size()
        G = num_groups
        assert C % G == 0
        
        x = x.view(N,G,-1)
        # these are of shape N x G x 1
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        
        
        x = x.view(N,C,H,W)
        
        return mean,var
        
    def update(self, layer,l_mu=0.01, l_sig=.5, *args, **kwargs):
        

        self.l_mu = l_mu
        self.l_sig = l_sig
        
        mu_z_layer  = self.group_norm(layer.zhat, layer.zhat.shape[1])[0]   # Instance Norm
        std_z_layer = self.group_norm(layer.z_dot, layer.z_dot.shape[1])[1] # Instance Norm

        # mu_z_layer  = self.group_norm(layer.zhat, 1)[0]   # Layer Norm
        # std_z_layer = self.group_norm(layer.z_dot, 1)[1] # Layer Norm
        
        
        
        
        batch = layer.zhat.shape[0]
        
        layer_loss_mu = (1/batch) * torch.sum((mu_z_layer)**2)
        #layer_loss_sig  = (1/batch) * torch.sum((std_z_layer  - 1)**2)
        
        
            
        h_param_names = ['Wei','i_conv.weight']


        homeostatic_params = [p for k, p in layer.named_parameters() if k in h_param_names]
        
        layer_loss_mu_grad = torch.autograd.grad(layer_loss_mu, 
                                        inputs=homeostatic_params,
                                        only_inputs=True, retain_graph=True)

        
        
        
        with torch.no_grad():
            for i, p in enumerate(homeostatic_params):
                p.grad = self.l_mu * layer_loss_mu_grad[i]

        if torch.isnan(layer.Wei.grad).any() or torch.isnan(layer.i_conv.weight.grad).any():
            print("There are NaN values")
            sys.exit()
        
        # assert (layer.Wex.grad==gradcheck).all() # check we aren't changing grads here

        super().update(layer, *args, **kwargs)




class HomeostaticMixin():
    """
    Homoestatic Mixin for dense layers only
    Todo: subtractive only networks use only sigma?
    """
    def __init__(self, l_mu=1, l_sig=1, l_ce=0):
        """
        Set the main homeostatic lambda (l_h) on construction, if passed to update, will override
        """
        self.l_mu = l_mu
        self.l_sig = l_sig
        self.l_ce = l_ce
        
    def update(self, layer, l_mu=None, l_sig=None, l_ce=None, **kwargs):
        """
        Args:
            lr : learning rate
            l_ce : cross entropy gradient coefficient 
            l_mu : mean hom loss gradient coeff
            l_sig: var hom loss gradient coeff 
        """
        if l_mu is not None: self.l_mu = l_mu
        if l_sig is not None: self.l_sig = l_sig
        if l_ce is not None: self.l_ce = l_ce

        gradcheck = layer.Wex.grad # check we aren't changing this grad
        
        batch = layer.z_hat.shape[1] # watch out for this! z_hat is ne x batch
        mu_z_layer  = layer.z_hat.mean(axis=0, keepdim=True) # 1 x batch
        var_z_layer = layer.z_dot.var(axis=0, keepdim=True) # 1 x batch
        
        layer_loss_mu = (1/batch) * torch.sum((mu_z_layer)**2, dim=1, keepdims=True)
        layer_loss_sig  = (1/batch) * torch.sum((var_z_layer  - 1)**2, dim=1,keepdims=True)
        
        #https://stackoverflow.com/questions/54754153/autograd-grad-for-tensor-in-pytorch
        # Only send the homeostatic loss to these parameters  
        
        # First handle mean homeostasis
        h_param_names_mu = ['Wei','Wix'] # This is where you couple and decouple 
        h_params_mu = [p for k, p in layer.named_parameters() if k in h_param_names_mu]
        h_params_all = h_params_mu[:]
        
        if hasattr(layer,'alpha'): # currently alpha attr only on divisive layers
            h_param_names_std = ['Wei','Wix','alpha']
            h_params_std = [p for k, p in layer.named_parameters() if k in h_param_names_std]
            h_params_all += h_params_std
    
        layer_loss_mu_grad = torch.autograd.grad(layer_loss_mu, 
                                                inputs=h_params_mu,
                                                only_inputs=True, 
                                                retain_graph=True)
        with torch.no_grad():
            for p in set(h_params_all):
                p.grad = self.l_ce * p.grad 

            for i, p in enumerate(h_params_mu):
                p.grad += self.l_mu * layer_loss_mu_grad[i] 
                
        # Next handle std dev homoestatis if layer is divisive
        if hasattr(layer,'alpha'):
            layer_loss_std_grad = torch.autograd.grad(layer_loss_sig, 
                                                    inputs=h_params_std,
                                                    only_inputs=True, 
                                                    retain_graph=False)

            with torch.no_grad():
                for i, p in enumerate(h_params_std):
                    p.grad += self.l_sig * layer_loss_std_grad[i]
                
        if torch.isnan(layer.Wex.grad).any() or torch.isnan(layer.Wex.grad).any():
            print("There are NaN values")
            sys.exit()

        assert (layer.Wex.grad==gradcheck).all() # check we aren't changing grads here
